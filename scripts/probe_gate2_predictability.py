#!/usr/bin/env python3
"""Offline Gate 2 probe: predict future consequences from deployable inputs.

This probe keeps the Gate 1 namespace split and asks two linked questions:

1. Can a small model predict the recorded future effect/interaction from H_t,A_t?
2. When those predictions replace the GT E/I inputs of a value bridge, how much
   of the held-out long-return signal remains?

It is a probe, not a formal Gate 2 validation.  It does not run physics or
update a policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

# Support both ``python -m scripts.probe_gate2_predictability`` and direct
# execution as ``python scripts/probe_gate2_predictability.py``.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.fit_gate1_value_bridge_v2 import TemporalBridge


def _features(dataset: Dict[str, object]) -> Tuple[torch.Tensor, torch.Tensor]:
    history = torch.cat((
        dataset["history_state"].float(),
        dataset["history_previous_action"].float(),
        dataset["history_context"].float(),
        dataset["history_progress"].float(),
    ), dim=-1)
    return history, dataset["action"].float()


def _namespace_split(source_namespace: torch.Tensor, heldout: int):
    namespaces = sorted(set(int(x) for x in source_namespace.tolist()))
    if heldout not in namespaces:
        raise ValueError(f"heldout namespace {heldout} not in {namespaces}")
    test = source_namespace == heldout
    if not bool(test.any()) or bool(test.all()):
        raise ValueError("namespace holdout must leave both train and test rows")
    return ~test, test, namespaces


def _stats(x: torch.Tensor, train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    values = x[train].reshape(-1, x.shape[-1]).float()
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, scale


def _sequence_stats(x: torch.Tensor, train: torch.Tensor):
    return _stats(x, train)


class ConsequencePredictor(nn.Module):
    def __init__(self, history_dim: int, action_dim: int, output_dim: int):
        super().__init__()
        self.history = nn.GRU(history_dim, 64, batch_first=True)
        self.action = nn.Sequential(nn.Linear(action_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        _, hidden = self.history(history)
        fused = torch.cat((hidden[-1], self.action(action)), dim=-1)
        return self.head(fused)


def _batches(indices: torch.Tensor, batch_size: int, shuffle: bool, generator: torch.Generator):
    if shuffle:
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
    for start in range(0, indices.numel(), batch_size):
        yield indices[start:start + batch_size]


def _fit_consequence(
    history: torch.Tensor,
    action: torch.Tensor,
    effect: torch.Tensor,
    interaction: torch.Tensor,
    train: torch.Tensor,
    test: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
):
    h_mean, h_scale = _sequence_stats(history, train)
    a_mean, a_scale = _stats(action, train)
    e_mean, e_scale = _sequence_stats(effect, train)
    i_mean, i_scale = _sequence_stats(interaction, train)
    model = ConsequencePredictor(
        history.shape[-1], action.shape[-1],
        effect.shape[1] * effect.shape[2] + interaction.shape[1] * interaction.shape[2],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_idx = train.nonzero(as_tuple=False).flatten()
    rng = torch.Generator().manual_seed(seed)
    history_loss: List[float] = []
    for _epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for idx in _batches(train_idx, batch_size, True, rng):
            h = ((history[idx] - h_mean.view(1, 1, -1)) /
                 h_scale.view(1, 1, -1)).to(device)
            a = ((action[idx] - a_mean) / a_scale).to(device)
            y = torch.cat((
                ((effect[idx] - e_mean.view(1, 1, -1)) / e_scale.view(1, 1, -1)).flatten(1),
                ((interaction[idx] - i_mean.view(1, 1, -1)) /
                 i_scale.view(1, 1, -1)).flatten(1),
            ), dim=-1).to(device)
            prediction = model(h, a)
            loss = nn.functional.mse_loss(prediction, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * idx.numel()
            count += idx.numel()
        history_loss.append(total / max(count, 1))

    model.eval()
    predictions = []
    with torch.no_grad():
        for idx in _batches(test.nonzero(as_tuple=False).flatten(), batch_size, False, rng):
            h = ((history[idx] - h_mean.view(1, 1, -1)) /
                 h_scale.view(1, 1, -1)).to(device)
            a = ((action[idx] - a_mean) / a_scale).to(device)
            predictions.append(model(h, a).cpu())
    flat_prediction = torch.cat(predictions, dim=0)
    test_idx = test.nonzero(as_tuple=False).flatten()
    effect_width = effect.shape[1] * effect.shape[2]
    effect_prediction = flat_prediction[:, :effect_width].reshape_as(effect[test_idx])
    interaction_prediction = flat_prediction[:, effect_width:].reshape_as(interaction[test_idx])
    effect_prediction = effect_prediction * e_scale.view(1, 1, -1) + e_mean.view(1, 1, -1)
    interaction_prediction = interaction_prediction * i_scale.view(1, 1, -1) + i_mean.view(1, 1, -1)

    def metrics(actual: torch.Tensor, predicted: torch.Tensor, mean: torch.Tensor):
        error = predicted - actual
        rmse = torch.sqrt((error ** 2).mean())
        baseline = torch.sqrt(((actual - mean.view(1, 1, -1)) ** 2).mean())
        return {
            "rmse": float(rmse),
            "mae": float(error.abs().mean()),
            "train_mean_baseline_rmse": float(baseline),
            "relative_rmse_reduction_vs_train_mean": float(1.0 - rmse / baseline.clamp_min(1e-8)),
        }

    report = {
        "epochs": epochs,
        "batch_size": batch_size,
        "training_loss_last": history_loss[-1],
        "training_loss_first": history_loss[0],
        "effect": metrics(effect[test_idx], effect_prediction, e_mean),
        "interaction": metrics(interaction[test_idx], interaction_prediction, i_mean),
    }
    normalization = {
        "history_mean": h_mean, "history_scale": h_scale,
        "action_mean": a_mean, "action_scale": a_scale,
        "effect_mean": e_mean, "effect_scale": e_scale,
        "interaction_mean": i_mean, "interaction_scale": i_scale,
    }
    return model, effect_prediction, interaction_prediction, report, normalization


def _fit_value_bridge(
    history: torch.Tensor,
    effect: torch.Tensor,
    interaction: torch.Tensor,
    target: torch.Tensor,
    auxiliary: torch.Tensor,
    source_namespace: torch.Tensor,
    source_run: torch.Tensor,
    episode_id: torch.Tensor,
    train: torch.Tensor,
    test: torch.Tensor,
    predicted_effect: torch.Tensor,
    predicted_interaction: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    baseline_mae: Optional[float],
):
    h_mean, h_scale = _sequence_stats(history, train)
    e_mean, e_scale = _sequence_stats(effect, train)
    i_mean, i_scale = _sequence_stats(interaction, train)
    y_mean = target[train].mean()
    y_scale = target[train].std(unbiased=False).clamp_min(1e-6)
    aux_mean = auxiliary[train, 2:].mean(dim=0)
    aux_scale = auxiliary[train, 2:].std(dim=0, unbiased=False).clamp_min(1e-6)
    model = TemporalBridge(["H", "E", "I"], history.shape[-1], effect.shape[-1], interaction.shape[-1], 0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_idx = train.nonzero(as_tuple=False).flatten()
    rng = torch.Generator().manual_seed(seed)
    for _epoch in range(epochs):
        model.train()
        for idx in _batches(train_idx, batch_size, True, rng):
            blocks = {
                "H": ((history[idx] - h_mean.view(1, 1, -1)) / h_scale.view(1, 1, -1)).to(device),
                "E": ((effect[idx] - e_mean.view(1, 1, -1)) / e_scale.view(1, 1, -1)).to(device),
                "I": ((interaction[idx] - i_mean.view(1, 1, -1)) / i_scale.view(1, 1, -1)).to(device),
            }
            aux_batch = auxiliary[idx].to(device).float()
            return_batch = ((target[idx] - y_mean) / y_scale).to(device)
            prediction = model(blocks)
            loss_return = nn.functional.mse_loss(prediction[:, 0], return_batch)
            loss_binary = nn.functional.binary_cross_entropy_with_logits(prediction[:, 1:3], aux_batch[:, :2])
            loss_cont = nn.functional.mse_loss(prediction[:, 3:], (aux_batch[:, 2:] - aux_mean.to(device)) / aux_scale.to(device))
            loss = loss_return + 0.2 * loss_binary + 0.2 * loss_cont
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    test_idx = test.nonzero(as_tuple=False).flatten()

    def predict_rows(e_rows: torch.Tensor, i_rows: torch.Tensor):
        values = []
        model.eval()
        for start in range(0, test_idx.numel(), batch_size):
            local = slice(start, min(start + batch_size, test_idx.numel()))
            idx = test_idx[local]
            blocks = {
                "H": ((history[idx] - h_mean.view(1, 1, -1)) / h_scale.view(1, 1, -1)).to(device),
                "E": ((e_rows[local] - e_mean.view(1, 1, -1)) / e_scale.view(1, 1, -1)).to(device),
                "I": ((i_rows[local] - i_mean.view(1, 1, -1)) / i_scale.view(1, 1, -1)).to(device),
            }
            with torch.no_grad():
                values.append((model(blocks)[:, 0].cpu() * y_scale + y_mean))
        return torch.cat(values)

    oracle_prediction = predict_rows(effect[test_idx], interaction[test_idx])
    predicted_prediction = predict_rows(predicted_effect, predicted_interaction)
    target_test = target[test_idx]

    groups = list(zip(source_namespace[test_idx].tolist(), source_run[test_idx].tolist(), episode_id[test_idx].tolist()))
    unique_groups = sorted(set(groups))

    def episode_mae(prediction: torch.Tensor):
        errors = (prediction - target_test).abs()
        return float(torch.stack([errors[torch.tensor([g == x for g in groups])].mean() for x in unique_groups]).mean())

    oracle_mae = episode_mae(oracle_prediction)
    predicted_mae = episode_mae(predicted_prediction)
    value_report = {
        "oracle_gt_ei_episode_balanced_mae": oracle_mae,
        "predicted_ei_episode_balanced_mae": predicted_mae,
        "predicted_minus_oracle_mae": predicted_mae - oracle_mae,
        "baseline_h_episode_balanced_mae": baseline_mae,
    }
    if baseline_mae is not None:
        oracle_gain = baseline_mae - oracle_mae
        predicted_gain = baseline_mae - predicted_mae
        value_report.update({
            "oracle_gain_vs_h": oracle_gain,
            "predicted_gain_vs_h": predicted_gain,
            "fraction_of_oracle_gain_preserved": predicted_gain / max(oracle_gain, 1e-8),
        })
    return model, value_report


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--heldout-source-namespace", required=True, type=int)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--value-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.num_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = torch.load(args.input, map_location="cpu", weights_only=False)
    history, action = _features(dataset)
    effect = dataset["effect"].float()
    interaction = dataset["interaction"].float()
    target = dataset["return_to_go"].float()
    auxiliary = dataset["episode_auxiliary"].float()
    source_namespace = dataset["source_namespace"].long()
    source_run = dataset["source_run"].long()
    episode_id = dataset["episode_id"].long()
    train, test, namespaces = _namespace_split(source_namespace, args.heldout_source_namespace)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    baseline_mae = None
    if args.baseline_report is not None:
        baseline_data = json.loads(args.baseline_report.read_text())
        baseline_mae = float(baseline_data["variants"]["V_H"]["test_episode_balanced_mae"])
    predictor, predicted_effect, predicted_interaction, consequence_report, normalization = _fit_consequence(
        history, action, effect, interaction, train, test, device,
        args.epochs, args.batch_size, args.seed,
    )
    value_model, value_report = _fit_value_bridge(
        history, effect, interaction, target, auxiliary, source_namespace, source_run,
        episode_id, train, test, predicted_effect, predicted_interaction, device,
        args.value_epochs, args.batch_size, args.seed + 1, baseline_mae,
    )
    report = {
        "schema": "ref2dex.gate2_predictability_probe.v1",
        "run_id": "P-20261004-gate2-predictability-ns%d" % args.heldout_source_namespace,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "input": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
        "device": str(device),
        "heldout_source_namespace": args.heldout_source_namespace,
        "source_namespace_count": len(namespaces),
        "source_namespace_keys": dataset.get("metadata", {}).get("source_namespace_keys"),
        "train_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        "train_episode_count": len(set(zip(source_namespace[train].tolist(), source_run[train].tolist(), episode_id[train].tolist()))),
        "test_episode_count": len(set(zip(source_namespace[test].tolist(), source_run[test].tolist(), episode_id[test].tolist()))),
        "input_contract": "(H_t,A_t) -> (E_1:32,I_1:32)",
        "no_physics_replay": True,
        "no_policy_training": True,
        "formal_validation": False,
        "consequence_predictor": consequence_report,
        "value_bridge": value_report,
        "interpretation": "probe only; value preservation is evaluated with a bridge trained on GT E/I",
        "status": "COMPLETED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({
        "schema": report["schema"],
        "predictor": predictor.state_dict(),
        "value_bridge": value_model.state_dict(),
        "normalization": normalization,
        "heldout_source_namespace": args.heldout_source_namespace,
    }, args.output.with_suffix(".pt"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
