#!/usr/bin/env python3
"""Value-aware K=8 consequence probe.

The consequence predictor keeps the K=8 Transformer architecture and receives
either the current action (Cm1) or the recorded action chunk (Cm8).  A local
return auxiliary head is used only during training; evaluation discards it and
feeds predicted E/I into a frozen GT-E/I bridge.  This tests whether ordinary
E/I MSE is ignoring value-relevant consequence modes without bypassing Cm.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.probe_gate2_action_chunk import (
    CHUNK_LENGTH, ChunkTransformerPredictor, _evaluate_bridge_predictions,
    _history, _split,
)
from scripts.probe_gate2_chunk_local_value import _gamma_by_run, _local_return
from scripts.fit_gate1_value_bridge_v2 import TemporalBridge


class ValueAwarePredictor(ChunkTransformerPredictor):
    def __init__(self, history_dim: int, action_dim: int, consequence_dim: int):
        super().__init__(history_dim, action_dim, consequence_dim)
        self.value_head = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, 1))

    def forward(self, history: torch.Tensor, action_sequence: torch.Tensor):
        history_tokens = self.history_projection(history)
        history_tokens = history_tokens + self.history_type + self.history_position[:, :history.shape[1]]
        action_tokens = self.action_projection(action_sequence)
        action_tokens = action_tokens + self.action_type + self.action_position[:, :action_sequence.shape[1]]
        memory = self.context_encoder(torch.cat((history_tokens, action_tokens), dim=1))
        query = (self.query + self.query_position).expand(history.shape[0], -1, -1)
        decoded = self.consequence_decoder(query, memory)
        consequence = self.head(decoded)
        local_value = self.value_head(decoded.mean(dim=1)).squeeze(-1)
        return consequence, local_value


def _stats(x: torch.Tensor, train: torch.Tensor):
    values = x[train].reshape(-1, x.shape[-1]).float()
    return values.mean(dim=0), values.std(dim=0, unbiased=False).clamp_min(1e-6)


def _batches(indices: torch.Tensor, batch_size: int, shuffle: bool, generator: torch.Generator):
    if shuffle:
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
    for start in range(0, indices.numel(), batch_size):
        yield indices[start:start + batch_size]


def _fit_one(
    name: str,
    history: torch.Tensor,
    action_sequence: torch.Tensor,
    effect: torch.Tensor,
    interaction: torch.Tensor,
    local_target: torch.Tensor,
    train: torch.Tensor,
    test: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    value_loss_weight: float,
):
    effect = effect[:, :CHUNK_LENGTH]
    interaction = interaction[:, :CHUNK_LENGTH]
    h_mean, h_scale = _stats(history, train)
    a_mean, a_scale = _stats(action_sequence, train)
    e_mean, e_scale = _stats(effect, train)
    i_mean, i_scale = _stats(interaction, train)
    value_mean = local_target[train].mean()
    value_scale = local_target[train].std(unbiased=False).clamp_min(1e-6)
    model = ValueAwarePredictor(history.shape[-1], action_sequence.shape[-1], effect.shape[-1] + interaction.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_idx = train.nonzero(as_tuple=False).flatten()
    test_idx = test.nonzero(as_tuple=False).flatten()
    if train_idx.numel() == 0 or test_idx.numel() == 0:
        raise ValueError("value-aware probe requires non-empty train and test splits")
    rng = torch.Generator().manual_seed(seed)
    train_losses: List[float] = []
    test_losses: List[float] = []

    def inputs(idx):
        h = ((history[idx] - h_mean.view(1, 1, -1)) / h_scale.view(1, 1, -1)).to(device)
        a = ((action_sequence[idx] - a_mean.view(1, 1, -1)) / a_scale.view(1, 1, -1)).to(device)
        return h, a

    def target_ei(idx):
        e = (effect[idx] - e_mean.view(1, 1, -1)) / e_scale.view(1, 1, -1)
        i = (interaction[idx] - i_mean.view(1, 1, -1)) / i_scale.view(1, 1, -1)
        return torch.cat((e, i), dim=-1).to(device)

    for _epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for idx in _batches(train_idx, batch_size, True, rng):
            prediction, value_prediction = model(*inputs(idx))
            loss_ei = nn.functional.mse_loss(prediction, target_ei(idx))
            value_target = ((local_target[idx] - value_mean) / value_scale).to(device)
            loss_value = nn.functional.mse_loss(value_prediction, value_target)
            loss = loss_ei + value_loss_weight * loss_value
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * idx.numel()
            count += idx.numel()
        train_losses.append(total / max(count, 1))
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for idx in _batches(test_idx, batch_size, False, rng):
                prediction, value_prediction = model(*inputs(idx))
                loss_ei = nn.functional.mse_loss(prediction, target_ei(idx))
                value_target = ((local_target[idx] - value_mean) / value_scale).to(device)
                loss_value = nn.functional.mse_loss(value_prediction, value_target)
                total += float((loss_ei + value_loss_weight * loss_value)) * idx.numel()
                count += idx.numel()
        test_losses.append(total / max(count, 1))

    model.eval()
    predictions = []
    value_predictions = []
    with torch.no_grad():
        for idx in _batches(test_idx, batch_size, False, rng):
            prediction, value_prediction = model(*inputs(idx))
            predictions.append(prediction.cpu())
            value_predictions.append(value_prediction.cpu())
    prediction = torch.cat(predictions, dim=0)
    e_count = effect.shape[-1]
    e_prediction = prediction[..., :e_count] * e_scale.view(1, 1, -1) + e_mean.view(1, 1, -1)
    i_prediction = prediction[..., e_count:] * i_scale.view(1, 1, -1) + i_mean.view(1, 1, -1)
    value_prediction = torch.cat(value_predictions) * value_scale + value_mean
    target_e = effect[test_idx]
    target_i = interaction[test_idx]
    def rmse(actual, predicted, mean):
        error = predicted - actual
        base = torch.sqrt(((actual - mean.view(1, 1, -1)) ** 2).mean())
        value = torch.sqrt((error ** 2).mean())
        return {"rmse": float(value), "mae": float(error.abs().mean()),
                "train_mean_baseline_rmse": float(base),
                "relative_rmse_reduction_vs_train_mean": float(1 - value / base.clamp_min(1e-8))}
    normalization = {
        "history_mean": h_mean, "history_scale": h_scale,
        "action_mean": a_mean, "action_scale": a_scale,
        "effect_mean": e_mean, "effect_scale": e_scale,
        "interaction_mean": i_mean, "interaction_scale": i_scale,
        "value_mean": value_mean, "value_scale": value_scale,
    }
    return model, e_prediction, i_prediction, {
        "name": name, "value_loss_weight": value_loss_weight, "epochs": epochs,
        "train_loss_first": train_losses[0], "train_loss_last": train_losses[-1],
        "test_loss_first": test_losses[0], "test_loss_last": test_losses[-1],
        "test_loss_best": min(test_losses),
        "effect": rmse(target_e, e_prediction, e_mean),
        "interaction": rmse(target_i, i_prediction, i_mean),
        "local_value_head_test_mae": float((value_prediction - local_target[test_idx]).abs().mean()),
        "local_value_head_test_rmse": float(torch.sqrt(((value_prediction - local_target[test_idx]) ** 2).mean())),
    }, normalization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--local-bridge-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--local-bridge-report", type=Path,
        help="JSON report produced with the local bridge checkpoint; defaults to its sibling .json",
    )
    parser.add_argument("--heldout-source-namespace", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.epochs <= 0 or args.batch_size <= 0 or args.value_loss_weight < 0:
        raise ValueError("epochs and batch_size must be positive; value_loss_weight must be non-negative")
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    dataset = torch.load(args.input, map_location="cpu", weights_only=False)
    history = _history(dataset)
    action = dataset["action"].float()
    future_action = dataset["future_action"].float()
    if future_action.shape[1] < CHUNK_LENGTH - 1 or future_action.shape[-1] != action.shape[-1]:
        raise ValueError("future_action incompatible with K=8")
    if "future_valid_mask" in dataset and not bool(dataset["future_valid_mask"][:, :CHUNK_LENGTH].all()):
        raise ValueError("invalid K=8 future action windows")
    action_chunk = torch.cat((action.unsqueeze(1), future_action[:, :CHUNK_LENGTH - 1]), dim=1)
    effect = dataset["effect"].float()
    interaction = dataset["interaction"].float()
    local_target, valid = _local_return(dataset, CHUNK_LENGTH)
    source_namespace = dataset["source_namespace"].long()
    source_run = dataset["source_run"].long()
    episode_id = dataset["episode_id"].long()
    train, test, namespaces = _split(source_namespace, args.heldout_source_namespace)
    train, test = train & valid, test & valid
    if not bool(train.any()) or not bool(test.any()):
        raise ValueError("local-return filtering removed a complete train/test split")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    cm1, cm1_e, cm1_i, cm1_report, cm1_normalization = _fit_one(
        "Cm1", history, action.unsqueeze(1), effect, interaction, local_target,
        train, test, device, args.epochs, args.batch_size, args.seed, args.value_loss_weight,
    )
    cm8, cm8_e, cm8_i, cm8_report, cm8_normalization = _fit_one(
        "Cm8", history, action_chunk, effect, interaction, local_target,
        train, test, device, args.epochs, args.batch_size, args.seed, args.value_loss_weight,
    )
    bridge_checkpoint = torch.load(args.local_bridge_checkpoint, map_location="cpu", weights_only=False)
    expected_schema = "ref2dex.gate2_chunk_local_value_probe.v1"
    if bridge_checkpoint.get("schema") != expected_schema:
        raise ValueError("unexpected local bridge checkpoint schema")
    if bridge_checkpoint.get("heldout_source_namespace") not in (None, args.heldout_source_namespace):
        raise ValueError("local bridge checkpoint namespace mismatch")
    gt_bridge = TemporalBridge(["H", "E", "I"], history.shape[-1], effect.shape[-1], interaction.shape[-1], 0).to(device)
    gt_bridge.load_state_dict(bridge_checkpoint["gt_bridge"])
    gt_norm = bridge_checkpoint["gt_normalization"]
    def bridge_mae(e_pred, i_pred):
        return _evaluate_bridge_predictions(
            gt_bridge, ["H", "E", "I"], history, e_pred, i_pred, local_target,
            source_namespace, source_run, episode_id, test, gt_norm, device, args.batch_size,
        )
    cm1_mae, cm8_mae = bridge_mae(cm1_e, cm1_i), bridge_mae(cm8_e, cm8_i)
    bridge_report_path = args.local_bridge_report or args.local_bridge_checkpoint.with_suffix(".json")
    if not bridge_report_path.exists():
        raise FileNotFoundError(f"local bridge report not found: {bridge_report_path}")
    bridge_report = json.loads(bridge_report_path.read_text())
    if bridge_report.get("schema") != expected_schema:
        raise ValueError("unexpected local bridge report schema")
    if bridge_report.get("heldout_source_namespace") != args.heldout_source_namespace:
        raise ValueError("local bridge report namespace mismatch")
    if bridge_report.get("chunk_length") != CHUNK_LENGTH:
        raise ValueError("local bridge report chunk length mismatch")
    if Path(bridge_report["input"]).resolve() != args.input.resolve():
        raise ValueError("local bridge report input mismatch")
    local_report = {
        "report": str(bridge_report_path.resolve()),
        "h_only_mae": float(bridge_report["h_only_episode_balanced_mae"]),
        "gt_ei_mae": float(bridge_report["gt_ei_episode_balanced_mae"]),
    }
    report = {
        "schema": "ref2dex.gate2_value_aware_chunk_probe.v1",
        "run_id": "P-20261004-gate2-value-aware-chunk-k8-ns%d" % args.heldout_source_namespace,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "input": str(args.input.resolve()),
        "local_bridge_checkpoint": str(args.local_bridge_checkpoint.resolve()),
        "heldout_source_namespace": args.heldout_source_namespace,
        "source_namespace_count": len(namespaces),
        "chunk_length": CHUNK_LENGTH,
        "value_loss_weight": args.value_loss_weight,
        "train_rows": int(train.sum()), "test_rows": int(test.sum()),
        "test_episode_count": len(set(zip(source_namespace[test].tolist(), source_run[test].tolist(), episode_id[test].tolist()))),
        "local_target_definition": "sum_{j=0..7} gamma^j reward[t+j]",
        "predictors": {"Cm1": cm1_report, "Cm8": cm8_report},
        "frozen_bridge": local_report,
        "predicted_ei_bridge_mae": {"Cm1": cm1_mae, "Cm8": cm8_mae,
                                    "Cm1_gain_vs_h": local_report["h_only_mae"] - cm1_mae,
                                    "Cm8_gain_vs_h": local_report["h_only_mae"] - cm8_mae,
                                    "GT_gain_vs_h": local_report["h_only_mae"] - local_report["gt_ei_mae"]},
        "no_physics_replay": True, "no_policy_training": True,
        "auxiliary_head_discarded_at_eval": True, "formal_validation": False,
        "status": "COMPLETED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({
        "schema": report["schema"],
        "input": str(args.input.resolve()),
        "heldout_source_namespace": args.heldout_source_namespace,
        "chunk_length": CHUNK_LENGTH,
        "value_loss_weight": args.value_loss_weight,
        "cm1": cm1.state_dict(), "cm8": cm8.state_dict(),
        "cm1_normalization": cm1_normalization, "cm8_normalization": cm8_normalization,
    }, args.output.with_suffix(".pt"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
