#!/usr/bin/env python3
"""Gate 2 action-chunk probe with a per-step Transformer decoder.

The probe is deliberately fixed to K=8.  Cm1 receives only the current
action, while Cm8 receives the current action and the next seven recorded
on-policy actions.  Both models use the same temporal architecture and emit
one consequence vector per future step; no H32 output flattening is used.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.fit_gate1_value_bridge_v2 import TemporalBridge


CHUNK_LENGTH = 8


def _history(dataset: Dict[str, object]) -> torch.Tensor:
    return torch.cat((
        dataset["history_state"].float(),
        dataset["history_previous_action"].float(),
        dataset["history_context"].float(),
        dataset["history_progress"].float(),
    ), dim=-1)


def _split(source_namespace: torch.Tensor, heldout: int):
    namespaces = sorted(set(int(value) for value in source_namespace.tolist()))
    if heldout not in namespaces:
        raise ValueError(f"heldout namespace {heldout} not in {namespaces}")
    test = source_namespace == heldout
    if not bool(test.any()) or bool(test.all()):
        raise ValueError("namespace holdout must leave both train and test rows")
    return ~test, test, namespaces


def _stats(x: torch.Tensor, train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    values = x[train].reshape(-1, x.shape[-1]).float()
    return values.mean(dim=0), values.std(dim=0, unbiased=False).clamp_min(1e-6)


def _batches(indices: torch.Tensor, batch_size: int, shuffle: bool, generator: torch.Generator):
    if shuffle:
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
    for start in range(0, indices.numel(), batch_size):
        yield indices[start:start + batch_size]


class ChunkTransformerPredictor(nn.Module):
    """Encode history/action tokens and decode one consequence vector per step."""

    def __init__(self, history_dim: int, action_dim: int, consequence_dim: int,
                 max_action_length: int = CHUNK_LENGTH, d_model: int = 96):
        super().__init__()
        self.d_model = d_model
        self.history_projection = nn.Linear(history_dim, d_model)
        self.action_projection = nn.Linear(action_dim, d_model)
        self.history_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.action_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.history_position = nn.Parameter(torch.zeros(1, 10, d_model))
        self.action_position = nn.Parameter(torch.zeros(1, max_action_length, d_model))
        self.query = nn.Parameter(torch.zeros(1, CHUNK_LENGTH, d_model))
        self.query_position = nn.Parameter(torch.zeros(1, CHUNK_LENGTH, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.consequence_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, consequence_dim))
        nn.init.normal_(self.history_position, std=0.02)
        nn.init.normal_(self.action_position, std=0.02)
        nn.init.normal_(self.query, std=0.02)
        nn.init.normal_(self.query_position, std=0.02)

    def forward(self, history: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        history_tokens = self.history_projection(history)
        history_tokens = history_tokens + self.history_type + self.history_position[:, :history.shape[1]]
        action_tokens = self.action_projection(action_sequence)
        action_tokens = action_tokens + self.action_type + self.action_position[:, :action_sequence.shape[1]]
        memory = self.context_encoder(torch.cat((history_tokens, action_tokens), dim=1))
        query = self.query + self.query_position
        query = query.expand(history.shape[0], -1, -1)
        decoded = self.consequence_decoder(query, memory)
        return self.head(decoded)


def _predictor_metrics(actual: torch.Tensor, predicted: torch.Tensor, mean: torch.Tensor):
    error = predicted - actual
    rmse = torch.sqrt((error ** 2).mean())
    baseline = torch.sqrt(((actual - mean.view(1, 1, -1)) ** 2).mean())
    return {
        "rmse": float(rmse),
        "mae": float(error.abs().mean()),
        "train_mean_baseline_rmse": float(baseline),
        "relative_rmse_reduction_vs_train_mean": float(1.0 - rmse / baseline.clamp_min(1e-8)),
    }


def _fit_predictor(
    name: str,
    history: torch.Tensor,
    action_sequence: torch.Tensor,
    effect: torch.Tensor,
    interaction: torch.Tensor,
    train: torch.Tensor,
    test: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
):
    target_effect = effect[:, :CHUNK_LENGTH]
    target_interaction = interaction[:, :CHUNK_LENGTH]
    history_mean, history_scale = _stats(history, train)
    action_mean, action_scale = _stats(action_sequence, train)
    effect_mean, effect_scale = _stats(target_effect, train)
    interaction_mean, interaction_scale = _stats(target_interaction, train)
    consequence_dim = target_effect.shape[-1] + target_interaction.shape[-1]
    model = ChunkTransformerPredictor(history.shape[-1], action_sequence.shape[-1], consequence_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_idx = train.nonzero(as_tuple=False).flatten()
    test_idx = test.nonzero(as_tuple=False).flatten()
    rng = torch.Generator().manual_seed(seed)
    train_losses: List[float] = []
    test_losses: List[float] = []

    def normalized_targets(idx: torch.Tensor):
        e = (target_effect[idx] - effect_mean.view(1, 1, -1)) / effect_scale.view(1, 1, -1)
        i = (target_interaction[idx] - interaction_mean.view(1, 1, -1)) / interaction_scale.view(1, 1, -1)
        return torch.cat((e, i), dim=-1)

    def normalized_inputs(idx: torch.Tensor):
        h = (history[idx] - history_mean.view(1, 1, -1)) / history_scale.view(1, 1, -1)
        a = (action_sequence[idx] - action_mean.view(1, 1, -1)) / action_scale.view(1, 1, -1)
        return h.to(device), a.to(device)

    for _epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for idx in _batches(train_idx, batch_size, True, rng):
            h, a = normalized_inputs(idx)
            y = normalized_targets(idx).to(device)
            prediction = model(h, a)
            loss = nn.functional.mse_loss(prediction, y)
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
                h, a = normalized_inputs(idx)
                y = normalized_targets(idx).to(device)
                total += float(nn.functional.mse_loss(model(h, a), y)) * idx.numel()
                count += idx.numel()
        test_losses.append(total / max(count, 1))

    predictions = []
    model.eval()
    with torch.no_grad():
        for idx in _batches(test_idx, batch_size, False, rng):
            h, a = normalized_inputs(idx)
            predictions.append(model(h, a).cpu())
    prediction = torch.cat(predictions, dim=0)
    effect_prediction = prediction[..., :effect.shape[-1]] * effect_scale.view(1, 1, -1) + effect_mean.view(1, 1, -1)
    interaction_prediction = prediction[..., effect.shape[-1]:] * interaction_scale.view(1, 1, -1) + interaction_mean.view(1, 1, -1)
    report = {
        "name": name,
        "action_sequence_length": int(action_sequence.shape[1]),
        "epochs": epochs,
        "batch_size": batch_size,
        "train_loss_first": train_losses[0],
        "train_loss_last": train_losses[-1],
        "test_loss_first": test_losses[0],
        "test_loss_last": test_losses[-1],
        "test_loss_best": min(test_losses),
        "effect": _predictor_metrics(target_effect[test_idx], effect_prediction, effect_mean),
        "interaction": _predictor_metrics(target_interaction[test_idx], interaction_prediction, interaction_mean),
    }
    normalization = {
        "history_mean": history_mean, "history_scale": history_scale,
        "action_mean": action_mean, "action_scale": action_scale,
        "effect_mean": effect_mean, "effect_scale": effect_scale,
        "interaction_mean": interaction_mean, "interaction_scale": interaction_scale,
    }
    return model, effect_prediction, interaction_prediction, report, normalization


def _fit_value_bridge(
    blocks: List[str],
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
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
):
    history_mean, history_scale = _stats(history, train)
    target_effect = effect[:, :CHUNK_LENGTH]
    target_interaction = interaction[:, :CHUNK_LENGTH]
    effect_mean, effect_scale = _stats(target_effect, train)
    interaction_mean, interaction_scale = _stats(target_interaction, train)
    target_mean = target[train].mean()
    target_scale = target[train].std(unbiased=False).clamp_min(1e-6)
    aux_mean = auxiliary[train, 2:].mean(dim=0)
    aux_scale = auxiliary[train, 2:].std(dim=0, unbiased=False).clamp_min(1e-6)
    model = TemporalBridge(blocks, history.shape[-1], target_effect.shape[-1], target_interaction.shape[-1], 0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_idx = train.nonzero(as_tuple=False).flatten()
    test_idx = test.nonzero(as_tuple=False).flatten()
    rng = torch.Generator().manual_seed(seed)
    for _epoch in range(epochs):
        model.train()
        for idx in _batches(train_idx, batch_size, True, rng):
            inputs = {
                "H": ((history[idx] - history_mean.view(1, 1, -1)) / history_scale.view(1, 1, -1)).to(device),
                "E": ((target_effect[idx] - effect_mean.view(1, 1, -1)) / effect_scale.view(1, 1, -1)).to(device),
                "I": ((target_interaction[idx] - interaction_mean.view(1, 1, -1)) / interaction_scale.view(1, 1, -1)).to(device),
            }
            prediction = model({key: inputs[key] for key in blocks})
            aux_batch = auxiliary[idx].to(device)
            return_target = ((target[idx] - target_mean) / target_scale).to(device)
            loss_return = nn.functional.mse_loss(prediction[:, 0], return_target)
            loss_binary = nn.functional.binary_cross_entropy_with_logits(prediction[:, 1:3], aux_batch[:, :2])
            loss_cont = nn.functional.mse_loss(prediction[:, 3:], (aux_batch[:, 2:] - aux_mean.to(device)) / aux_scale.to(device))
            loss = loss_return + 0.2 * loss_binary + 0.2 * loss_cont
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    def predict(e_rows: torch.Tensor, i_rows: torch.Tensor):
        values = []
        model.eval()
        with torch.no_grad():
            for start in range(0, test_idx.numel(), batch_size):
                local = slice(start, min(start + batch_size, test_idx.numel()))
                idx = test_idx[local]
                inputs = {
                    "H": ((history[idx] - history_mean.view(1, 1, -1)) / history_scale.view(1, 1, -1)).to(device),
                    "E": ((e_rows[local] - effect_mean.view(1, 1, -1)) / effect_scale.view(1, 1, -1)).to(device),
                    "I": ((i_rows[local] - interaction_mean.view(1, 1, -1)) / interaction_scale.view(1, 1, -1)).to(device),
                }
                values.append(model({key: inputs[key] for key in blocks})[:, 0].cpu() * target_scale + target_mean)
        return torch.cat(values)

    oracle_prediction = predict(target_effect[test_idx], target_interaction[test_idx])
    target_test = target[test_idx]
    groups = list(zip(source_namespace[test_idx].tolist(), source_run[test_idx].tolist(), episode_id[test_idx].tolist()))
    unique_groups = sorted(set(groups))

    def episode_mae(prediction: torch.Tensor):
        errors = (prediction - target_test).abs()
        return float(torch.stack([errors[torch.tensor([group == value for group in groups])].mean() for value in unique_groups]).mean())

    return model, {
        "blocks": blocks,
        "test_episode_count": len(unique_groups),
        "oracle_gt_ei_episode_balanced_mae": episode_mae(oracle_prediction),
        "target_normalization_mean": float(target_mean),
        "target_normalization_scale": float(target_scale),
    }, (history_mean, history_scale, effect_mean, effect_scale, interaction_mean, interaction_scale,
        target_mean, target_scale)


def _evaluate_bridge_predictions(
    bridge: nn.Module,
    blocks: List[str],
    history: torch.Tensor,
    effect_test_aligned: torch.Tensor,
    interaction_test_aligned: torch.Tensor,
    target: torch.Tensor,
    source_namespace: torch.Tensor,
    source_run: torch.Tensor,
    episode_id: torch.Tensor,
    test: torch.Tensor,
    normalization,
    device: torch.device,
    batch_size: int,
):
    (history_mean, history_scale, effect_mean, effect_scale, interaction_mean, interaction_scale,
     target_mean, target_scale) = normalization
    test_idx = test.nonzero(as_tuple=False).flatten()
    values = []
    bridge.eval()
    with torch.no_grad():
        for start in range(0, test_idx.numel(), batch_size):
            local = slice(start, min(start + batch_size, test_idx.numel()))
            idx = test_idx[local]
            inputs = {
                "H": ((history[idx] - history_mean.view(1, 1, -1)) / history_scale.view(1, 1, -1)).to(device),
                "E": ((effect_test_aligned[local] - effect_mean.view(1, 1, -1)) / effect_scale.view(1, 1, -1)).to(device),
                "I": ((interaction_test_aligned[local] - interaction_mean.view(1, 1, -1)) / interaction_scale.view(1, 1, -1)).to(device),
            }
            # The return head is trained on standardized targets. Convert back to
            # return units before comparing with the raw target tensor.
            values.append(bridge({key: inputs[key] for key in blocks})[:, 0].cpu() * target_scale + target_mean)
    prediction = torch.cat(values)
    target_test = target[test_idx]
    groups = list(zip(source_namespace[test_idx].tolist(), source_run[test_idx].tolist(), episode_id[test_idx].tolist()))
    unique_groups = sorted(set(groups))
    errors = (prediction - target_test).abs()
    return float(torch.stack([errors[torch.tensor([group == value for group in groups])].mean() for value in unique_groups]).mean())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--heldout-source-namespace", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=24)
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
    history = _history(dataset)
    action = dataset["action"].float()
    future_action = dataset["future_action"].float()
    if future_action.shape[1] < CHUNK_LENGTH - 1:
        raise ValueError("future_action does not contain enough steps for K=8")
    if "future_valid_mask" in dataset and not bool(dataset["future_valid_mask"][:, :CHUNK_LENGTH].all()):
        raise ValueError("K=8 action chunk contains invalid future-action windows")
    if action.shape[-1] != future_action.shape[-1]:
        raise ValueError("current and future action dimensions do not match")
    action_chunk = torch.cat((action.unsqueeze(1), future_action[:, :CHUNK_LENGTH - 1]), dim=1)
    effect = dataset["effect"].float()
    interaction = dataset["interaction"].float()
    target = dataset["return_to_go"].float()
    auxiliary = dataset["episode_auxiliary"].float()
    source_namespace = dataset["source_namespace"].long()
    source_run = dataset["source_run"].long()
    episode_id = dataset["episode_id"].long()
    train, test, namespaces = _split(source_namespace, args.heldout_source_namespace)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    cm1, cm1_e, cm1_i, cm1_report, cm1_norm = _fit_predictor(
        "Cm1", history, action[:, :].unsqueeze(1), effect, interaction, train, test,
        device, args.epochs, args.batch_size, args.seed,
    )
    cm8, cm8_e, cm8_i, cm8_report, cm8_norm = _fit_predictor(
        "Cm8", history, action_chunk, effect, interaction, train, test,
        device, args.epochs, args.batch_size, args.seed,
    )
    h_bridge, h_bridge_meta, h_norm = _fit_value_bridge(
        ["H"], history, effect, interaction, target, auxiliary, source_namespace,
        source_run, episode_id, train, test, device, args.value_epochs,
        args.batch_size, args.seed + 10,
    )
    gt_bridge, gt_bridge_meta, gt_norm = _fit_value_bridge(
        ["H", "E", "I"], history, effect, interaction, target, auxiliary,
        source_namespace, source_run, episode_id, train, test, device,
        args.value_epochs, args.batch_size, args.seed + 11,
    )
    h_mae = _evaluate_bridge_predictions(
        h_bridge, ["H"], history, effect[test, :CHUNK_LENGTH], interaction[test, :CHUNK_LENGTH],
        target, source_namespace, source_run, episode_id, test, h_norm, device, args.batch_size,
    )
    gt_mae = gt_bridge_meta["oracle_gt_ei_episode_balanced_mae"]
    cm1_mae = _evaluate_bridge_predictions(
        gt_bridge, ["H", "E", "I"], history, cm1_e, cm1_i, target,
        source_namespace, source_run, episode_id, test, gt_norm, device, args.batch_size,
    )
    cm8_mae = _evaluate_bridge_predictions(
        gt_bridge, ["H", "E", "I"], history, cm8_e, cm8_i, target,
        source_namespace, source_run, episode_id, test, gt_norm, device, args.batch_size,
    )
    gt_gain = h_mae - gt_mae
    value_report = {
        "h_only_episode_balanced_mae": h_mae,
        "gt_ei_episode_balanced_mae": gt_mae,
        "cm1_predicted_ei_episode_balanced_mae": cm1_mae,
        "cm8_predicted_ei_episode_balanced_mae": cm8_mae,
        "gt_gain_vs_h": gt_gain,
        "cm1_gain_vs_h": h_mae - cm1_mae,
        "cm8_gain_vs_h": h_mae - cm8_mae,
        "cm1_fraction_of_gt_gain_preserved": (h_mae - cm1_mae) / max(gt_gain, 1e-8),
        "cm8_fraction_of_gt_gain_preserved": (h_mae - cm8_mae) / max(gt_gain, 1e-8),
        "cm8_minus_cm1_mae": cm8_mae - cm1_mae,
    }
    report = {
        "schema": "ref2dex.gate2_action_chunk_probe.v1",
        "run_id": "P-20261004-gate2-action-chunk-k8-ns%d" % args.heldout_source_namespace,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "input": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
        "device": str(device),
        "chunk_length": CHUNK_LENGTH,
        "heldout_source_namespace": args.heldout_source_namespace,
        "source_namespace_count": len(namespaces),
        "source_namespace_keys": dataset.get("metadata", {}).get("source_namespace_keys"),
        "train_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        "train_episode_count": len(set(zip(source_namespace[train].tolist(), source_run[train].tolist(), episode_id[train].tolist()))),
        "test_episode_count": len(set(zip(source_namespace[test].tolist(), source_run[test].tolist(), episode_id[test].tolist()))),
        "action_contracts": {
            "Cm1": "(H_t,a_t) -> (E,I)_{t+1:t+8}",
            "Cm8": "(H_t,a_t:t+7) -> (E,I)_{t+1:t+8}",
        },
        "decoder": "2-layer Transformer encoder/decoder with one output token per consequence step; no output flattening",
        "no_physics_replay": True,
        "no_policy_training": True,
        "formal_validation": False,
        "predictors": {"Cm1": cm1_report, "Cm8": cm8_report},
        "value": value_report,
        "status": "COMPLETED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({
        "schema": report["schema"],
        "cm1": cm1.state_dict(), "cm8": cm8.state_dict(),
        "h_bridge": h_bridge.state_dict(), "gt_bridge": gt_bridge.state_dict(),
        "cm1_normalization": cm1_norm, "cm8_normalization": cm8_norm,
        "heldout_source_namespace": args.heldout_source_namespace,
    }, args.output.with_suffix(".pt"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
