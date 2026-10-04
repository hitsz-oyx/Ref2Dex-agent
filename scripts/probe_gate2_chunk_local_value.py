#!/usr/bin/env python3
"""Diagnose whether K=8 consequences explain an eight-step local return.

This is a narrow follow-up to the K=8 action-chunk probe.  It keeps the same
namespace split and bridge architecture, but replaces the long Monte-Carlo
return target with the exact discounted reward sum over the next eight
decision rows. H-only and GT-E/I bridges are fitted first; an optional frozen
K=8 predictor checkpoint can then be evaluated on that same local bridge.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.probe_gate2_action_chunk import (
    CHUNK_LENGTH, ChunkTransformerPredictor, _evaluate_bridge_predictions,
    _fit_value_bridge, _history, _split,
)


def _gamma_by_run(dataset):
    runs = dataset.get("metadata", {}).get("runs", [])
    mapping = {}
    for item in runs:
        if "source_run" not in item or "gamma" not in item:
            raise ValueError("metadata.runs must provide source_run and gamma")
        source_run = int(item["source_run"])
        gamma = float(item["gamma"])
        if source_run in mapping and mapping[source_run] != gamma:
            raise ValueError(f"conflicting gamma values for source_run={source_run}")
        mapping[source_run] = gamma
    return mapping


def _local_return(dataset, horizon: int):
    reward = dataset["reward"].float()
    namespace = dataset["source_namespace"].long()
    source_run = dataset["source_run"].long()
    episode_id = dataset["episode_id"].long()
    step = dataset["step"].long()
    gamma_by_run = _gamma_by_run(dataset)
    groups = {}
    for row, key in enumerate(zip(namespace.tolist(), source_run.tolist(), episode_id.tolist())):
        groups.setdefault(key, []).append(row)
    result = torch.zeros_like(reward)
    valid = torch.zeros_like(reward, dtype=torch.bool)
    powers_cache = {run: torch.tensor([gamma ** i for i in range(horizon)], dtype=torch.float32)
                    for run, gamma in gamma_by_run.items()}
    for key, rows in groups.items():
        run = key[1]
        if run not in powers_cache:
            raise ValueError(f"missing gamma for source_run={run}")
        rows = sorted(rows, key=lambda row: int(step[row]))
        steps = [int(step[row]) for row in rows]
        if len(set(steps)) != len(steps):
            raise ValueError(f"duplicate step in episode group {key}")
        if any(right != left + 1 for left, right in zip(steps, steps[1:])):
            raise ValueError(f"non-contiguous steps in episode group {key}")
        values = reward[rows]
        powers = powers_cache[run]
        for position, row in enumerate(rows):
            if position + horizon > len(rows):
                continue
            result[row] = (values[position:position + horizon] * powers).sum()
            valid[row] = True
    return result, valid


def _predict_consequence(checkpoint, key, history, action_sequence, effect_dim,
                         interaction_dim, test, device, batch_size):
    model = ChunkTransformerPredictor(
        history.shape[-1], action_sequence.shape[-1], effect_dim + interaction_dim,
    ).to(device)
    model.load_state_dict(checkpoint[key])
    normalization = checkpoint[f"{key}_normalization"]
    h_mean, h_scale = normalization["history_mean"], normalization["history_scale"]
    a_mean, a_scale = normalization["action_mean"], normalization["action_scale"]
    e_mean, e_scale = normalization["effect_mean"], normalization["effect_scale"]
    i_mean, i_scale = normalization["interaction_mean"], normalization["interaction_scale"]
    test_idx = test.nonzero(as_tuple=False).flatten()
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, test_idx.numel(), batch_size):
            local = slice(start, min(start + batch_size, test_idx.numel()))
            idx = test_idx[local]
            h = ((history[idx] - h_mean.view(1, 1, -1)) / h_scale.view(1, 1, -1)).to(device)
            a = ((action_sequence[idx] - a_mean.view(1, 1, -1)) / a_scale.view(1, 1, -1)).to(device)
            predictions.append(model(h, a).cpu())
    prediction = torch.cat(predictions, dim=0)
    effect_prediction = prediction[..., :effect_dim] * e_scale.view(1, 1, -1) + e_mean.view(1, 1, -1)
    interaction_prediction = prediction[..., effect_dim:] * i_scale.view(1, 1, -1) + i_mean.view(1, 1, -1)
    return effect_prediction, interaction_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--heldout-source-namespace", required=True, type=int)
    parser.add_argument("--predictor-checkpoint", type=Path,
                        help="optional K=8 Cm1/Cm8 checkpoint for local-value evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--value-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.num_threads)
    dataset = torch.load(args.input, map_location="cpu", weights_only=False)
    if "reward" not in dataset or "step" not in dataset:
        raise ValueError("dataset must contain reward and step for local target")
    history = _history(dataset)
    effect = dataset["effect"].float()
    interaction = dataset["interaction"].float()
    action = dataset["action"].float()
    future_action = dataset["future_action"].float()
    if future_action.shape[1] < CHUNK_LENGTH - 1 or future_action.shape[-1] != action.shape[-1]:
        raise ValueError("future_action is incompatible with K=8 action chunks")
    if "future_valid_mask" in dataset and not bool(dataset["future_valid_mask"][:, :CHUNK_LENGTH].all()):
        raise ValueError("K=8 action chunk contains invalid future-action windows")
    action_chunk = torch.cat((action.unsqueeze(1), future_action[:, :CHUNK_LENGTH - 1]), dim=1)
    target, valid = _local_return(dataset, CHUNK_LENGTH)
    auxiliary = dataset["episode_auxiliary"].float()
    source_namespace = dataset["source_namespace"].long()
    source_run = dataset["source_run"].long()
    episode_id = dataset["episode_id"].long()
    train, test, namespaces = _split(source_namespace, args.heldout_source_namespace)
    train = train & valid
    test = test & valid
    if not bool(train.any()) or not bool(test.any()):
        raise ValueError("local-return filtering removed a complete train/test split")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    h_bridge, h_meta, _h_norm = _fit_value_bridge(
        ["H"], history, effect, interaction, target, auxiliary, source_namespace,
        source_run, episode_id, train, test, device, args.value_epochs,
        args.batch_size, args.seed + 10,
    )
    gt_bridge, gt_meta, _gt_norm = _fit_value_bridge(
        ["H", "E", "I"], history, effect, interaction, target, auxiliary,
        source_namespace, source_run, episode_id, train, test, device,
        args.value_epochs, args.batch_size, args.seed + 11,
    )
    h_mae = h_meta["oracle_gt_ei_episode_balanced_mae"]
    gt_mae = gt_meta["oracle_gt_ei_episode_balanced_mae"]
    report = {
        "schema": "ref2dex.gate2_chunk_local_value_probe.v1",
        "run_id": "P-20261004-gate2-chunk-local-value-k8-ns%d" % args.heldout_source_namespace,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "input": str(args.input.resolve()),
        "heldout_source_namespace": args.heldout_source_namespace,
        "source_namespace_count": len(namespaces),
        "chunk_length": CHUNK_LENGTH,
        "target_definition": "sum_{j=0..7} gamma^j reward[t+j] from recorded decision rows",
        "train_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        "valid_local_return_rows": int(valid.sum()),
        "test_episode_count": gt_meta["test_episode_count"],
        "local_target_mean": float(target[valid].mean()),
        "local_target_std": float(target[valid].std(unbiased=False)),
        "h_only_episode_balanced_mae": h_mae,
        "gt_ei_episode_balanced_mae": gt_mae,
        "gt_ei_minus_h_mae": gt_mae - h_mae,
        "gt_ei_relative_mae_change": (gt_mae - h_mae) / max(h_mae, 1e-8),
        "no_physics_replay": True,
        "no_policy_training": True,
        "formal_validation": False,
        "status": "COMPLETED",
    }
    if args.predictor_checkpoint is not None:
        checkpoint = torch.load(args.predictor_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("heldout_source_namespace") != args.heldout_source_namespace:
            raise ValueError("predictor checkpoint heldout namespace does not match local-value run")
        if checkpoint.get("schema") != "ref2dex.gate2_action_chunk_probe.v1":
            raise ValueError("unexpected predictor checkpoint schema")
        cm1_effect, cm1_interaction = _predict_consequence(
            checkpoint, "cm1", history, action.unsqueeze(1), effect.shape[-1],
            interaction.shape[-1], test, device, args.batch_size,
        )
        cm8_effect, cm8_interaction = _predict_consequence(
            checkpoint, "cm8", history, action_chunk, effect.shape[-1],
            interaction.shape[-1], test, device, args.batch_size,
        )
        cm1_mae = _evaluate_bridge_predictions(
            gt_bridge, ["H", "E", "I"], history, cm1_effect, cm1_interaction,
            target, source_namespace, source_run, episode_id, test, _gt_norm,
            device, args.batch_size,
        )
        cm8_mae = _evaluate_bridge_predictions(
            gt_bridge, ["H", "E", "I"], history, cm8_effect, cm8_interaction,
            target, source_namespace, source_run, episode_id, test, _gt_norm,
            device, args.batch_size,
        )
        report["predictor_checkpoint"] = str(args.predictor_checkpoint.resolve())
        report["predicted_ei"] = {
            "cm1_episode_balanced_mae": cm1_mae,
            "cm8_episode_balanced_mae": cm8_mae,
            "cm1_gain_vs_h": h_mae - cm1_mae,
            "cm8_gain_vs_h": h_mae - cm8_mae,
            "gt_gain_vs_h": h_mae - gt_mae,
            "cm1_fraction_of_gt_gain_preserved": (h_mae - cm1_mae) / max(h_mae - gt_mae, 1e-8),
            "cm8_fraction_of_gt_gain_preserved": (h_mae - cm8_mae) / max(h_mae - gt_mae, 1e-8),
            "cm8_minus_cm1_mae": cm8_mae - cm1_mae,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({
        "schema": report["schema"], "h_bridge": h_bridge.state_dict(),
        "gt_bridge": gt_bridge.state_dict(), "h_normalization": _h_norm,
        "gt_normalization": _gt_norm,
    }, args.output.with_suffix(".pt"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
