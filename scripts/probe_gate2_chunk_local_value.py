#!/usr/bin/env python3
"""Diagnose whether K=8 consequences explain an eight-step local return.

This is a narrow follow-up to the K=8 action-chunk probe.  It keeps the same
namespace split and bridge architecture, but replaces the long Monte-Carlo
return target with the exact discounted reward sum over the next eight
decision rows.  Only H-only and GT-E/I bridges are fitted; no new consequence
predictor or policy comparison is introduced.
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
from scripts.probe_gate2_action_chunk import CHUNK_LENGTH, _fit_value_bridge, _history, _split


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--heldout-source-namespace", required=True, type=int)
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"schema": report["schema"], "h_bridge": h_bridge.state_dict(), "gt_bridge": gt_bridge.state_dict()}, args.output.with_suffix(".pt"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
