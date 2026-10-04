#!/usr/bin/env python3
"""Compare consequence gain against the matched future-action control.

For each held-out episode, this computes
``(MAE_H - MAE_HEI) - (MAE_HF - MAE_HFEI)``.  Positive values mean the
effect/interaction gain exceeds the gain available from future actions.  The
episode and checkpoint-namespace bootstrap intervals are descriptive audits;
they do not turn a pooled split into unseen-actor validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import torch


def _table(report: Dict[str, object], variant: str) -> Dict[tuple, float]:
    rows = report.get("variants", {}).get(variant, {}).get("heldout_episode_error_table")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"variant {variant} has no heldout episode table")
    return {
        (int(row["source_namespace"]), int(row["source_run"]), int(row["episode_id"])):
        float(row["test_mae"])
        for row in rows
    }


def _quantiles(values: torch.Tensor, seed: int, repeats: int) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    sampled = values[torch.randint(len(values), (repeats, len(values)), generator=generator)]
    return [float(value) for value in torch.quantile(
        sampled.mean(dim=1), torch.tensor([0.025, 0.975], dtype=values.dtype)
    )]


def compare(direct: Dict[str, object], control: Dict[str, object], seed: int,
            repeats: int) -> Dict[str, object]:
    direct_h = _table(direct, "V_H")
    direct_hei = _table(direct, "V_HEI")
    control_hf = _table(control, "V_HF")
    control_hfei = _table(control, "V_HFEI")
    keys = set(direct_h)
    if keys != set(direct_hei) or keys != set(control_hf) or keys != set(control_hfei):
        raise ValueError("direct/control reports do not share held-out episode groups")
    keys = sorted(keys)
    delta = torch.tensor([
        (direct_h[key] - direct_hei[key]) - (control_hf[key] - control_hfei[key])
        for key in keys
    ], dtype=torch.float64)
    namespace_ids = sorted(set(key[0] for key in keys))
    namespace_delta = torch.tensor([
        delta[torch.tensor([key[0] == namespace for key in keys])].mean()
        for namespace in namespace_ids
    ], dtype=torch.float64)
    namespace_keys = direct.get("source_namespace_keys")
    if namespace_keys != control.get("source_namespace_keys"):
        raise ValueError("direct/control namespace mappings differ")
    return {
        "episode_count": len(keys),
        "namespace_count": len(namespace_ids),
        "episode_paired_delta_mean": float(delta.mean()),
        "episode_paired_delta_ci95": _quantiles(delta, seed, repeats),
        "namespace_paired_delta_mean": float(namespace_delta.mean()),
        "namespace_paired_delta_ci95": _quantiles(namespace_delta, seed + 1, repeats),
        "namespace_delta_by_id": {
            str(namespace): float(value)
            for namespace, value in zip(namespace_ids, namespace_delta)
        },
        "source_namespace_keys": namespace_keys,
        "estimand": "(H-HEI) - (HF-HFEI) episode/namespace-balanced MAE reduction",
        "formal_validation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", action="append", required=True, type=Path)
    parser.add_argument("--control", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=10000)
    args = parser.parse_args()
    if len(args.direct) != len(args.control):
        raise ValueError("direct and control report counts must match")
    results = []
    for offset, (direct_path, control_path) in enumerate(zip(args.direct, args.control)):
        direct = json.loads(direct_path.read_text())
        control = json.loads(control_path.read_text())
        result = compare(direct, control, int(direct.get("seed", offset)) + 9000, args.repeats)
        result["direct_input"] = str(direct_path.resolve())
        result["control_input"] = str(control_path.resolve())
        results.append(result)
    output = {"schema": "ref2dex.gate1_paired_control_audit.v1", "results": results}
    # When each input is a distinct outer namespace holdout, summarize the
    # held-out-namespace estimand separately from the within-fold episode CI.
    if len(results) >= 2 and all(result["namespace_count"] == 1 for result in results):
        values = torch.tensor(
            [result["namespace_paired_delta_mean"] for result in results], dtype=torch.float64
        )
        output["outer_namespace_bootstrap"] = {
            "namespace_count": len(results),
            "mean_delta": float(values.mean()),
            "ci95": _quantiles(values, 99001, args.repeats),
            "delta_by_holdout": [float(value) for value in values],
            "estimand": "outer leave-one-namespace paired incremental MAE reduction",
        }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
