#!/usr/bin/env python3
"""Audit Gate 1 deltas with checkpoint-namespace cluster bootstrap.

Fit reports use episode-balanced MAE as the primary probe estimand.  This
    post-fit audit treats all episodes from one checkpoint namespace as one
    cluster, which is the conservative uncertainty unit needed for actor-level
Validation.  It never replays a model or changes the primary fit result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch


def _table(report: Dict[str, object], name: str) -> Dict[tuple, Dict[str, object]]:
    variants = report.get("variants", {})
    if name not in variants:
        raise ValueError(f"variant {name} missing from report")
    rows = variants[name].get("heldout_episode_error_table")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"variant {name} has no heldout episode table")
    out = {}
    for row in rows:
        key = (
            int(row["source_namespace"]),
            int(row["source_run"]),
            int(row["episode_id"]),
        )
        if key in out:
            raise ValueError(f"duplicate heldout episode {key}")
        out[key] = row
    return out


def audit(report: Dict[str, object], base_name: str, variant_name: str,
          repeats: int = 10000, seed: int | None = None) -> Dict[str, object]:
    base = _table(report, base_name)
    variant = _table(report, variant_name)
    if set(base) != set(variant):
        raise ValueError("cluster bootstrap arms do not share heldout episodes")
    by_cluster: Dict[int, List[float]] = {}
    for key in sorted(base):
        namespace, _, _ = key
        by_cluster.setdefault(namespace, []).append(
            float(base[key]["test_mae"]) - float(variant[key]["test_mae"])
        )
    clusters = sorted(by_cluster)
    if len(clusters) < 4:
        raise ValueError("actor cluster bootstrap requires at least four source namespaces")
    cluster_delta = torch.tensor([
        sum(by_cluster[run]) / len(by_cluster[run]) for run in clusters
    ], dtype=torch.float64)
    if seed is None:
        seed = int(report.get("seed", 0)) + 7000
    generator = torch.Generator().manual_seed(seed)
    sampled = cluster_delta[torch.randint(
        len(clusters), (repeats, len(clusters)), generator=generator
    )]
    bootstrap = sampled.mean(dim=1)
    quantiles = torch.quantile(
        bootstrap, torch.tensor([0.025, 0.975], dtype=bootstrap.dtype)
    )
    base_mae = sum(float(base[key]["test_mae"]) for key in base) / len(base)
    variant_mae = sum(float(variant[key]["test_mae"]) for key in variant) / len(variant)
    return {
        "base_variant": f"{base_name}->{variant_name}",
        "seed": int(seed),
        "repeats": repeats,
        "cluster_key": "source_namespace (checkpoint namespace)",
        "cluster_count": len(clusters),
        "cluster_ids": clusters,
        "episode_count": len(base),
        "episode_balanced_base_mae": base_mae,
        "episode_balanced_variant_mae": variant_mae,
        "episode_balanced_relative_reduction": (base_mae - variant_mae) / max(base_mae, 1e-8),
        "cluster_mean_delta": float(cluster_delta.mean()),
        "cluster_delta_by_namespace": {str(namespace): float(delta) for namespace, delta in zip(clusters, cluster_delta)},
        "actor_cluster_bootstrap_delta_ci95": [float(quantiles[0]), float(quantiles[1])],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--base", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = []
    for path in args.input:
        report = json.loads(path.read_text())
        result = audit(report, args.base, args.variant, args.repeats)
        result["input"] = str(path.resolve())
        results.append(result)
    output = {
        "schema": "ref2dex.gate1_actor_cluster_bootstrap.v1",
        "base": args.base,
        "variant": args.variant,
        "results": results,
        "formal_validation": False,
        "primary_fit_unchanged": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
