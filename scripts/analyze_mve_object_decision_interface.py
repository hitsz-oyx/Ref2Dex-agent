#!/usr/bin/env python3
"""Audit a frozen object-projected Cm MVE decision-interface Probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


def cluster_bootstrap(values, arm, cluster, arm_count, draws, seed):
    unique, code = np.unique(cluster, return_inverse=True)
    rng = np.random.default_rng(seed)
    sums = np.zeros((arm_count, len(unique)), dtype=np.float64)
    counts = np.zeros_like(sums)
    for index in range(arm_count):
        selected = arm == index
        sums[index] = np.bincount(code[selected], weights=values[selected], minlength=len(unique))
        counts[index] = np.bincount(code[selected], minlength=len(unique))
    result = np.empty((draws, arm_count), dtype=np.float64)
    for draw in range(draws):
        multiplicity = np.bincount(
            rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        denominator = (counts * multiplicity).sum(-1)
        result[draw] = (sums * multiplicity).sum(-1) / np.maximum(denominator, 1)
    return result


def audit(record: Path, output: Path, draws: int):
    payload = torch.load(record, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.mve_object_decision_interface.v1":
        raise ValueError("MVE decision-interface schema mismatch")
    names = list(payload["arm_names"])
    expected = ["cup", "direct_q", "mve_object", "cm_value", "random"]
    if names != expected:
        raise ValueError(f"arm contract changed: {names}")
    arm = payload["assignment"].numpy().astype(int)
    selected = payload["selected_index"].numpy().astype(int)
    env = payload["env_id"].numpy().astype(int)
    state = payload["state"]
    future = payload["future_state"]
    if payload["future_done"].any() or not torch.isfinite(state).all():
        raise ValueError("invalid pre-state or terminal window")
    if not torch.isfinite(future).all() or not torch.isfinite(payload["future_reward"]).all():
        raise ValueError("nonfinite future outcome")
    height = (future[:, :, 38] - state[:, None, 38]).clamp_min(0) * 1000
    metrics = {
        "height_last3_min_mm": height[:, -3:].amin(-1).numpy(),
        "height_last3_mean_mm": height[:, -3:].mean(-1).numpy(),
        "local_reward_sum": payload["future_reward"].sum(-1).numpy(),
        "contact_last3_fraction": payload["future_contact"].all(-1)[:, -3:].float().mean(-1).numpy(),
        "clear_last3_fraction": (payload["future_clearance"][:, -3:] >= .002).float().mean(-1).numpy(),
    }
    scores = {
        "direct_q": payload["direct_q_scores"].numpy(),
        "mve_object": payload["mve_object_scores"].numpy(),
        "cm_value": payload["cm_value_scores"].numpy(),
    }
    groups = np.asarray([f"{int(m)}:{int(s)}" for m, s in
                         zip(payload["motion_id"], payload["start_frame"])])
    supports = np.bincount(arm, minlength=len(names)).tolist()
    result = {
        "schema": "ref2dex.mve_object_decision_interface_audit.v1",
        "run_status": "COMPLETED",
        "record": str(record.resolve()),
        "record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
        "rows": len(arm),
        "env_clusters": int(np.unique(env).size),
        "arm_names": names,
        "supports": supports,
        "changed_by_arm": {
            name: int(((arm == index) & (selected != 0)).sum())
            for index, name in enumerate(names)
        },
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "mve_continuation": payload.get("mve_continuation", "legacy_value"),
        "metrics": {},
        "intervals": {},
    }
    for metric_index, (metric_name, values) in enumerate(metrics.items()):
        bootstrap = cluster_bootstrap(values, arm, env, len(names), draws, 20261003 + metric_index)
        result["metrics"][metric_name] = {
            name: float(values[arm == index].mean()) for index, name in enumerate(names)
        }
        result["intervals"][metric_name] = {}
        for index, name in enumerate(names):
            delta = bootstrap[:, index] - bootstrap[:, 0]
            result["intervals"][metric_name][name] = {
                "delta_vs_cup_mean": float(delta.mean()),
                "lower90": float(np.percentile(delta, 10)),
                "upper90": float(np.percentile(delta, 90)),
            }
    random = arm == names.index("random")
    result["random_ranking"] = {}
    for name, score in scores.items():
        chosen = score[random, selected[random]]
        result["random_ranking"][name] = {
            "rows": int(random.sum()),
            "score_mean": float(chosen.mean()) if random.any() else None,
            "spearman": {
                metric: (float(spearmanr(chosen, values[random]).statistic)
                         if random.sum() >= 3 and np.std(chosen) > 0 and np.std(values[random]) > 0
                         else None)
                for metric, values in metrics.items()
            },
        }
        result.setdefault("score_panel", {})[name] = {
            "top_minus_cup_mean": float((score.max(-1) - score[:, 0]).mean()),
            "top_beats_cup": int((score.max(-1) > score[:, 0]).sum()),
            "top_counts": np.bincount(score.argmax(-1), minlength=score.shape[1]).tolist(),
        }
    gates = {
        "support_ok": min(supports) >= 24,
        "mve_changed_ok": result["changed_by_arm"]["mve_object"] >= 24,
        "mve_positive_height": result["intervals"]["height_last3_min_mm"]["mve_object"]["lower90"] > 0,
        "mve_positive_reward": result["intervals"]["local_reward_sum"]["mve_object"]["lower90"] > 0,
        "random_ranking_support_ok": int(random.sum()) >= 24,
    }
    result["gates"] = gates
    result["probe_label"] = (
        "PROMISING" if all(gates.values()) else
        "UNCLEAR" if not (gates["support_ok"] and gates["random_ranking_support_ok"])
        else "UNPROMISING"
    )
    result["interpretation"] = (
        "Frozen native decision-interface Probe. mve_object uses Cm predicted object "
        "translation/velocity/contact/events plus current hand/orientation for Q continuation; "
        "no future state, model update, actor training, or final-success claim."
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    args = parser.parse_args()
    audit(args.record, args.output, args.draws)


if __name__ == "__main__":
    main()
