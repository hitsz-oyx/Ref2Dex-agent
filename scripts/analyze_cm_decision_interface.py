#!/usr/bin/env python3
"""Audit the frozen Cm decision-interface candidate-panel Probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def bootstrap_cluster(values, arm, env, arms, seed, draws=2000):
    unique, env_code = np.unique(env, return_inverse=True)
    rng = np.random.default_rng(seed)
    result = np.empty((draws, len(arms)), dtype=np.float64)
    sums = np.zeros((len(arms), len(unique)), dtype=np.float64)
    counts = np.zeros_like(sums)
    for i, a in enumerate(arms):
        mask = arm == a
        sums[i] = np.bincount(env_code[mask], weights=values[mask], minlength=len(unique))
        counts[i] = np.bincount(env_code[mask], minlength=len(unique))
    for b in range(draws):
        sampled = rng.integers(0, len(unique), len(unique))
        multiplicity = np.bincount(sampled, minlength=len(unique))
        denominator = (counts * multiplicity).sum(-1)
        result[b] = (sums * multiplicity).sum(-1) / np.maximum(denominator, 1)
    return result


def analyze(record, output, bootstrap_draws=2000):
    data = torch.load(record, map_location="cpu", weights_only=False)
    if data.get("schema") != "ref2dex.cm_decision_interface.v1":
        raise ValueError("decision-interface schema mismatch")
    names = list(data["arm_names"])
    arm = data["assignment"].numpy()
    env = data["env_id"].numpy()
    if not np.isfinite(data["state"].numpy()).all():
        raise ValueError("nonfinite pre-state")
    if (data["future_done"].any() or not torch.isfinite(data["future_state"]).all()
            or not torch.isfinite(data["future_reward"]).all()):
        raise ValueError("terminal or nonfinite outcome in complete record")
    future_state = data["future_state"].numpy()
    state = data["state"].numpy()
    contact = data["future_contact"].all(-1).numpy()
    clearance = data["future_clearance"].numpy()
    height = np.maximum(future_state[..., 38] - state[:, None, 38], 0.)
    metrics = {
        "height_last3_min_mm": height[..., -3:].min(-1) * 1000,
        "height_last3_mean_mm": height[..., -3:].mean(-1) * 1000,
        "contact_last3_fraction": contact[..., -3:].mean(-1),
        "clear_last3_fraction": (clearance[..., -3:] >= .002).mean(-1),
        "local_reward_sum": data["future_reward"].numpy().sum(-1),
    }
    supports = {name: int((arm == i).sum()) for i, name in enumerate(names)}
    changed_by_arm = {
        name: int(((arm == i) & (data["selected_index"].numpy() != 0)).sum())
        for i, name in enumerate(names)
    }
    score_report = {}
    for key in ("direct_q_scores", "cm_value_scores"):
        scores = data[key].numpy()
        top = scores.max(-1)
        score_report[key] = {
            "top_minus_cup_mean": float((top - scores[:, 0]).mean()),
            "top_beats_cup_count": int((top > scores[:, 0]).sum()),
            "selected_minus_cup_mean_by_arm": {
                names[i]: float((scores[arm == i, data["selected_index"].numpy()[arm == i]]
                                 - scores[arm == i, 0]).mean())
                for i in range(len(names))
            },
        }
    estimates = {}
    intervals = {}
    for key, value in metrics.items():
        raw = {name: float(value[arm == i].mean()) for i, name in enumerate(names)}
        boot = bootstrap_cluster(value, arm, env, list(range(len(names))), 20705, bootstrap_draws)
        delta = {name: boot[:, i] - boot[:, 0] for i, name in enumerate(names)}
        estimates[key] = raw
        intervals[key] = {
            name: {
                "delta_vs_cup_mean": float(np.nanmean(delta[name])),
                "delta_vs_cup_lower90": float(np.nanpercentile(delta[name], 10)),
                "delta_vs_cup_upper90": float(np.nanpercentile(delta[name], 90)),
            }
            for name in names
        }
    support_ok = all(supports[name] >= 24 for name in names)
    changed_ok = changed_by_arm.get("cm_value", 0) >= 24
    # The Probe asks for positive held local utility and a positive height
    # delta versus Cup.  These are screening gates, not validation claims.
    height_cm = intervals["height_last3_min_mm"]["cm_value"]["delta_vs_cup_lower90"]
    reward_cm = intervals["local_reward_sum"]["cm_value"]["delta_vs_cup_lower90"]
    ranking_ok = height_cm > 0 and reward_cm > 0
    label = "PROMISING" if support_ok and changed_ok and ranking_ok else (
        "UNCLEAR" if not (support_ok and changed_ok) else "UNPROMISING")
    result = {
        "schema": "ref2dex.cm_decision_interface_audit.v1",
        "run_status": "COMPLETED",
        "record": str(record.resolve()),
        "record_sha256": __import__("hashlib").sha256(record.read_bytes()).hexdigest(),
        "rows": len(arm), "env_clusters": int(len(np.unique(env))),
        "supports": supports, "changed_by_arm": changed_by_arm,
        "score_report": score_report, "estimates": estimates,
        "cluster_bootstrap_draws": bootstrap_draws, "intervals": intervals,
        "gates": {"support_ok": support_ok, "changed_ok": changed_ok,
                  "ranking_positive_height_and_reward": ranking_ok},
        "probe_label": label,
        "interpretation": "Probe screening only; not a full policy success or stability claim",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap-draws", type=int, default=2000)
    args = p.parse_args()
    analyze(args.record, args.output, args.bootstrap_draws)


if __name__ == "__main__":
    main()
