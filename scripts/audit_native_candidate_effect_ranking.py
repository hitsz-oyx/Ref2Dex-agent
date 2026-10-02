#!/usr/bin/env python3
"""Audit whether frozen Cm candidate scores stratify native treatment effects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_executable_contact_opportunity import clustered_interval


def bootstrap_gap(high_values, high_groups, low_values, low_groups, seed=9852):
    keys = np.unique(np.concatenate((high_groups, low_groups)))
    if len(keys) < 2:
        return None
    rng = np.random.default_rng(seed)
    high_totals = {key: high_values[high_groups == key].sum() for key in keys}
    high_counts = {key: int((high_groups == key).sum()) for key in keys}
    low_totals = {key: low_values[low_groups == key].sum() for key in keys}
    low_counts = {key: int((low_groups == key).sum()) for key in keys}
    samples = []
    for _ in range(1000):
        selected = rng.choice(keys, len(keys), replace=True)
        h_count = sum(high_counts[key] for key in selected)
        l_count = sum(low_counts[key] for key in selected)
        if h_count and l_count:
            samples.append(sum(high_totals[key] for key in selected) / h_count -
                           sum(low_totals[key] for key in selected) / l_count)
    return np.quantile(samples, [.05, .95]).tolist() if len(samples) >= 900 else None


def run(args):
    records = [torch.load(p, map_location="cpu", weights_only=False) for p in args.records]
    names = records[0]["policy_names"]
    if names != ["cm", "state_policy", "shuffled", "always_base", "always_fixed", "cm_calibrated"]:
        raise ValueError(f"unexpected panel {names}")
    assignment = torch.cat([r["assignment"] for r in records]).numpy()
    propensity = torch.cat([r["propensity"] for r in records]).numpy()
    outcome = torch.cat([r["outcome"]["score_mm"] for r in records]).numpy()
    motion = torch.cat([r["motion_id"] for r in records]).numpy()
    start = torch.cat([r["start_frame"] for r in records]).numpy()
    groups = np.array([f"{m}/{s}" for m, s in zip(motion, start)])
    episode = np.array(sum((r["episode_id"] for r in records), []))
    diagnostics = torch.cat([r["diagnostics"]["score_mm"] for r in records])
    recommendations = torch.cat([r["recommendations"] for r in records])
    # score_mm has [window, H10, ensemble, candidate]; mode 0 is Cm.
    # The collector flattens valid windows before writing the payload.
    score_panel = diagnostics[:, 0, 0, :]
    recommendation_panel = recommendations[:, 0, :]
    reports = {}
    for arm in (0, 5):
        score = score_panel.gather(-1, recommendation_panel[..., arm, None]).squeeze(-1)
        fixed = score_panel[..., 7]
        relative = (score - fixed).numpy().reshape(-1)
        mask = np.isin(assignment, (arm, 4))
        q25, q75 = np.quantile(relative[mask], [.25, .75])
        strata = {"low": mask & (relative <= q25), "high": mask & (relative >= q75)}
        effects = {}
        for label, take in strata.items():
            weight = (assignment == arm) / propensity - (assignment == 4) / propensity
            values = weight[take] * outcome[take]
            effects[label] = dict(rows=int(take.sum()), arm_rows=int((assignment[take] == arm).sum()),
                                  fixed_rows=int((assignment[take] == 4).sum()),
                                  effect_mm=float(values.mean()) if len(values) else None,
                                  group90=clustered_interval(values, groups[take]) if len(values) else None,
                                  episode90=clustered_interval(values, episode[take]) if len(values) else None)
        gap = effects["high"]["effect_mm"] - effects["low"]["effect_mm"]
        high_weight = (assignment == arm) / propensity - (assignment == 4) / propensity
        reports[names[arm]] = dict(score_q25=float(q25), score_q75=float(q75),
                                   effects=effects, high_minus_low_mm=float(gap),
                                   high_minus_low_group90=bootstrap_gap(
                                       high_weight[strata["high"]] * outcome[strata["high"]], groups[strata["high"]],
                                       high_weight[strata["low"]] * outcome[strata["low"]], groups[strata["low"]]))
    result = dict(schema="ref2dex.native_candidate_effect_ranking_audit.v1",
                  run_status="COMPLETED", label="UNCLEAR", rows=len(assignment),
                  reports=reports, arms=names,
                  scope="known-propensity native treatment-effect stratification; not individual regret, policy proof, or PPO")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
