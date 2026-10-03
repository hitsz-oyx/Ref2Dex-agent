#!/usr/bin/env python3
"""Audit randomized native-candidate advantage records without fitting a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


NAMES = ["expert0", "expert1", "expert2", "expert3", "expert4", "expert5", "base_hold", "fixed_cup"]


def group_bootstrap(values, arm, groups, draws, seed):
    unique, code = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    result = np.full((draws, 8), np.nan)
    sums = np.zeros((8, len(unique)))
    counts = np.zeros_like(sums)
    for k in range(8):
        mask = arm == k
        sums[k] = np.bincount(code[mask], weights=values[mask], minlength=len(unique))
        counts[k] = np.bincount(code[mask], minlength=len(unique))
    for b in range(draws):
        multiplicity = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        denominator = (counts * multiplicity).sum(-1)
        result[b] = (sums * multiplicity).sum(-1) / np.maximum(denominator, 1)
    return result


def policy_bootstrap(values, choice, arm, groups, draws, seed):
    unique, code = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    numerator = np.bincount(code, weights=np.where(arm == choice, values * 8, 0.), minlength=len(unique))
    denominator = np.bincount(code, minlength=len(unique)).astype(float)
    result = []
    for _ in range(draws):
        multiplicity = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        result.append((numerator * multiplicity).sum() / max((denominator * multiplicity).sum(), 1.))
    return np.asarray(result)


def interval(delta):
    return dict(mean=float(np.nanmean(delta)), lower90=float(np.nanpercentile(delta, 5)),
                upper90=float(np.nanpercentile(delta, 95)))


def analyze(records: list[Path], output: Path, draws: int = 10000):
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in records]
    if not payloads or any(p.get("schema") != "ref2dex.cm_candidate_advantage.v2" for p in payloads):
        raise ValueError("candidate advantage schema mismatch")
    required_provenance = ("simulator_seed", "panel_seed", "python_hash_seed", "episode_index")
    if any(any(key not in p for key in required_provenance) for p in payloads):
        raise ValueError("candidate advantage provenance is incomplete")
    def episode_seed(payload):
        prefix = payload["episode_id"][0].split("/", 1)[0]
        if not prefix.startswith("sim"):
            raise ValueError("episode ids do not carry simulator seed")
        return int(prefix[3:])

    if any(p["simulator_seed"] != episode_seed(p) for p in payloads if p["episode_id"]):
        raise ValueError("episode ids do not carry simulator seed")
    checkpoint = payloads[0]["checkpoint_sha256"]
    if any(p["checkpoint_sha256"] != checkpoint for p in payloads):
        raise ValueError("frozen Cm checkpoint drift")
    if any(p["fixed_cup_index"] != 7 or not p["frozen_cm"] or p["optimizer_used"] for p in payloads):
        raise ValueError("frozen execution contract failed")
    assignment = torch.cat([p["assignment"] for p in payloads]).numpy().astype(int)
    motion = torch.cat([p["motion_id"] for p in payloads]).numpy().astype(int)
    start = torch.cat([p["start_frame"] for p in payloads]).numpy().astype(int)
    choice = torch.cat([p["cm_choice"] for p in payloads]).numpy().astype(int)
    score = torch.cat([p["outcome"]["score_mm"] for p in payloads]).numpy()
    first_delta = torch.cat([p["outcome"]["first_delta_mm"] for p in payloads]).numpy()
    retained = torch.cat([p["outcome"]["retained"] for p in payloads]).numpy().astype(float)
    contact = torch.cat([p["outcome"]["contact_last3"] for p in payloads]).numpy().astype(float)
    clearance = torch.cat([p["outcome"]["clearance_last3"] for p in payloads]).numpy().astype(float)
    cm_scores = torch.cat([p["cm_diagnostics"]["score_mm"] for p in payloads]).numpy()
    cm_std = torch.cat([p["cm_diagnostics"]["relative_std_mm"] for p in payloads]).numpy()
    cm_retention = torch.cat([p["cm_diagnostics"]["retention"] for p in payloads]).numpy()
    cm_release = torch.cat([p["cm_diagnostics"]["release"] for p in payloads]).numpy()
    cm_ood = torch.cat([p["cm_ood"] for p in payloads]).numpy().astype(bool)
    cm_candidate_ood = torch.cat([p["cm_candidate_ood"] for p in payloads]).numpy().astype(bool)
    if not np.all(np.isfinite(score)) or not np.all(np.isfinite(cm_scores)):
        raise ValueError("nonfinite labels or predictions")
    if not np.allclose(np.concatenate([p["propensity"].numpy() for p in payloads]), 1 / 8):
        raise ValueError("propensity is not uniform p=1/8")
    if np.any(assignment < 0) or np.any(assignment >= 8):
        raise ValueError("invalid assignment")
    groups = np.asarray([f"{m}/{s}" for m, s in zip(motion, start)])
    episodes = [episode for p in payloads for episode in p["episode_id"]]
    if len(episodes) != len(set(episodes)):
        raise ValueError("episode/window ids are not unique")
    metrics = {"score_mm": score, "first_delta_mm": first_delta, "retained": retained,
               "contact_last3": contact, "clearance_last3": clearance}
    supports = [int((assignment == k).sum()) for k in range(8)]
    episode_support = [len({episodes[i] for i in np.flatnonzero(assignment == k)}) for k in range(8)]
    arm_stats = {}
    intervals = {}
    for key, value in metrics.items():
        boot = group_bootstrap(value, assignment, groups, draws, 12603)
        arm_stats[key] = [float(value[assignment == k].mean()) if (assignment == k).any() else None for k in range(8)]
        intervals[key] = {NAMES[k]: interval(boot[:, k] - boot[:, 7]) for k in range(8)}

    fixed_choice = np.full(len(score), 7)
    raw_top = cm_scores.argmax(-1)
    row = np.arange(len(score))
    uncertainty_fallback = np.where(
        (cm_scores[row, raw_top] - cm_scores[:, 7] > 2 * cm_std[row, raw_top])
        & ~cm_candidate_ood[row, raw_top]
        & (cm_retention[row, raw_top] >= cm_retention[:, 7] - .05)
        & (cm_release[row, raw_top] <= cm_release[:, 7] + .05),
        raw_top, 7,
    )
    policy_reports = {}
    for name, policy in (("cm_selector", choice), ("cm_raw_top", raw_top),
                         ("cm_uncertainty_fallback_2std", uncertainty_fallback),
                         ("fixed_cup", fixed_choice)):
        value = policy_bootstrap(score, policy, assignment, groups, draws, 12604)
        fixed = policy_bootstrap(score, fixed_choice, assignment, groups, draws, 12604)
        policy_reports[name] = dict(
            choice_counts=np.bincount(policy, minlength=8).tolist(),
            changed_from_fixed=int((policy != 7).sum()),
            changed_fraction=float((policy != 7).mean()),
            matched_rows=int((policy == assignment).sum()),
            value_mm=float(np.mean(value)),
            delta_vs_fixed=interval(value - fixed),
        )
    policy_metric_reports = {}
    for metric_name, values in metrics.items():
        policy_metric_reports[metric_name] = {}
        fixed = policy_bootstrap(values, fixed_choice, assignment, groups, draws, 12605)
        for name, policy in (("cm_selector", choice), ("cm_raw_top", raw_top),
                             ("cm_uncertainty_fallback_2std", uncertainty_fallback),
                             ("fixed_cup", fixed_choice)):
            policy_metric_reports[metric_name][name] = interval(
                policy_bootstrap(values, policy, assignment, groups, draws, 12605) - fixed
            )
    predicted_arm_means = cm_scores.mean(0)
    observed_arm_means = np.asarray(arm_stats["score_mm"])
    rank_pred = np.argsort(np.argsort(predicted_arm_means))
    rank_observed = np.argsort(np.argsort(observed_arm_means))
    rank_spearman = float(np.corrcoef(rank_pred, rank_observed)[0, 1])
    result = dict(
        schema="ref2dex.cm_candidate_advantage_audit.v2",
        run_status="COMPLETED",
        records=[str(p.resolve()) for p in records],
        rows=len(score), groups=int(len(np.unique(groups))), episodes=int(len(set(episodes))),
        checkpoint_sha256=checkpoint, candidate_names=NAMES, supports=supports,
        episode_support=episode_support,
        simulator_seeds=sorted({int(p["simulator_seed"]) for p in payloads}),
        panel_seeds=sorted({int(p["panel_seed"]) for p in payloads}),
        python_hash_seeds=sorted({p["python_hash_seed"] for p in payloads}),
        arm_score_mean_mm=arm_stats["score_mm"],
        arm_first_delta_mean_mm=arm_stats["first_delta_mm"],
        intervals_vs_fixed=intervals,
        policy_reports=policy_reports,
        policy_metric_reports=policy_metric_reports,
        predicted_arm_score_mean_mm=predicted_arm_means.tolist(),
        observed_arm_score_rank_spearman=rank_spearman,
        cm_context_ood=int(cm_ood.sum()), cm_candidate_ood=int(cm_candidate_ood.sum()),
        cluster_bootstrap_draws=draws,
        support_gate=bool(min(supports) >= 24 and min(episode_support) >= 12),
        interpretation="Known-propensity frozen-Cm candidate screen; not a stable grasp or trained-policy claim",
    )
    fallback = policy_reports["cm_uncertainty_fallback_2std"]
    fallback_passed = (
        result["support_gate"] and .05 <= fallback["changed_fraction"] <= .80
        and fallback["delta_vs_fixed"]["lower90"] > 0
        and policy_metric_reports["contact_last3"]["cm_uncertainty_fallback_2std"]["lower90"] >= -.05
        and policy_metric_reports["clearance_last3"]["cm_uncertainty_fallback_2std"]["lower90"] >= -.05
    )
    fallback_label = (
        "UNCLEAR" if not result["support_gate"]
        else "PROMISING_LOCAL_SIGNAL" if fallback_passed
        else "UNPROMISING"
    )
    result["predeclared_selector_label"] = "UNPROMISING" if result["support_gate"] else "UNCLEAR"
    result["exploratory_uncertainty_fallback"] = dict(
        label=fallback_label,
        rule="fixed top score only when top-minus-fixed > 2 ensemble std, no candidate OOD, retention >= fixed-.05, release <= fixed+.05",
        scope="offline known-propensity screen on frozen Cm predictions; requires fresh native validation before training",
    )
    result["probe_label"] = fallback_label
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.root.glob("run*/records.pt")) + sorted(args.root.glob("retry*/records.pt"))
    if not paths:
        raise ValueError("no scientific records")
    analyze(paths, args.output)


if __name__ == "__main__":
    main()
