#!/usr/bin/env python3
"""Analyze the native A/B validation of the frozen Cm uncertainty fallback."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


NAMES = ["expert0", "expert1", "expert2", "expert3", "expert4", "expert5", "base_hold", "fixed_cup"]


def bootstrap_delta(values, policy, groups, draws, seed):
    unique, code = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    result = np.empty(draws, dtype=float)
    for i in range(draws):
        multiplicity = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        denominator = float(multiplicity[code].sum())
        a = np.where(policy == 0, values * 2.0, 0.0)
        b = np.where(policy == 1, values * 2.0, 0.0)
        result[i] = ((a * multiplicity[code]).sum() - (b * multiplicity[code]).sum()) / max(denominator, 1.0)
    return result


def interval(values):
    return dict(mean=float(np.mean(values)), lower90=float(np.percentile(values, 5)),
                upper90=float(np.percentile(values, 95)))


def policy_value(values, policy, choice):
    selected = policy == choice
    return float(np.where(selected, values * 2.0, 0.0).sum() / max(len(values), 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    records = sorted(args.root.glob("seed*/records.pt"))
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in records]
    if not payloads or any(p.get("schema") != "ref2dex.cm_uncertainty_native.v1" for p in payloads):
        raise ValueError("uncertainty native schema mismatch")
    checkpoint = payloads[0]["checkpoint_sha256"]
    if any(p["checkpoint_sha256"] != checkpoint for p in payloads):
        raise ValueError("frozen Cm checkpoint drift")
    if any(p.get("fixed_cup_index") != 7 or not p.get("frozen_cm") or p.get("optimizer_used")
           or not p.get("assignment_after_observation") for p in payloads):
        raise ValueError("native A/B contract failed")
    policy = torch.cat([p["policy_assignment"] for p in payloads]).numpy().astype(int)
    chosen = torch.cat([p["chosen_candidate"] for p in payloads]).numpy().astype(int)
    cm_choice = torch.cat([p["cm_choice"] for p in payloads]).numpy().astype(int)
    propensity = torch.cat([p["propensity"] for p in payloads]).numpy()
    motion = torch.cat([p["motion_id"] for p in payloads]).numpy().astype(int)
    start = torch.cat([p["start_frame"] for p in payloads]).numpy().astype(int)
    episode = sum((p["episode_id"] for p in payloads), [])
    score = torch.cat([p["outcome"]["score_mm"] for p in payloads]).numpy().astype(float)
    first = torch.cat([p["outcome"]["first_delta_mm"] for p in payloads]).numpy().astype(float)
    retained = torch.cat([p["outcome"]["retained"] for p in payloads]).numpy().astype(float)
    contact = torch.cat([p["outcome"]["contact_last3"] for p in payloads]).numpy().astype(float)
    clearance = torch.cat([p["outcome"]["clearance_last3"] for p in payloads]).numpy().astype(float)
    if len(set(propensity.tolist())) != 1 or not np.isclose(propensity[0], .5):
        raise ValueError("propensity is not fixed p=.5")
    if not np.all(np.isfinite(score)) or not np.all(np.isfinite(first)):
        raise ValueError("nonfinite native outcome")
    if not np.all((policy == 0) | (policy == 1)) or not np.all((chosen >= 0) & (chosen < 8)):
        raise ValueError("invalid policy labels")
    groups = np.asarray([f"{m}:{s}" for m, s in zip(motion, start)])
    metrics = {"score_mm": score, "first_delta_mm": first, "retained": retained,
               "contact_last3": contact, "clearance_last3": clearance}
    reports = {
        key: interval(bootstrap_delta(values, policy, groups, args.draws, 61003 + index))
        for index, (key, values) in enumerate(metrics.items())
    }
    counts = np.bincount(policy, minlength=2)
    episodes = np.asarray(episode)
    episode_counts = [int(np.unique(episodes[policy == arm]).size) for arm in (0, 1)]
    changed = (chosen[policy == 0] != 7)
    candidate_counts = np.bincount(chosen[policy == 0], minlength=8)
    support_gate = bool(counts.min() >= 48 and min(episode_counts) >= 20)
    risk_gate = bool(all(reports[key]["lower90"] >= -0.05 for key in ("retained", "contact_last3", "clearance_last3")))
    score_gate = reports["score_mm"]["lower90"] > 0.0
    label = "PROMISING" if support_gate and score_gate and risk_gate else "UNPROMISING"
    result = dict(
        schema="ref2dex.cm_uncertainty_native_audit.v1",
        run_status="COMPLETED", records=[str(path.resolve()) for path in records], seeds=[p["seed"] for p in payloads],
        rows=len(score), groups=int(np.unique(groups).size), episodes=len(set(episode)),
        checkpoint_sha256=checkpoint, candidate_names=NAMES,
        policy_definition="0=cm_uncertainty_fallback_2std, 1=fixed_cup",
        policy_counts=counts.tolist(), policy_episode_counts=episode_counts,
        policy_value_reports={
            "cm_uncertainty_fallback_2std": {key: policy_value(values, policy, 0) for key, values in metrics.items()},
            "fixed_cup": {key: policy_value(values, policy, 1) for key, values in metrics.items()},
        },
        delta_reports=reports, changed_from_fixed=int(changed.sum()),
        changed_fraction=float(changed.mean()) if len(changed) else 0.0,
        cm_choice_counts=np.bincount(cm_choice[policy == 0], minlength=8).tolist(),
        chosen_candidate_counts= candidate_counts.tolist(),
        support_gate=support_gate, score_gate=score_gate, risk_gate=risk_gate,
        support_requirements={"each_policy_rows": 48, "each_policy_episodes": 20},
        cluster_bootstrap_draws=args.draws, label=label,
        interpretation="Fresh native randomized A/B local utility validation; still not a trained-policy or task-success claim",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
