"""Audit randomized physical action effects without individual-pair claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def weighted_step_difference(outcome, assignment, steps):
    numerator = 0.0
    denominator = 0
    per_step = []
    for step in np.unique(steps):
        selected = steps == step
        plus = selected & (assignment == 1)
        minus = selected & (assignment == -1)
        if not plus.any() or not minus.any():
            continue
        count = int(plus.sum() + minus.sum())
        difference = float(outcome[plus].mean() - outcome[minus].mean())
        numerator += count * difference
        denominator += count
        per_step.append({"step": int(step), "selected": count,
                         "plus": int(plus.sum()), "minus": int(minus.sum()),
                         "difference": difference})
    if not denominator:
        raise ValueError("no randomized treatment contrast")
    return numerator / denominator, per_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=240924)
    args = parser.parse_args()
    if args.output.exists() or args.permutations < 100 or args.bootstraps < 100:
        parser.error("new output and at least 100 resamples required")
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    if payload["run_status"] != "COMPLETED":
        raise ValueError("transition run was not completed")
    data = payload["records"]
    steps = data["global_step"].numpy()
    assignment = data["assignment"].numpy()
    count = len(steps)
    unique_steps = np.unique(steps)
    if count != len(unique_steps) * args.num_envs:
        raise ValueError("record layout is not complete per-step environment blocks")
    for index, step in enumerate(unique_steps):
        if not np.all(steps[index * args.num_envs:(index + 1) * args.num_envs] == step):
            raise ValueError("record ordering differs from declared environment blocks")
    if not np.all(np.isin(assignment, [-1, 0, 1])):
        raise ValueError("invalid assignment label")
    contact = data["pre_contact"].numpy().astype(bool)
    if not np.all(contact[assignment != 0]):
        raise ValueError("randomized action was applied without pre-contact")
    obj_dz = (data["next_object_state"][:, 2] - data["object_state"][:, 2]).numpy() * 1000
    hand_dz = (data["next_q"][:, 2] - data["q"][:, 2]).numpy() * 1000
    before_obj_vz = data["object_state"][:, 9].numpy() * 1000
    before_hand_z = data["q"][:, 2].numpy() * 1000
    action_z = data["executed_action"][:, 2].numpy()
    base_action_z = data["base_action"][:, 2].numpy()
    if not np.all(np.isfinite(obj_dz)) or not np.all(np.isfinite(hand_dz)):
        raise FloatingPointError("non-finite physical effect")
    if not np.allclose(action_z[assignment != 0] - base_action_z[assignment != 0],
                       payload["delta_z_action"] * assignment[assignment != 0], atol=1e-5):
        raise ValueError("actual action perturbation differs from randomized assignment")
    observed_obj, per_step_obj = weighted_step_difference(obj_dz, assignment, steps)
    observed_hand, per_step_hand = weighted_step_difference(hand_dz, assignment, steps)
    pre_vz_diff, _ = weighted_step_difference(before_obj_vz, assignment, steps)
    pre_hand_diff, _ = weighted_step_difference(before_hand_z, assignment, steps)
    groups = [np.flatnonzero((steps == step) & (assignment != 0))
              for step in unique_steps]
    rng = np.random.default_rng(args.seed)
    permuted_effects = np.empty(args.permutations)
    for sample in range(args.permutations):
        permuted = assignment.copy()
        for indices in groups:
            permuted[indices] = rng.permutation(permuted[indices])
        permuted_effects[sample] = weighted_step_difference(obj_dz, permuted, steps)[0]
    p_two_sided = float((1 + np.count_nonzero(
        np.abs(permuted_effects) >= abs(observed_obj))) / (args.permutations + 1))
    bootstrap_effects = np.empty(args.bootstraps)
    for sample in range(args.bootstraps):
        sampled_envs = rng.integers(0, args.num_envs, size=args.num_envs)
        sampled_indices = np.concatenate([
            step_index * args.num_envs + sampled_envs
            for step_index in range(len(unique_steps))])
        bootstrap_effects[sample] = weighted_step_difference(
            obj_dz[sampled_indices], assignment[sampled_indices],
            steps[sampled_indices])[0]
    report = {
        "schema": "ref2dex.randomized_action_effect_report.v1",
        "source": str(args.input), "run_status": payload["run_status"],
        "rows": count, "contact_interventions": int((assignment != 0).sum()),
        "plus": int((assignment == 1).sum()), "minus": int((assignment == -1).sum()),
        "steps": unique_steps.tolist(), "delta_z_action": payload["delta_z_action"],
        "object_dz_plus_minus_mm": observed_obj,
        "object_dz_cluster_bootstrap_95ci_mm": np.quantile(
            bootstrap_effects, [.025, .975]).tolist(),
        "object_dz_stratified_permutation_p_two_sided": p_two_sided,
        "hand_dz_plus_minus_mm": observed_hand,
        "pre_object_vz_plus_minus_mm_per_s": pre_vz_diff,
        "pre_hand_z_plus_minus_mm": pre_hand_diff,
        "per_step_object_dz": per_step_obj,
        "per_step_hand_dz": per_step_hand,
        "resamples": {"permutations": args.permutations,
                      "environment_cluster_bootstraps": args.bootstraps,
                      "seed": args.seed},
        "limits": [
            "sequential randomization identifies average immediate effects on reached states",
            "repeated environments and treatment carryover limit naive independent-row inference",
            "no individual same-state counterfactuals and no Cm policy-utility evidence",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "contact_interventions", "object_dz_plus_minus_mm",
        "object_dz_cluster_bootstrap_95ci_mm",
        "object_dz_stratified_permutation_p_two_sided",
        "hand_dz_plus_minus_mm")}, sort_keys=True))


if __name__ == "__main__":
    main()
