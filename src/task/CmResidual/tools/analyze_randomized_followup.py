"""Estimate short-horizon randomized action effects, not per-state counterfactuals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.task.CmResidual.tools.analyze_randomized_action import weighted_step_difference


def effects(values, assignment, steps, num_envs, *, bootstraps, seed):
    rng = np.random.default_rng(seed)
    observed = {key: weighted_step_difference(value, assignment, steps)[0]
                for key, value in values.items()}
    unique_steps = np.unique(steps)
    draws = {key: np.empty(bootstraps) for key in values}
    for sample in range(bootstraps):
        for _ in range(100):
            envs = rng.integers(0, num_envs, size=num_envs)
            indices = np.concatenate([j * num_envs + envs for j in range(len(unique_steps))])
            if all(np.any(assignment[indices][steps[indices] == step] == arm)
                   for step in unique_steps for arm in (-1, 1)):
                break
        else:
            raise ValueError("bootstrap cannot retain both treatments in every stratum")
        for key, value in values.items():
            draws[key][sample] = weighted_step_difference(
                value[indices], assignment[indices], steps[indices])[0]
    return {key: {"plus_minus": estimate,
                  "environment_cluster_95ci": np.quantile(draws[key], [.025, .975]).tolist()}
            for key, estimate in observed.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=240924)
    args = parser.parse_args()
    if args.output.exists() or args.num_envs < 4 or args.bootstraps < 100:
        parser.error("new output, >=4 environments and >=100 bootstraps required")
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.randomized_action_followup.v1" or (
            payload.get("run_status") != "COMPLETED"):
        raise ValueError("incomplete followup randomized data")
    data = payload["records"]
    horizon = payload["followup_horizon"]
    steps = data["global_step"].numpy()
    assignment = data["assignment"].numpy()
    unique_steps = np.unique(steps)
    if len(steps) != len(unique_steps) * args.num_envs or not np.all(
            steps.reshape(-1, args.num_envs) == unique_steps[:, None]):
        raise ValueError("unexpected environment block layout")
    if not np.all(np.isin(assignment, [-1, 0, 1])) or not np.all(
            data["pre_contact"].numpy()[assignment != 0]):
        raise ValueError("invalid randomized contact assignment")
    action_delta = (data["executed_action"][:, 2] - data["base_action"][:, 2]).numpy()
    if not np.allclose(action_delta[assignment != 0],
                       payload["delta_z_action"] * assignment[assignment != 0], atol=1e-5):
        raise ValueError("action clipped or differs from assigned dose")
    values = {
        "one_step_object_dz_mm": ((data["next_object_state"][:, 2] -
                                   data["object_state"][:, 2]).numpy() * 1000),
        "followup_object_dz_mm": ((data["followup_object_state"][:, 2] -
                                   data["object_state"][:, 2]).numpy() * 1000),
        "followup_contact_fraction": (data["followup_contact_count"].numpy() / horizon),
        "followup_final_contact": data["followup_contact"].numpy().astype(float),
        "followup_survival": ((data["followup_progress"] - data["progress"] == horizon) &
                              (data["followup_reset"].reshape(-1) == 0)).numpy().astype(float),
    }
    if not all(np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError("non-finite followup outcome")
    result = effects(values, assignment, steps, args.num_envs,
                     bootstraps=args.bootstraps, seed=args.seed)
    report = {"schema": "ref2dex.randomized_followup_report.v1",
              "source": str(args.input), "run_status": "COMPLETED",
              "num_envs": args.num_envs, "horizon": horizon,
              "contact_interventions": int((assignment != 0).sum()),
              "steps": unique_steps.tolist(), "effects": result,
              "resamples": {"environment_cluster_bootstraps": args.bootstraps,
                            "seed": args.seed},
              "limits": ["Sequential randomization estimates average treatment effects on reached states.",
                         "Repeated environments and previous-treatment carryover are clustered, not independent rows.",
                         "A short-horizon effect is not long-horizon policy utility or a per-state counterfactual."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"contact_interventions": report["contact_interventions"],
                      "effects": result}, sort_keys=True))


if __name__ == "__main__":
    main()
