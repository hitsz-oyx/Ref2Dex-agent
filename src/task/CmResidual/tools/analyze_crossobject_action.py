"""Audit randomized five-step action effects separately by object identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.task.CmResidual.tools.analyze_randomized_action import weighted_step_difference
from src.task.CmResidual.randomized_action import FINGER_SYNERGY_INDICES


def estimate(records: dict, *, horizon: int, num_envs: int) -> dict:
    assignment = records["assignment"].reshape(-1).numpy()
    steps = records["global_step"].reshape(-1).numpy()
    motion = records["motion_id"].reshape(-1).numpy()
    unique_steps = np.unique(steps)
    if (len(steps) != num_envs * len(unique_steps) or
            not np.array_equal(steps.reshape(-1, num_envs),
                               np.broadcast_to(unique_steps[:, None],
                                               (len(unique_steps), num_envs))) or
            not np.isin(assignment, [-1, 0, 1]).all()):
        raise ValueError("incomplete or invalid randomized step-major records")
    if not records["pre_contact"].reshape(-1).numpy()[assignment != 0].all():
        raise ValueError("treatment without pre-contact")
    action_delta = (records["executed_action"][:, 2] -
                    records["base_action"][:, 2]).numpy()
    if not np.isfinite(action_delta).all():
        raise FloatingPointError("non-finite action")
    # The caller checks the dose against the run's declared delta.
    survival = ((records["followup_progress"].reshape(-1) -
                 records["progress"].reshape(-1)).numpy() == horizon) & (
                     records["followup_reset"].reshape(-1).numpy() == 0)
    dz_mm = ((records["followup_object_state"][:, 2] -
              records["object_state"][:, 2]).numpy() * 1000)
    contact_fraction = records["followup_contact_count"].reshape(-1).numpy() / horizon
    outcome = np.where(survival, dz_mm * contact_fraction, 0.0)
    if not all(np.isfinite(x).all() for x in (outcome, dz_mm, contact_fraction)):
        raise FloatingPointError("non-finite physical outcome")

    def summarize(mask: np.ndarray) -> dict:
        treated = mask & (assignment != 0)
        plus = int((mask & (assignment == 1)).sum())
        minus = int((mask & (assignment == -1)).sum())
        if not plus or not minus:
            raise ValueError("missing randomized treatment arm")
        return {"treated": int(treated.sum()), "plus": plus, "minus": minus,
                "contact_supported_dz_plus_minus_mm": weighted_step_difference(
                    outcome[mask], assignment[mask], steps[mask])[0],
                "raw_dz_plus_minus_mm": weighted_step_difference(
                    dz_mm[mask], assignment[mask], steps[mask])[0],
                "contact_fraction_plus_minus": weighted_step_difference(
                    contact_fraction[mask], assignment[mask], steps[mask])[0],
                "reset_rate_plus": float((~survival[mask & (assignment == 1)]).mean()),
                "reset_rate_minus": float((~survival[mask & (assignment == -1)]).mean())}

    object_ids = sorted(np.unique(motion).tolist())
    by_object = {str(identifier): summarize(motion == identifier) for identifier in object_ids}
    pooled_strata = steps + motion * 1000
    pooled = weighted_step_difference(outcome, assignment, pooled_strata)[0]
    pooled_contact = weighted_step_difference(contact_fraction, assignment,
                                              pooled_strata)[0]
    return {"rows": len(steps), "steps": unique_steps.tolist(),
            "motion_ids": object_ids, "by_object": by_object,
            "pooled_contact_supported_dz_plus_minus_mm": pooled,
            "pooled_contact_fraction_plus_minus": pooled_contact,
            "outcome": outcome, "assignment": assignment, "strata": pooled_strata,
            "contact_fraction": contact_fraction,
            "num_envs": num_envs}


def bootstrap_ci(result: dict, *, draws: int, seed: int,
                 metric: str = "outcome") -> list[float]:
    rng = np.random.default_rng(seed)
    nenv = result["num_envs"]
    nsteps = len(result["steps"])
    samples = []
    for _ in range(draws):
        envs = rng.integers(0, nenv, nenv)
        indices = np.concatenate([step * nenv + envs for step in range(nsteps)])
        try:
            value = weighted_step_difference(
                result[metric][indices], result["assignment"][indices],
                result["strata"][indices])[0]
        except ValueError:
            continue
        samples.append(value)
    if len(samples) < .9 * draws:
        raise ValueError("too few valid environment bootstrap samples")
    return np.quantile(samples, [.025, .975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--bootstraps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=240924)
    args = parser.parse_args()
    if args.output.exists() or args.num_envs < 4 or args.bootstraps < 100:
        parser.error("new output, >=4 environments and >=100 bootstraps required")
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    finger = payload.get("schema") == "ref2dex.randomized_finger_followup.v1"
    wrist = payload.get("schema") == "ref2dex.randomized_action_followup.v1"
    if (not (finger or wrist) or payload.get("run_status") != "COMPLETED" or
            payload.get("followup_horizon") != (10 if finger else 5) or
            (finger and payload.get("finger_indices") != list(FINGER_SYNERGY_INDICES)) or
            (wrist and payload.get("intervention_axes") != [2])):
        raise ValueError("unexpected randomized followup source")
    records = payload["records"]
    assignment = records["assignment"].reshape(-1).numpy()
    dose = (records["executed_action"] - records["base_action"]).numpy()
    expected = np.zeros_like(dose)
    axes = list(FINGER_SYNERGY_INDICES) if finger else [2]
    expected[:, axes] = (payload["delta_z_action"] * assignment)[:, None]
    if not np.allclose(dose, expected, atol=1e-5):
        raise ValueError("executed treatment dose mismatch")
    result = estimate(records, horizon=10 if finger else 5,
                      num_envs=args.num_envs)
    report = {key: value for key, value in result.items()
              if key not in ("outcome", "contact_fraction", "assignment",
                             "strata", "num_envs")}
    report["environment_cluster_95ci_mm"] = bootstrap_ci(
        result, draws=args.bootstraps, seed=args.seed)
    report["contact_fraction_cluster_95ci"] = bootstrap_ci(
        result, draws=args.bootstraps, seed=args.seed + 1,
        metric="contact_fraction")
    report["source"] = str(args.input.resolve())
    report["action_kind"] = "finger_synergy" if finger else "wrist_z"
    report["schema"] = "ref2dex.crossobject_action_effect_report.v1"
    report["run_status"] = "COMPLETED"
    report["bootstrap_draws"] = args.bootstraps
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"pooled_mm": report["pooled_contact_supported_dz_plus_minus_mm"],
                      "ci_mm": report["environment_cluster_95ci_mm"],
                      "pooled_contact_fraction": report["pooled_contact_fraction_plus_minus"],
                      "contact_ci": report["contact_fraction_cluster_95ci"],
                      "by_object": report["by_object"]}, sort_keys=True))


if __name__ == "__main__":
    main()
