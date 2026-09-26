"""Object-stratified signed-arm choice diagnostic for expert-state RCT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.task.CmResidual.tools.analyze_randomized_action import weighted_step_difference
from src.task.CmResidual.tools.probe_expert_crossobject_cm import OFFICIAL_SHA, SPLIT_SHA
from src.task.CmResidual.tools.probe_intervention_handflow import sha256


ARMS = (-3, -2, -1, 1, 2, 3)
NAMES = {"-3": "z-", "-2": "y-", "-1": "x-",
         "1": "x+", "2": "y+", "3": "z+"}


def load(path: Path) -> tuple[dict, dict]:
    manifest = json.loads((path / "run_manifest.json").read_text())
    transitions = path / "transitions.pt"
    if (manifest.get("run_status") != "COMPLETED" or
            manifest.get("checkpoint_sha256") != OFFICIAL_SHA or
            manifest.get("motion_manifest_sha256") != SPLIT_SHA or
            manifest.get("source_actor_role") != "official_data_collector" or
            manifest.get("source_partition") != "train" or
            manifest.get("source_objects") != ["airplane", "mug", "toothpaste"] or
            manifest.get("transitions_sha256") != sha256(transitions)):
        raise ValueError("expert multi-axis provenance drift")
    payload = torch.load(transitions, map_location="cpu", weights_only=False)
    if (payload.get("schema") != "ref2dex.randomized_multiaxis_followup.v1" or
            payload.get("run_status") != "COMPLETED" or
            payload.get("intervention_axes") != [0, 1, 2] or
            payload.get("followup_horizon") != 10 or
            payload.get("delta_z_action") != .1):
        raise ValueError("multi-axis run schema drift")
    records = payload["records"]
    assignment = records["assignment"].numpy()
    if not np.isin(assignment, [0, *ARMS]).all() or not records[
            "pre_contact"][assignment != 0].all():
        raise ValueError("invalid pre-contact randomization")
    actual = records["executed_action"] - records["base_action"]
    expected = torch.zeros_like(actual)
    for arm in ARMS:
        selected = assignment == arm
        expected[selected, abs(arm) - 1] = np.sign(arm) * .1
    if not torch.allclose(actual, expected, atol=1e-5):
        raise ValueError("executed multi-axis dose mismatch")
    steps = records["global_step"].numpy()
    envs = manifest["command"][manifest["command"].index("--num_envs") + 1]
    num_envs = int(envs)
    unique_steps = np.unique(steps)
    if (num_envs != 128 or len(steps) != len(unique_steps) * num_envs or
            not np.array_equal(steps.reshape(-1, num_envs),
                               np.broadcast_to(unique_steps[:, None],
                                               (len(unique_steps), num_envs)))):
        raise ValueError("incomplete 128-env step-major blocks")
    return records, manifest


def arm_cell_means(rows: dict, object_id: int, *, horizon: int) -> dict:
    step = rows["global_step"].numpy()
    motion = rows["motion_id"].numpy()
    assignment = rows["assignment"].numpy()
    if horizon == 10:
        survival = ((rows["followup_progress"] - rows["progress"] == 10) &
                    (rows["followup_reset"] == 0)).numpy()
        dz = (rows["followup_object_state"][:, 2] - rows["object_state"][:, 2]).numpy() * 1000
        contact = rows["followup_contact_count"].numpy() / 10
    elif horizon == 1:
        survival = np.ones(len(step), dtype=bool)
        dz = (rows["next_object_state"][:, 2] - rows["object_state"][:, 2]).numpy() * 1000
        contact = rows["next_contact"].numpy().astype(float)
    else:
        raise ValueError("only measured horizons 1 and 10 are available")
    outcome = np.where(survival, dz * contact, 0)
    if not np.isfinite(outcome).all():
        raise FloatingPointError("non-finite outcome")
    result = {}
    for arm in ARMS:
        cells = []
        counts = []
        contacts = []
        for value in np.unique(step):
            selected = (motion == object_id) & (step == value) & (assignment == arm)
            if selected.any():
                cells.append(float(outcome[selected].mean()))
                contacts.append(float(contact[selected].mean()))
                counts.append(int(selected.sum()))
        result[NAMES[str(arm)]] = {
            "adjusted_outcome_mm": float(np.mean(cells)) if cells else None,
            "mean_contact_fraction": float(np.mean(contacts)) if contacts else None,
            "treated": sum(counts), "steps_with_arm": len(cells),
            "min_cell_count": min(counts) if cells else 0}
    return result


def z_contrast(rows: dict, object_id: int, *, horizon: int,
               bootstraps: int = 500, seed: int = 240960) -> dict:
    step = rows["global_step"].numpy()
    motion = rows["motion_id"].numpy()
    assignment = rows["assignment"].numpy()
    labels = np.where(np.abs(assignment) == 3, np.sign(assignment), 0)
    if horizon == 1:
        dz = (rows["next_object_state"][:, 2] - rows["object_state"][:, 2]).numpy() * 1000
        contact = rows["next_contact"].numpy().astype(float)
        survival = np.ones(len(step), dtype=bool)
    else:
        dz = (rows["followup_object_state"][:, 2] - rows["object_state"][:, 2]).numpy() * 1000
        contact = rows["followup_contact_count"].numpy() / 10
        survival = ((rows["followup_progress"] - rows["progress"] == 10) &
                    (rows["followup_reset"] == 0)).numpy()
    y = np.where(survival, dz * contact, 0)
    mask = motion == object_id
    observed = weighted_step_difference(y[mask], labels[mask], step[mask])[0]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstraps):
        chosen_env = rng.integers(0, 128, 128)
        ids = np.concatenate([i * 128 + chosen_env for i in range(len(np.unique(step)))])
        selected = mask[ids]
        try:
            draws.append(weighted_step_difference(y[ids][selected], labels[ids][selected],
                                                  step[ids][selected])[0])
        except ValueError:
            continue
    if len(draws) < .9 * bootstraps:
        raise ValueError("too few valid environment cluster resamples")
    return {"z_plus_minus_mm": observed,
            "environment_cluster_95ci_mm": np.quantile(draws, [.025, .975]).tolist(),
            "effective_bootstraps": len(draws)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    rows, manifest = load(args.input)
    objects = {name: arm_cell_means(rows, i, horizon=10)
               for i, name in enumerate(manifest["source_objects"])}
    one_step_objects = {name: arm_cell_means(rows, i, horizon=1)
                        for i, name in enumerate(manifest["source_objects"])}
    temporal_z = {name: {"h1": z_contrast(rows, i, horizon=1, seed=240960+i),
                         "h10": z_contrast(rows, i, horizon=10, seed=240970+i)}
                  for i, name in enumerate(manifest["source_objects"])}
    for name, arms in objects.items():
        if any(value["treated"] < 50 or value["steps_with_arm"] < 12
               for value in arms.values()):
            raise ValueError(f"too few object/arm cells for {name}")
    global_mean = {arm: float(np.mean([objects[name][arm]["adjusted_outcome_mm"]
                                       for name in objects])) for arm in NAMES.values()}
    fixed = max(global_mean, key=global_mean.get)
    advantage = {}
    for name, arms in objects.items():
        best = max(arms, key=lambda arm: arms[arm]["adjusted_outcome_mm"])
        advantage[name] = {"best_arm": best, "global_fixed_arm": fixed,
                           "best_minus_fixed_mm": (arms[best]["adjusted_outcome_mm"] -
                                                   arms[fixed]["adjusted_outcome_mm"]),
                           "best_contact_fraction": arms[best]["mean_contact_fraction"]}
    gate = sum(item["best_arm"] != fixed and
               item["best_minus_fixed_mm"] >= 5 and
               item["best_contact_fraction"] >= .5
               for item in advantage.values()) >= 2
    report = {"schema": "ref2dex.expert_multiaxis_choice_probe.v1",
              "classification": "Probe", "input_sha256": sha256(args.input / "transitions.pt"),
              "objects": objects, "one_step_objects": one_step_objects,
              "temporal_z_contrast": temporal_z,
              "global_adjusted_outcome_mm": global_mean,
              "global_fixed_arm": fixed, "object_advantage": advantage,
              "continue_to_replication": gate,
              "limits": ["In-sample best-arm selection is optimistic.",
                         "No individual counterfactual or Cm policy utility is identified."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"global_fixed_arm": fixed, "object_advantage": advantage,
                      "continue_to_replication": gate}, sort_keys=True))


if __name__ == "__main__":
    main()
