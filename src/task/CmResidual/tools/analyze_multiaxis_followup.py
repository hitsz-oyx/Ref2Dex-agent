"""Audit randomized wrist x/y/z 10-step effects without paired-state claims."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from src.task.CmResidual.tools.analyze_randomized_action import weighted_step_difference


AXES = (0, 1, 2)
DOSE = .1
HORIZON = 10


def load_run(path: Path, num_envs: int) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != "ref2dex.randomized_multiaxis_followup.v1" or
            payload.get("run_status") != "COMPLETED" or
            tuple(payload.get("intervention_axes", ())) != AXES or
            payload.get("followup_horizon") != HORIZON or
            abs(payload.get("delta_z_action", 0) - DOSE) > 1e-8):
        raise ValueError("multiaxis followup contract mismatch")
    rows = payload["records"]
    steps = rows["global_step"].numpy()
    unique_steps = np.unique(steps)
    if len(steps) != len(unique_steps) * num_envs or not np.all(
            steps.reshape(-1, num_envs) == unique_steps[:, None]):
        raise ValueError("incomplete step-major environment blocks")
    assignment = rows["assignment"].numpy()
    if not np.isin(assignment, [-3, -2, -1, 0, 1, 2, 3]).all() or not rows[
            "pre_contact"][assignment != 0].all():
        raise ValueError("invalid randomized assignment")
    difference = rows["executed_action"] - rows["base_action"]
    expected = torch.zeros_like(difference)
    for code, axis in enumerate(AXES, 1):
        chosen = np.abs(assignment) == code
        expected[chosen, axis] = torch.as_tensor(assignment[chosen].copy()).sign() * DOSE
    if not torch.allclose(difference, expected, atol=1e-5):
        raise ValueError("executed action differs from assigned dose")
    for step in unique_steps:
        labels = assignment[steps == step]
        counts = [int((labels == code).sum()) for code in (-3, -2, -1, 1, 2, 3)]
        if max(counts) - min(counts) > 1:
            raise ValueError("unbalanced axis allocation")
    current = rows["object_state"].numpy()
    next_state = rows["next_object_state"].numpy()
    future = rows["followup_object_state"].numpy()
    values = {
        "one_step_object_mm": (next_state[:, :3] - current[:, :3]) * 1000,
        "h10_object_mm": (future[:, :3] - current[:, :3]) * 1000,
        "h10_contact_pp": rows["followup_contact_count"].numpy() / HORIZON * 100,
        "h10_final_contact_pp": rows["followup_contact"].numpy() * 100,
        "h10_survival_pp": (((rows["followup_progress"] - rows["progress"] == HORIZON) &
                              (rows["followup_reset"].reshape(-1) == 0)).numpy() * 100),
        "pre_object_velocity": current[:, 7:10],
    }
    if not all(np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError("non-finite multiaxis outcome")
    return {"path": str(path), "steps": steps, "assignment": assignment,
            "values": values, "num_envs": num_envs}


def axis_effect(parts: list[dict], axis: int, outcome: str, *,
                bootstraps: int, seed: int) -> dict:
    code = AXES.index(axis) + 1
    rng = np.random.default_rng(seed)
    chosen = []
    for part_index, part in enumerate(parts):
        labels = np.where(np.abs(part["assignment"]) == code,
                          np.sign(part["assignment"]), 0)
        value = part["values"][outcome]
        if value.ndim == 2:
            value = value[:, axis]
        chosen.append((value, labels, part["steps"] + part_index * 1000,
                       len(np.unique(part["steps"])), part["num_envs"]))
    observed = weighted_step_difference(
        np.concatenate([row[0] for row in chosen]),
        np.concatenate([row[1] for row in chosen]),
        np.concatenate([row[2] for row in chosen]))[0]
    draws = []
    for _ in range(bootstraps):
        sampled = []
        for value, labels, steps, blocks, envs in chosen:
            env_ids = rng.integers(0, envs, size=envs)
            indices = np.concatenate([block * envs + env_ids for block in range(blocks)])
            sampled.append((value[indices], labels[indices], steps[indices]))
        try:
            draws.append(weighted_step_difference(
                np.concatenate([row[0] for row in sampled]),
                np.concatenate([row[1] for row in sampled]),
                np.concatenate([row[2] for row in sampled]))[0])
        except ValueError:
            continue
    if len(draws) < .9 * bootstraps:
        raise ValueError("insufficient valid environment-cluster resamples")
    return {"plus_minus": observed,
            "environment_cluster_95ci": np.quantile(draws, [.025, .975]).tolist(),
            "effective_bootstraps": len(draws)}


def analyze(args) -> None:
    parts = [load_run(path, args.num_envs) for path in args.input]
    report = {"schema": "ref2dex.multiaxis_followup_effects.v1",
              "classification": "Probe", "inputs": [str(path) for path in args.input],
              "axes": AXES, "delta_action": DOSE, "horizon": HORIZON,
              "runs": {}, "pooled": {}, "limits": [
                  "Sequential randomization estimates average effects on reached states.",
                  "Neither individual counterfactuals nor Cm policy utility are identified."]}
    outcomes = ("one_step_object_mm", "h10_object_mm", "h10_contact_pp",
                "h10_final_contact_pp", "h10_survival_pp", "pre_object_velocity")
    for axis in AXES:
        axis_name = ("x", "y", "z")[axis]
        report["pooled"][axis_name] = {
            outcome: axis_effect(parts, axis, outcome, bootstraps=args.bootstraps,
                                 seed=241000 + axis * 100 + index)
            for index, outcome in enumerate(outcomes)}
        for part_index, part in enumerate(parts):
            report["runs"].setdefault(Path(part["path"]).parent.name, {})[axis_name] = {
                outcome: axis_effect([part], axis, outcome,
                                     bootstraps=args.bootstraps,
                                     seed=241100 + part_index * 1000 + axis * 100 + index)
                for index, outcome in enumerate(outcomes)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"run_status": "COMPLETED", "output": str(args.output),
                      "pooled_h10_object_mm": {axis: report["pooled"][axis]["h10_object_mm"]
                                               for axis in ("x", "y", "z")}}, sort_keys=True))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()
    if (args.output.exists() or not 1 <= len(args.input) <= 2 or
            args.num_envs != 64 or not 100 <= args.bootstraps <= 3000):
        parser.error("new output, one/two 64-env inputs and bounded bootstrap required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.parent / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    root = Path(__file__).resolve().parents[4]
    manifest = {"run_status": "STARTED", "run_id": args.output.parent.name,
                "experiment_id": "P-20260924-multiaxis-h10-effects",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=root, text=True).strip(),
                "inputs": {str(path.resolve()): sha256(path) for path in args.input},
                "num_envs": args.num_envs, "bootstraps": args.bootstraps,
                "cpu_threads": 2, "gpu_count": 0, "wall_budget_minutes": 10,
                "output_budget_mb": 10}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        torch.set_num_threads(2)
        analyze(args)
        manifest.update(run_status="COMPLETED", report_sha256=sha256(args.output))
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
