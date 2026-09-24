"""Audit randomized one-vs-two-step wrist-x physical effects."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(path: Path, num_envs: int) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != "ref2dex.randomized_sequence_h10.v1" or
            payload.get("run_status") != "COMPLETED" or
            payload.get("intervention_axes") not in ([0], [2]) or
            payload.get("sequence_lengths") != [1, 2] or
            payload.get("followup_horizon") != 10 or
            abs(payload.get("delta_z_action", 0) - .1) > 1e-8):
        raise ValueError("sequence transition contract mismatch")
    rows = payload["records"]
    axis = payload["intervention_axes"][0]
    assignment = rows["assignment"].numpy()
    steps = rows["global_step"].numpy()
    unique_steps = np.unique(steps)
    if (len(steps) != len(unique_steps) * num_envs or
            not np.all(steps.reshape(-1, num_envs) == unique_steps[:, None]) or
            not np.isin(assignment, [-2, -1, 0, 1, 2]).all() or
            not rows["pre_contact"][assignment != 0].all()):
        raise ValueError("incomplete or invalid sequence assignment")
    first = rows["executed_action"] - rows["base_action"]
    second = rows["second_executed_action"] - rows["second_base_action"]
    expected_first, expected_second = torch.zeros_like(first), torch.zeros_like(second)
    expected_first[:, axis] = rows["assignment"].sign().float() * .1
    expected_second[:, axis] = (rows["assignment"].abs() == 2).float() * (
        rows["assignment"].sign().float() * .1)
    if (not torch.allclose(first, expected_first, atol=1e-5) or
            not torch.allclose(second, expected_second, atol=1e-5)):
        raise ValueError("executed sequence differs from randomized dose")
    for step in unique_steps:
        labels = assignment[steps == step]
        counts = [int((labels == code).sum()) for code in (-2, -1, 1, 2)]
        if max(counts) - min(counts) > 1:
            raise ValueError("unbalanced sequence allocation")
    current = rows["object_state"].numpy()
    followup = rows["followup_object_state"].numpy()
    values = {"object_axis_mm": (followup[:, axis] - current[:, axis]) * 1000,
              "contact_fraction_pp": rows["followup_contact_count"].numpy() * 10,
              "final_contact_pp": rows["followup_contact"].numpy() * 100}
    if axis == 0:
        values["object_x_mm"] = values["object_axis_mm"]
    if not all(np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError("non-finite sequence outcome")
    return {"assignment": assignment, "steps": steps, "values": values,
            "num_envs": num_envs, "num_blocks": len(unique_steps),
            "axis": axis, "path": str(path)}


def contrasts(value: np.ndarray, assignment: np.ndarray, steps: np.ndarray) -> dict:
    totals = {name: 0.0 for name in ("one_effect", "two_effect", "interaction",
                                      "two_minus_one_contact")}
    weight = 0
    for step in np.unique(steps):
        mask = steps == step
        cells = {code: value[mask & (assignment == code)] for code in (-2, -1, 1, 2)}
        if any(not len(cell) for cell in cells.values()):
            continue
        means = {code: float(cell.mean()) for code, cell in cells.items()}
        one = means[1] - means[-1]
        two = means[2] - means[-2]
        n = sum(len(cell) for cell in cells.values())
        totals["one_effect"] += n * one
        totals["two_effect"] += n * two
        totals["interaction"] += n * (two - one)
        totals["two_minus_one_contact"] += n * (
            (means[2] + means[-2] - means[1] - means[-1]) / 2)
        weight += n
    if not weight:
        raise ValueError("no complete randomized sequence strata")
    return {name: result / weight for name, result in totals.items()}


def bootstrap(part: dict, outcome: str, count: int, seed: int) -> dict:
    value = part["values"][outcome]
    assignment, steps = part["assignment"], part["steps"]
    observed = contrasts(value, assignment, steps)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(count):
        env_ids = rng.integers(0, part["num_envs"], size=part["num_envs"])
        indices = np.concatenate([block * part["num_envs"] + env_ids
                                  for block in range(part["num_blocks"])])
        try:
            draws.append(contrasts(value[indices], assignment[indices], steps[indices]))
        except ValueError:
            continue
    if len(draws) < .9 * count:
        raise ValueError("too few valid environment-cluster draws")
    return {name: {"point": point,
                   "environment_cluster_95ci": np.quantile(
                       [draw[name] for draw in draws], [.025, .975]).tolist()}
            for name, point in observed.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--bootstraps", type=int, default=500)
    args = parser.parse_args()
    if (args.output.exists() or args.num_envs != 64 or
            not 100 <= args.bootstraps <= 3000):
        parser.error("new output, 64 environments and bounded resamples required")
    part = load_run(args.input, args.num_envs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.parent / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest = {"run_status": "STARTED", "run_id": args.output.parent.name,
                "experiment_id": "P-20260924-cm-two-step-physical-effect",
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                      cwd=ROOT, text=True).strip(),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "input": str(args.input.resolve()), "input_sha256": sha256(args.input),
                "num_envs": args.num_envs, "bootstraps": args.bootstraps,
                "gpu_count": 0, "cpu_threads": 2, "wall_budget_minutes": 10,
                "output_budget_mb": 10}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        report = {"schema": "ref2dex.two_step_sequence_effect.v1",
                  "classification": "Probe", "input": str(args.input),
                  "axis": part["axis"],
                  "object_axis_mm": bootstrap(part, "object_axis_mm", args.bootstraps, 241131),
                  "contact_fraction_pp": bootstrap(part, "contact_fraction_pp",
                                                   args.bootstraps, 241132),
                  "final_contact_pp": bootstrap(part, "final_contact_pp",
                                                args.bootstraps, 241133),
                  "limits": "Population randomized sequence effect; not individual counterfactual or policy utility"}
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest.update(run_status="COMPLETED", report_sha256=sha256(args.output))
        print(json.dumps({"run_status": "COMPLETED",
                          "interaction_axis_mm": report["object_axis_mm"]["interaction"],
                          "contact_two_minus_one_pp": report["contact_fraction_pp"][
                              "two_minus_one_contact"]}, sort_keys=True), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
