"""Audit x-primer then z+ randomized sequence against z+-only control."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from src.task.CmResidual.tools.analyze_two_step_sequence import sha256


ROOT = Path(__file__).resolve().parents[4]


def load_run(path: Path, num_envs: int) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != "ref2dex.crossaxis_primer_h10.v1" or
            payload.get("run_status") != "COMPLETED" or
            payload.get("intervention_axes") != [0, 2] or
            payload.get("crossaxis_primer") is not True or
            payload.get("followup_horizon") != 10 or
            abs(payload.get("delta_z_action", 0) - .1) > 1e-8):
        raise ValueError("cross-axis primer contract mismatch")
    rows = payload["records"]
    labels = rows["assignment"].numpy()
    steps = rows["global_step"].numpy()
    unique = np.unique(steps)
    if (len(steps) != len(unique) * num_envs or
            not np.all(steps.reshape(-1, num_envs) == unique[:, None]) or
            not np.isin(labels, [-1, 0, 1, 2]).all() or
            not rows["pre_contact"][labels != 0].all()):
        raise ValueError("invalid primer assignment")
    first = rows["executed_action"] - rows["base_action"]
    second = rows["second_executed_action"] - rows["second_base_action"]
    expected_first = torch.zeros_like(first)
    expected_second = torch.zeros_like(second)
    expected_first[:, 0] = (rows["assignment"].abs() == 1).float() * (
        rows["assignment"].sign().float() * .1)
    expected_second[:, 2] = (rows["assignment"] != 0).float() * .1
    if (not torch.allclose(first, expected_first, atol=1e-5) or
            not torch.allclose(second, expected_second, atol=1e-5)):
        raise ValueError("executed cross-axis doses differ from assignment")
    for step in unique:
        cell = labels[steps == step]
        counts = [int((cell == code).sum()) for code in (-1, 1, 2)]
        if max(counts) - min(counts) > 1:
            raise ValueError("unbalanced three-arm allocation")
    dz_mm = (rows["followup_object_state"][:, 2] - rows["object_state"][:, 2]).numpy() * 1000
    contact = rows["followup_contact_count"].numpy() / 10
    values = {"contact_supported_dz_mm": dz_mm * contact,
              "dz_mm": dz_mm, "contact_fraction_pp": contact * 100}
    if not all(np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError("nonfinite primer outcome")
    return {"steps": steps, "assignment": labels, "values": values,
            "num_envs": num_envs, "num_blocks": len(unique)}


def contrasts(value: np.ndarray, labels: np.ndarray, steps: np.ndarray) -> dict:
    totals = {name: 0.0 for name in ("x_minus_vs_control", "x_plus_vs_control")}
    weight = 0
    for step in np.unique(steps):
        mask = steps == step
        cells = {code: value[mask & (labels == code)] for code in (-1, 1, 2)}
        if any(not len(cell) for cell in cells.values()):
            continue
        n = sum(len(cell) for cell in cells.values())
        totals["x_minus_vs_control"] += n * (float(cells[-1].mean()) - float(cells[2].mean()))
        totals["x_plus_vs_control"] += n * (float(cells[1].mean()) - float(cells[2].mean()))
        weight += n
    if not weight:
        raise ValueError("no complete three-arm strata")
    return {key: total / weight for key, total in totals.items()}


def bootstrap(part: dict, outcome: str, count: int, seed: int) -> dict:
    value = part["values"][outcome]
    observed = contrasts(value, part["assignment"], part["steps"])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(count):
        env_ids = rng.integers(0, part["num_envs"], size=part["num_envs"])
        ids = np.concatenate([block * part["num_envs"] + env_ids
                              for block in range(part["num_blocks"])])
        try:
            draws.append(contrasts(value[ids], part["assignment"][ids], part["steps"][ids]))
        except ValueError:
            continue
    if len(draws) < .9 * count:
        raise ValueError("too few valid primer bootstrap draws")
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
                "experiment_id": "P-20260924-cm-crossaxis-primer",
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "input": str(args.input.resolve()), "input_sha256": sha256(args.input),
                "bootstraps": args.bootstraps, "num_envs": args.num_envs,
                "gpu_count": 0, "cpu_threads": 2, "wall_budget_minutes": 10,
                "output_budget_mb": 10}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        report = {"schema": "ref2dex.crossaxis_primer_effect.v1",
                  "classification": "Probe",
                  "contact_supported_dz_mm": bootstrap(part, "contact_supported_dz_mm",
                                                      args.bootstraps, 241181),
                  "dz_mm": bootstrap(part, "dz_mm", args.bootstraps, 241182),
                  "contact_fraction_pp": bootstrap(part, "contact_fraction_pp",
                                                   args.bootstraps, 241183),
                  "limits": "randomized population effect, not per-state counterfactual or policy utility"}
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest.update(run_status="COMPLETED", report_sha256=sha256(args.output))
        print(json.dumps({"run_status": "COMPLETED",
                          "primary": report["contact_supported_dz_mm"]["x_minus_vs_control"],
                          "contact_cost": report["contact_fraction_pp"][
                              "x_minus_vs_control"]}, sort_keys=True), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
