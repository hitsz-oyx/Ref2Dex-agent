"""Pool two prespecified randomized sequence seeds by step and environment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from src.task.CmResidual.tools.analyze_two_step_sequence import contrasts, load_run, sha256


ROOT = Path(__file__).resolve().parents[4]


def pooled(parts: list[dict], outcome: str, count: int, seed: int) -> dict:
    if len(parts) != 2 or parts[0]["axis"] != parts[1]["axis"]:
        raise ValueError("two matching axis runs required")
    values = [part["values"][outcome] for part in parts]
    steps = [part["steps"] + index * 1000 for index, part in enumerate(parts)]
    observed = contrasts(np.concatenate(values),
                         np.concatenate([part["assignment"] for part in parts]),
                         np.concatenate(steps))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(count):
        sampled = []
        for part, value, stratum in zip(parts, values, steps):
            env_ids = rng.integers(0, part["num_envs"], size=part["num_envs"])
            ids = np.concatenate([block * part["num_envs"] + env_ids
                                  for block in range(part["num_blocks"])])
            sampled.append((value[ids], part["assignment"][ids], stratum[ids]))
        try:
            draws.append(contrasts(*(np.concatenate([row[index] for row in sampled])
                                     for index in range(3))))
        except ValueError:
            continue
    if len(draws) < .9 * count:
        raise ValueError("too few valid pooled cluster resamples")
    return {name: {"point": point,
                   "environment_cluster_95ci": np.quantile(
                       [draw[name] for draw in draws], [.025, .975]).tolist()}
            for name, point in observed.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=500)
    args = parser.parse_args()
    if (args.output.exists() or len(args.input) != 2 or
            args.input[0] == args.input[1] or not 100 <= args.bootstraps <= 3000):
        parser.error("two distinct inputs, new output and bounded resamples required")
    parts = [load_run(path, 64) for path in args.input]
    if parts[0]["axis"] != 2 or parts[1]["axis"] != 2:
        raise ValueError("this pooled decision is pinned to wrist z")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.parent / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest = {"run_status": "STARTED", "run_id": args.output.parent.name,
                "experiment_id": "P-20260924-cm-two-step-z-effect",
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "inputs": {str(path.resolve()): sha256(path) for path in args.input},
                "bootstraps": args.bootstraps, "gpu_count": 0, "cpu_threads": 2,
                "wall_budget_minutes": 10, "output_budget_mb": 10}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        result = {"schema": "ref2dex.two_step_sequence_pool.v1",
                  "classification": "Probe", "axis": 2,
                  "object_z_mm": pooled(parts, "object_axis_mm", args.bootstraps, 241171),
                  "contact_fraction_pp": pooled(parts, "contact_fraction_pp",
                                                args.bootstraps, 241172)}
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        manifest.update(run_status="COMPLETED", report_sha256=sha256(args.output))
        print(json.dumps({"run_status": "COMPLETED",
                          "interaction_z_mm": result["object_z_mm"]["interaction"],
                          "contact_two_minus_one_pp": result["contact_fraction_pp"][
                              "two_minus_one_contact"]}, sort_keys=True), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
