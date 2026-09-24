"""Check whether global mesh scale repairs five-object geometric Cm LOO transfer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import trimesh

from src.task.CmResidual.tools.audit_original_cmv2_on_policy import ASSETS
from src.task.CmResidual.tools.probe_crossobject_causal_head import (
    effect_estimate, fit, inputs, evaluate,
)
from src.task.CmResidual.tools.probe_expert_crossobject_cm import ROOT, calibrate
from src.task.CmResidual.tools.probe_intervention_handflow import sha256
from src.task.CmResidual.tools.probe_train5_causal_head import (
    MOTION_NAMES, OBJECT_NAMES, SOURCE_SHA, load_rows, select,
)


PRIOR = ROOT / "outputs/CmResidual/agent_train5_causal_head_loo_s186/report.json"


@lru_cache(maxsize=10)
def descriptor(name: str) -> torch.Tensor:
    path = ASSETS / "mjcf/objects" / name / f"{name}.obj"
    mesh = trimesh.load(path, force="mesh", process=False)
    extent = np.asarray(mesh.extents, dtype=np.float64)
    volume, area = float(mesh.volume), float(mesh.area)
    if (extent.shape != (3,) or not np.isfinite(extent).all() or
            (extent <= 0).any() or not math.isfinite(volume) or volume <= 0 or
            not math.isfinite(area) or area <= 0):
        raise ValueError(f"invalid mesh geometry {name}")
    return torch.tensor([*(extent / .1), math.log(volume / 1e-4),
                         math.log(area / .03)], dtype=torch.float32)


def shape_inputs(rows: dict, names: list[str], setup: dict) -> tuple[torch.Tensor, torch.Tensor]:
    state, difference = inputs(rows, names, setup, "geometric")
    shape = torch.stack([descriptor(names[int(i)]) for i in rows["motion_id"]])
    return torch.cat((state, shape), dim=-1), difference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    if args.output.exists() or args.steps != 300:
        parser.error("new output and exactly 300 matched steps required")
    if not PRIOR.is_file():
        raise FileNotFoundError(PRIOR)
    old = json.loads(PRIOR.read_text())
    if (old.get("schema") != "ref2dex.train5_causal_head_loo.v1" or
            old.get("continue_to_alarmclock") is not False):
        raise ValueError("prior six-region comparison drift")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    manifest_path = args.output / "run_manifest.json"
    manifest = {"run_status": "STARTED", "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "source_sha256": SOURCE_SHA, "prior_report_sha256": sha256(PRIOR),
                "folds": OBJECT_NAMES, "descriptor": "extent_xyz_volume_area",
                "steps": 300, "gpu_count": 0, "cpu_threads": 2,
                "wall_budget_minutes": 30, "output_budget_mb": 10,
                "stop_rule": "source drift, invalid mesh, non-finite or wall budget"}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        torch.set_num_threads(2)
        rows = load_rows()
        results = {}
        for fold, object_name in enumerate(OBJECT_NAMES):
            held_ids = [i for i, name in enumerate(MOTION_NAMES) if name == object_name]
            train_ids = [i for i, name in enumerate(MOTION_NAMES) if name != object_name]
            train, train_names = select(rows, train_ids)
            val, val_names = select(rows, held_ids)
            if set(train_names) & set(val_names):
                raise ValueError("object identity leakage")
            setup = calibrate(train)
            train_state, train_diff = shape_inputs(train, train_names, setup)
            val_state, val_diff = shape_inputs(val, val_names, setup)
            model = fit(train, train_state, train_diff,
                        seed=240970 + fold, steps=args.steps)
            results[object_name] = evaluate(model, val, val_state, val_diff,
                                            seed=240980 + fold)
            print(json.dumps({"completed_fold": object_name,
                              "predicted_ate_mm": results[object_name]["predicted_ate_mm"]},
                             sort_keys=True), flush=True)
        mae = float(np.mean([abs(results[name]["predicted_ate_mm"] -
                                 results[name]["actual_ate_mm"])
                             for name in OBJECT_NAMES]))
        prior = old["object_ate_mae_mm"]
        gate = mae <= .8 * prior["geometric"] and mae < min(prior["raw"], prior["constant"])
        report = {"schema": "ref2dex.train5_shape_causal_head_loo.v1",
                  "classification": "Probe", "source_actor_role": "official_data_collector",
                  "folds": results, "shape_geometric_object_ate_mae_mm": mae,
                  "prior_object_ate_mae_mm": prior,
                  "continue_to_new_heldout": gate,
                  "elapsed_seconds": time.monotonic() - started}
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest.update(run_status="COMPLETED", report_sha256=sha256(report_path))
        print(json.dumps({"shape_geometric_object_ate_mae_mm": mae,
                          "prior_object_ate_mae_mm": prior,
                          "continue_to_new_heldout": gate}, sort_keys=True), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
