"""Final data-coverage LOO check for the unchanged six-region signed Cm."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from src.task.CmResidual.tools.probe_crossobject_causal_head import (
    effect_estimate, fit, inputs, evaluate,
)
from src.task.CmResidual.tools.probe_expert_crossobject_cm import ROOT, calibrate
from src.task.CmResidual.tools.probe_intervention_handflow import sha256
from src.task.CmResidual.tools.probe_train5_causal_head import load_rows, select


SOURCE = ROOT / "outputs/CmResidual/agent_expert_train8_randomized_s186_h5"
SOURCE_SHA = "77e8f90d6ef8ef016d8cd74b46e97ae6fc4f6c9be63e290f4fe1f383cff7b6e1"
SPLIT_SHA = "7aae15dfc75ed88ef5481b8e26ae83d30214153cc124f4644fd63f6c71a3812b"
OBJECT_NAMES = ["airplane", "cubesmall", "cup", "duck", "mug", "phone", "toothpaste", "waterbottle"]
MOTION_NAMES = ["cubesmall", "cup", "duck", "mug", "phone", "toothpaste",
                "waterbottle", "airplane", "cubesmall"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    if args.output.exists() or args.steps != 300:
        parser.error("new output and exactly 300 matched steps required")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    manifest_path = args.output / "run_manifest.json"
    manifest = {"run_status": "STARTED", "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "source_sha256": SOURCE_SHA, "split_sha256": SPLIT_SHA,
                "folds": OBJECT_NAMES, "motion_id_to_object": MOTION_NAMES,
                "steps": 300, "gpu_count": 0, "cpu_threads": 2,
                "wall_budget_minutes": 30, "output_budget_mb": 10,
                "stop_rule": "source drift, object leakage, non-finite or wall budget"}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        torch.set_num_threads(2)
        rows = load_rows(source=SOURCE, source_sha=SOURCE_SHA, split_sha=SPLIT_SHA,
                         source_objects=OBJECT_NAMES, motion_names=MOTION_NAMES)
        results = {}
        for fold, object_name in enumerate(OBJECT_NAMES):
            held_ids = [i for i, name in enumerate(MOTION_NAMES) if name == object_name]
            train_ids = [i for i, name in enumerate(MOTION_NAMES) if name != object_name]
            train, train_names = select(rows, train_ids, MOTION_NAMES)
            val, val_names = select(rows, held_ids, MOTION_NAMES)
            if set(train_names) & set(val_names):
                raise ValueError("object identity leakage")
            setup = calibrate(train)
            models = {}
            for kind in ("geometric", "raw"):
                train_state, train_difference = inputs(train, train_names, setup, kind)
                val_state, val_difference = inputs(val, val_names, setup, kind)
                model = fit(train, train_state, train_difference,
                            seed=240970 + fold, steps=args.steps)
                models[kind] = evaluate(model, val, val_state, val_difference,
                                        seed=240980 + fold)
            models["constant"] = {"predicted_ate_mm": effect_estimate(train),
                                  "actual_ate_mm": effect_estimate(val)}
            models["fit_rows"] = len(train["q"])
            models["heldout_rows"] = len(val["q"])
            results[object_name] = models
            print(json.dumps({"completed_fold": object_name,
                              "actual_ate_mm": models["constant"]["actual_ate_mm"]},
                             sort_keys=True), flush=True)
        mae = {kind: float(np.mean([abs(results[name][kind]["predicted_ate_mm"] -
                                        results[name][kind]["actual_ate_mm"])
                                    for name in OBJECT_NAMES]))
               for kind in ("geometric", "raw", "constant")}
        gate = mae["geometric"] <= .8 * min(mae["raw"], mae["constant"])
        report = {"schema": "ref2dex.train8_causal_head_loo.v1",
                  "classification": "Probe", "source_actor_role": "official_data_collector",
                  "folds": results, "object_ate_mae_mm": mae,
                  "continue_six_region_route": gate,
                  "elapsed_seconds": time.monotonic() - started}
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest.update(run_status="COMPLETED", report_sha256=sha256(report_path))
        print(json.dumps({"object_ate_mae_mm": mae,
                          "continue_six_region_route": gate}, sort_keys=True), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
