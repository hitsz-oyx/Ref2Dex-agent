"""Train raw action-effect Cm on eight objects, then open alarmclock RCT labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import torch

from src.task.CmResidual.tools.probe_crossobject_causal_head import (
    effect_estimate, fit, inputs, evaluate,
)
from src.task.CmResidual.tools.probe_expert_crossobject_cm import ROOT
from src.task.CmResidual.tools.probe_intervention_handflow import sha256
from src.task.CmResidual.tools.probe_train5_causal_head import load_rows
from src.task.CmResidual.tools.probe_train8_causal_head import (
    SOURCE as TRAIN_SOURCE, SOURCE_SHA as TRAIN_SHA,
    SPLIT_SHA, OBJECT_NAMES as TRAIN_OBJECTS, MOTION_NAMES as TRAIN_MOTIONS,
)


TEST_SOURCE = ROOT / "outputs/CmResidual/agent_expert_alarmclock_randomized_s187_h5"
TEST_SHA = "f273b6d04e425b54f0d047e24474fe4afa2859ed7fec12cbc3a7078366aad8ca"


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
                "train_sha256": TRAIN_SHA, "test_sha256": TEST_SHA,
                "split_sha256": SPLIT_SHA, "fit_objects": TRAIN_OBJECTS,
                "heldout_object": "alarmclock", "steps": 300,
                "gpu_count": 0, "cpu_threads": 2, "wall_budget_minutes": 10,
                "output_budget_mb": 10,
                "stop_rule": "source drift, leakage, non-finite or wall budget"}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        torch.set_num_threads(2)
        train = load_rows(source=TRAIN_SOURCE, source_sha=TRAIN_SHA,
                          split_sha=SPLIT_SHA, source_objects=TRAIN_OBJECTS,
                          motion_names=TRAIN_MOTIONS, partition="train")
        train_state, train_difference = inputs(train, TRAIN_MOTIONS, {}, "raw")
        model = fit(train, train_state, train_difference,
                    seed=240990, steps=args.steps)
        model_path = args.output / "model.pt"
        torch.save(model.state_dict(), model_path)
        manifest["model_sha256"] = sha256(model_path)
        # Alarmclock randomized outcomes are loaded only after the fit is frozen.
        test = load_rows(source=TEST_SOURCE, source_sha=TEST_SHA,
                         split_sha=SPLIT_SHA, source_objects=["alarmclock"],
                         motion_names=["alarmclock"], partition="heldout")
        test_state, test_difference = inputs(test, ["alarmclock"], {}, "raw")
        heldout = evaluate(model, test, test_state, test_difference, seed=240991)
        ranked = heldout["ranked_uplift"]
        predicted, actual = heldout["predicted_ate_mm"], heldout["actual_ate_mm"]
        sign_match = (predicted > 0) == (actual > 0)
        ate_error = abs(predicted - actual)
        ranking_pass = (ranked.get("actual_high_minus_low", float("-inf")) >= 10 and
                        ranked.get("cluster_95ci", [float("-inf")])[0] > 0)
        gate = sign_match and ate_error <= 10 and ranking_pass
        report = {"schema": "ref2dex.raw_effect_cm_alarmclock_probe.v1",
                  "classification": "Probe", "source_actor_role": "official_data_collector",
                  "train_rows": len(train["q"]), "heldout_rows": len(test["q"]),
                  "train_randomized_ate_mm": effect_estimate(train),
                  "heldout": heldout, "heldout_ate_error_mm": ate_error,
                  "heldout_sign_match": sign_match,
                  "heldout_ranking_gate_passed": ranking_pass,
                  "continue_to_short_horizon_candidate_check": gate,
                  "elapsed_seconds": time.monotonic() - started}
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest.update(run_status="COMPLETED", report_sha256=sha256(report_path))
        print(json.dumps({"predicted_ate_mm": predicted, "actual_ate_mm": actual,
                          "ate_error_mm": ate_error, "ranked_uplift": ranked,
                          "continue_to_short_horizon_candidate_check": gate},
                         sort_keys=True), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
