"""Apply the fixed train5 H10 support gate to the balanced e360 actor."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import torch

from src.task.CmResidual.tools.analyze_train5_state_coverage import (
    CHECKPOINT_SHA as E320_SHA, OBJECT_NAMES, ROOT, SPLIT_SHA, sha256, windows,
)


SOURCE = (ROOT / "outputs/Dexplore/agent_crossobject_train5_balanced_s179_e360/"
          "eval_s178_e360_full_coverage_train5")
TRANSITION_SHA = "a262959dd23987c3ea8e32224c259274923658d9f0f4fadee9c25f6144df0dfc"
CHECKPOINT_SHA = "a41fd8281dcf4639579a9cd71baa104f007969cf0642c506d3d358507025f03f"
MOTION_NAMES = ["cubesmall", "mug", "toothpaste", "waterbottle", "airplane"]
BASELINE = ROOT / "outputs/CmResidual/agent_train5_state_coverage_s178174/report.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output directory required")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    manifest_path = args.output / "run_manifest.json"
    manifest = {
        "run_status": "STARTED", "started_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.output.name, "work_version": "crossobject-balanced-coverage",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                              text=True).strip(),
        "source_transition_sha256": TRANSITION_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA, "parent_checkpoint_sha256": E320_SHA,
        "split_sha256": SPLIT_SHA, "gpu_count": 0, "cpu_threads": 2,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        source_manifest = json.loads((SOURCE / "run_manifest.json").read_text())
        path = SOURCE / "transitions.pt"
        if (source_manifest.get("run_status") != "COMPLETED" or
                source_manifest.get("checkpoint_sha256") != CHECKPOINT_SHA or
                source_manifest.get("input_manifest_sha256") != SPLIT_SHA or
                source_manifest.get("cfg_env") !=
                "dexplore/data/cfg/inspire_object_balanced.yaml" or
                source_manifest.get("transition_sha256") != TRANSITION_SHA or
                sha256(path) != TRANSITION_SHA):
            raise ValueError("balanced evaluation source drift")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "ref2dex.cmlite_transition.v1":
            raise ValueError("transition schema drift")
        by_object = windows(payload, MOTION_NAMES)
        baseline = json.loads(BASELINE.read_text())["by_train_object"]
        passing = sum(item["passes_support_gate"] for item in by_object.values())
        lost = sorted(name for name in OBJECT_NAMES
                      if baseline[name]["passes_support_gate"] and
                      not by_object[name]["passes_support_gate"])
        results = json.loads((SOURCE / "results.json").read_text())
        held_lift = {name: {"successes": 0, "episodes": 0} for name in OBJECT_NAMES}
        for episode in results["per_episode"]:
            name = MOTION_NAMES[int(episode["motion_id"])]
            held_lift[name]["successes"] += int(episode["lift_success"])
            held_lift[name]["episodes"] += 1
        gate = passing >= 4 and not lost
        report = {
            "schema": "ref2dex.train5_balanced_coverage.v1", "classification": "Probe",
            "by_train_object": by_object, "held_lift_by_object": held_lift,
            "passing_train_objects": passing, "previously_passing_objects_lost": lost,
            "continue_to_dense_cm_representation": bool(gate),
            "decision": ("proceed_to_dense_interaction_representation_probe" if gate else
                         "simple_balancing_is_unpromising"),
            "limits": ["Same evaluation seed as the baseline coverage audit.",
                       "This is a state-support Probe, not policy-utility evidence."],
        }
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest.update(run_status="COMPLETED", completed_at=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.monotonic() - started,
                        report_sha256=sha256(report_path))
        print(json.dumps({"passing_train_objects": passing,
                          "previously_passing_objects_lost": lost,
                          "continue_to_dense_cm_representation": gate,
                          "held_lift_by_object": held_lift}, sort_keys=True))
    except BaseException as error:
        manifest.update(run_status="FAILED", completed_at=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.monotonic() - started,
                        failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
