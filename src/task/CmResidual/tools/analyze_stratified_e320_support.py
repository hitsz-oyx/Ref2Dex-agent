"""Build the fixed object-balanced H10 support report from frozen e320 rollouts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import torch

from src.task.CmResidual.tools.analyze_train5_state_coverage import ROOT, sha256, windows


CHECKPOINT_SHA = "6907c12f8ee4ffa9af22ccae8ffe7599d524e401ff2d8f21b8804185fd5d4961"
SOURCES = {
    "mixed": {
        "root": ROOT / "outputs/Dexplore/agent_crossobject_train5_s179_e320/eval_s178_e320_full_coverage_train5",
        "transition_sha": "c3192ed5c26be339810c2757291ec7f02790ce0cd1d6b552e7f0130336841745",
        "split_sha": "694b74889e5d5523be3b96485e5e91bdc31fb420aec2766ec0fb4a2dc40d1c3c",
        "mapping": ["cubesmall", "mug", "toothpaste", "waterbottle", "airplane", "cubesmall"],
    },
    "waterbottle": {
        "root": ROOT / "outputs/Dexplore/agent_crossobject_train5_s179_e320/eval_s178_e320_full_waterbottle_baseline",
        "transition_sha": "1eb857ff5d68b6a62e95547f9f3a485e087d8e11ad8091ac39d61f8690dfd305",
        "split_sha": "6b7e6afc233f626258dce2c7c774606677313b1714f8b6150734e2b997c0bb04",
        "mapping": ["waterbottle"],
    },
    "toothpaste": {
        "root": ROOT / "outputs/Dexplore/agent_crossobject_train5_s179_e320/eval_s178_e320_full_toothpaste_support",
        "transition_sha": "a1cc38689b5da5aa0ea56d8fb6153306f8e8d930a1cdf3e24e52839559ca3cf5",
        "split_sha": "05d710a1f3c3e6ef4c31caab0828b4eeced05a7bc91e316fd123919177ce4a78",
        "mapping": ["toothpaste"],
    },
}


def load(name: str) -> dict[str, dict]:
    source = SOURCES[name]
    manifest = json.loads((source["root"] / "run_manifest.json").read_text())
    path = source["root"] / "transitions.pt"
    if (manifest.get("run_status") != "COMPLETED" or
            manifest.get("checkpoint_sha256") != CHECKPOINT_SHA or
            manifest.get("input_manifest_sha256") != source["split_sha"] or
            manifest.get("transition_sha256") != source["transition_sha"] or
            sha256(path) != source["transition_sha"]):
        raise ValueError(f"{name} source drift")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return windows(payload, source["mapping"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output directory required")
    args.output.mkdir(parents=True)
    manifest_path = args.output / "run_manifest.json"
    manifest = {
        "run_status": "STARTED", "started_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.output.name, "work_version": "stratified-e320-support",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                              text=True).strip(),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "sources": {name: {key: str(value) if isinstance(value, Path) else value
                           for key, value in source.items() if key != "mapping"}
                    for name, source in SOURCES.items()},
        "gpu_count": 0, "cpu_threads": 2,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        mixed, water, toothpaste = load("mixed"), load("waterbottle"), load("toothpaste")
        selected = {name: mixed[name] for name in ("airplane", "cubesmall", "mug")}
        selected["toothpaste"] = toothpaste["toothpaste"]
        selected["waterbottle"] = water["waterbottle"]
        passing = sum(item["passes_support_gate"] for item in selected.values())
        report = {
            "schema": "ref2dex.stratified_e320_support.v1", "classification": "Probe",
            "selected_source_by_object": {"airplane": "mixed", "cubesmall": "mixed",
                                          "mug": "mixed", "toothpaste": "toothpaste",
                                          "waterbottle": "waterbottle"},
            "by_train_object": selected, "passing_train_objects": passing,
            "adequate_representation_support": bool(passing == 5),
            "decision": ("proceed_to_dense_interaction_representation_probe" if passing == 5
                         else "redesign_collection_or_target"),
            "limits": ["Rollouts are stratified coverage data, not the deployment mixture.",
                       "All sources use the same frozen e320 checkpoint and evaluation seed."],
        }
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest.update(run_status="COMPLETED", completed_at=datetime.now(timezone.utc).isoformat(),
                        report_sha256=sha256(report_path))
        print(json.dumps({"passing_train_objects": passing,
                          "adequate_representation_support": passing == 5}, sort_keys=True))
    except BaseException as error:
        manifest.update(run_status="FAILED", completed_at=datetime.now(timezone.utc).isoformat(),
                        failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
