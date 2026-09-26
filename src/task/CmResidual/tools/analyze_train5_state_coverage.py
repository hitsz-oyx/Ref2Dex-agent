"""Audit H10 load-bearing state support under the self-trained train5 actor."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch


ROOT = Path(__file__).resolve().parents[4]
RUN_ROOT = ROOT / "outputs/Dexplore/agent_crossobject_train5_s179_e320"
SOURCES = {
    "train": (RUN_ROOT / "eval_s178_e320_full_coverage_train5",
              "c3192ed5c26be339810c2757291ec7f02790ce0cd1d6b552e7f0130336841745"),
    "heldout": (RUN_ROOT / "eval_s174_e320_full_coverage_alarmclock",
                "046578396f217deabb6618041539c30d169bf9448a27cc369e0dbaec47efa870"),
}
SPLIT_SHA = "694b74889e5d5523be3b96485e5e91bdc31fb420aec2766ec0fb4a2dc40d1c3c"
CHECKPOINT_SHA = "6907c12f8ee4ffa9af22ccae8ffe7599d524e401ff2d8f21b8804185fd5d4961"
MOTION_NAMES = ["cubesmall", "mug", "toothpaste", "waterbottle", "airplane", "cubesmall"]
OBJECT_NAMES = ["airplane", "cubesmall", "mug", "toothpaste", "waterbottle"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(partition: str) -> tuple[dict, dict, dict]:
    directory, expected = SOURCES[partition]
    manifest = json.loads((directory / "run_manifest.json").read_text())
    results = json.loads((directory / "results.json").read_text())
    transition = directory / "transitions.pt"
    expected_seed = 178 if partition == "train" else 174
    expected_root = "train" if partition == "train" else "heldout"
    if (manifest.get("run_status") != "COMPLETED" or
            manifest.get("checkpoint_sha256") != CHECKPOINT_SHA or
            manifest.get("input_manifest_sha256") != SPLIT_SHA or
            Path(manifest["motion_root"]).name != expected_root or
            manifest.get("seed") != expected_seed or manifest.get("num_envs") != 64 or
            manifest.get("transition_sha256") != expected or sha256(transition) != expected):
        raise ValueError(f"{partition} source drift")
    payload = torch.load(transition, map_location="cpu", weights_only=False)
    required = ("q", "action", "object_state", "hand_contact", "object_contact",
                "done", "progress", "data_id")
    if payload.get("schema") != "ref2dex.cmlite_transition.v1" or any(
            key not in payload for key in required):
        raise ValueError(f"{partition} transition schema drift")
    if any(not torch.isfinite(payload[key].float()).all() for key in required):
        raise FloatingPointError(f"non-finite {partition} transition")
    return payload, manifest, results


def windows(payload: dict, motion_names: list[str], horizon: int = 10) -> dict[str, dict]:
    num_envs = 64
    if len(payload["q"]) % num_envs:
        raise ValueError("transition rows are not complete step-major blocks")
    output = {name: {"contact_windows": 0, "positive_windows": 0,
                     "failure_windows": 0, "positive_envs": set(),
                     "failure_envs": set(), "environments": set()}
              for name in sorted(set(motion_names))}
    for env_id in range(num_envs):
        indices = torch.arange(env_id, len(payload["q"]), num_envs)
        ended = payload["done"].reshape(-1)[indices].bool().nonzero().reshape(-1)
        if not len(ended):
            raise ValueError(f"environment {env_id} lacks a complete first episode")
        indices = indices[:int(ended[0])]
        if len(indices) <= horizon:
            continue
        motion_ids = payload["data_id"].reshape(-1)[indices].long()
        if not torch.equal(motion_ids, motion_ids[:1].expand_as(motion_ids)):
            raise ValueError("motion identity changed within a first episode")
        motion_id = int(motion_ids[0])
        if not 0 <= motion_id < len(motion_names):
            raise ValueError("motion ID outside pinned mapping")
        name = motion_names[motion_id]
        item = output[name]
        item["environments"].add(env_id)
        contact = (payload["hand_contact"].reshape(-1)[indices].bool() &
                   payload["object_contact"].reshape(-1)[indices].bool())
        z = payload["object_state"][indices, 2]
        for start in torch.where(contact[:-horizon])[0].tolist():
            future_contact = float(contact[start + 1:start + horizon + 1].float().mean())
            dz_mm = float((z[start + horizon] - z[start]) * 1000)
            positive = dz_mm >= 10 and future_contact >= .5
            failure = future_contact < .5 or dz_mm <= 0
            item["contact_windows"] += 1
            if positive:
                item["positive_windows"] += 1
                item["positive_envs"].add(env_id)
            if failure:
                item["failure_windows"] += 1
                item["failure_envs"].add(env_id)
    for item in output.values():
        for key in ("positive_envs", "failure_envs", "environments"):
            item[key] = len(item[key])
        item["passes_support_gate"] = bool(
            item["positive_windows"] >= 32 and item["failure_windows"] >= 32 and
            item["positive_envs"] >= 4)
    return output


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
        "run_id": args.output.name, "work_version": "crossobject-state-coverage",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                              text=True).strip(),
        "source_sha256": {key: value[1] for key, value in SOURCES.items()},
        "split_sha256": SPLIT_SHA, "checkpoint_sha256": CHECKPOINT_SHA,
        "gpu_count": 0, "cpu_threads": 2, "wall_budget_minutes": 10,
        "stop_rule": "source drift, incomplete first episode, non-finite data or mapping drift",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        torch.set_num_threads(2)
        train, _, train_results = load("train")
        heldout, _, heldout_results = load("heldout")
        by_object = windows(train, MOTION_NAMES)
        heldout_coverage = windows(heldout, ["alarmclock"])
        passing = sum(item["passes_support_gate"] for item in by_object.values())
        report = {
            "schema": "ref2dex.train5_state_coverage.v1", "classification": "Probe",
            "horizon_steps": 10,
            "positive_definition": "current contact; H10 dz >=10mm and future contact fraction >=0.5",
            "failure_definition": "current contact; H10 future contact fraction <0.5 or dz <=0",
            "by_train_object": by_object, "heldout_diagnostic": heldout_coverage,
            "train_lift_success_rate": train_results["summary"]["lift_success_rate"],
            "heldout_lift_success_rate": heldout_results["summary"]["lift_success_rate"],
            "passing_train_objects": passing,
            "adequate_representation_support": bool(passing >= 4),
            "decision": ("proceed_to_dense_interaction_representation_probe" if passing >= 4 else
                         "policy_state_coverage_is_blocker"),
            "limits": ["Overlapping H10 windows are coverage counts, not independent samples.",
                       "This audit does not measure Cm accuracy or policy utility."],
        }
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest.update(run_status="COMPLETED", completed_at=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.monotonic() - started,
                        report_sha256=sha256(report_path))
        print(json.dumps({"passing_train_objects": passing,
                          "adequate_representation_support": passing >= 4,
                          "by_train_object": by_object}, sort_keys=True))
    except BaseException as error:
        manifest.update(run_status="FAILED", completed_at=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.monotonic() - started,
                        failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
