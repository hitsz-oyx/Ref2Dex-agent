"""Run and audit the frozen twelve-clip official-TRAIN PointWAM qualification."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))
from oakink_wm.pointwam_data import sha256, source_identity
from prepare_pointwam_pilot import write_json


def audit_pack(path: Path, split: str, episode: str) -> dict:
    with np.load(path, allow_pickle=False) as pack:
        checks = {}
        for kind in ("scene", "hand"):
            xyz, valid = pack[kind + "_xyz_m"], pack[kind + "_valid"]
            target, tvalid = pack[kind + "_displacement_m"], pack[kind + "_target_valid"]
            checks[kind + "_finite_missing"] = bool(np.isfinite(xyz[valid]).all() and np.isnan(xyz[~valid]).all())
            checks[kind + "_target_mask"] = bool(np.array_equal(tvalid, valid[1:] & valid[0]))
            checks[kind + "_displacement"] = bool(np.allclose(target[tvalid], (xyz[1:] - xyz[0])[tvalid]))
        checks["source_identity"] = str(pack["source_split"]) == split and str(pack["source_episode_id"]) == episode
        checks["training_exclusion"] = not bool(pack["training_allowed"])
        checks["clock"] = bool(np.allclose(pack["timestamps_s"][::3], pack["source_frame_ids"] / 30))
        checks["hand_order_shape"] = pack["hand_order"].tolist() == ["right", "left"] and pack["hand_xyz_m"].shape == (31, 2, 10, 3)
        checks["unique_identities"] = len(np.unique(pack["track_ids"])) == len(pack["track_ids"])
        checks["instruction"] = bool(str(pack["language"]).strip())
    return dict(checks=checks, passed=all(checks.values()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--depth-vendor", type=Path, required=True)
    p.add_argument("--depth-checkpoint", type=Path, required=True)
    p.add_argument("--tracker-vendor", type=Path, required=True)
    p.add_argument("--tracker-checkpoint", type=Path, required=True)
    p.add_argument("--seconds", type=int, default=1800)
    a = p.parse_args()
    if not 1 <= a.seconds <= 1800:
        p.error("group processing is capped at30minutes")
    selected = json.loads(a.selection.read_text())
    source = json.loads(a.source_manifest.read_text())
    if (len(selected) != 6 or len({r["member"] for r in selected}) != 6
            or any(r["partition"] not in ("fit_candidate", "dev_candidate") for r in selected)
            or source["selection_sha256"] != sha256(a.selection) or source["selection"] != selected):
        raise ValueError("frozen selection/partition identity mismatch")
    # Audit all input identities before any GPU subprocess starts.
    identities = []
    for r in selected:
        hdf5 = a.source_manifest.parent / "raw" / r["member"]
        identity = source_identity(hdf5, hdf5.with_suffix(".mp4"), a.source_manifest)
        if identity["split"] != "official_train":
            raise ValueError("test/additional data cannot enter this qualification")
        identities.append(identity)
    pinned = ((a.depth_checkpoint, "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934"),
              (a.tracker_checkpoint, "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"))
    for path, expected in pinned:
        if sha256(path) != expected:
            raise ValueError("teacher checkpoint differs from the frozen pilot")
    for path, expected in ((a.depth_vendor, "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6"),
                           (a.tracker_vendor, "82e02e8029753ad4ef13cf06be7f4fc5facdda4d")):
        if subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip() != expected:
            raise ValueError("teacher source differs from the frozen pilot")
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = dict(status="RUNNING", run_id=a.output.name, training_allowed=False,
                  pid=os.getpid(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  source_manifest_sha256=sha256(a.source_manifest), selection=selected,
                  selection_sha256=sha256(a.selection), seconds_cap=a.seconds, episodes=[], clips=[])

    def save(stage: str) -> None:
        report.update(stage=stage, elapsed_s=time.monotonic() - start)
        write_json(a.output / "progress.json", report)
        print(json.dumps(dict(stage=stage, episodes_done=len(report["episodes"]),
                              clips_done=len(report["clips"]), elapsed_s=report["elapsed_s"])), flush=True)

    try:
        for index, (r, identity) in enumerate(zip(selected, identities)):
            remaining = a.seconds - (time.monotonic() - start)
            if remaining < 1:
                raise TimeoutError("group deadline")
            hdf5 = a.source_manifest.parent / "raw" / r["member"]
            run = a.output / (f"episode{index:02d}-" + hdf5.parent.name + "-" + hdf5.stem)
            command = [sys.executable, "-u", str(TASK / "tools/run/prepare_pointwam_pilot.py"),
                       "--hdf5", str(hdf5), "--video", str(hdf5.with_suffix(".mp4")),
                       "--source-manifest", str(a.source_manifest),
                       "--palm-offsets", str(TASK / "configs/pointwam_pilot_palm_offsets.json"),
                       "--depth-backend", "omega", "--depth-vendor", str(a.depth_vendor),
                       "--depth-checkpoint", str(a.depth_checkpoint),
                       "--tracker-vendor", str(a.tracker_vendor),
                       "--tracker-checkpoint", str(a.tracker_checkpoint),
                       "--output", str(run), "--clips", "2", "--query-batch", "4096",
                       "--seconds", str(int(remaining))]
            report["current_episode"] = identity["episode_id"]
            report["current_log"] = str(run.with_suffix(".log").resolve())
            save("PROCESSING_EPISODE")
            with run.with_suffix(".log").open("x") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
            result = json.loads((run / "manifest.json").read_text())
            report["episodes"].append(dict(member=r["member"], partition=r["partition"],
                                            exit_code=completed.returncode, status=result["status"],
                                            manifest=str((run / "manifest.json").resolve()),
                                            error=result.get("error"), command=command))
            # Data shortages are measured outcomes. Implementation/input/clock
            # errors stop the group rather than retrying them on other episodes.
            shortages = ("no clip passes", "insufficient hand/depth scale samples",
                         "insufficient hand/depth scale inliers", "no initial visible/depth-valid")
            if completed.returncode and not any(s in result.get("error", "") for s in shortages):
                raise RuntimeError(f"episode contract/execution failure: {result.get('error')}")
            for c in result["clips"]:
                pack = Path(c["path"]) / "pointwam_clip.npz"
                audit = audit_pack(pack, identity["split"], identity["episode_id"])
                qualified = (c["engineering_gates_pass"] and audit["passed"]
                             and c["all_future_valid_tracks"] >= 512 and c["scene_valid_fraction"] >= .5)
                report["clips"].append(dict(member=r["member"], task=hdf5.parent.name,
                                              episode_id=identity["episode_id"], partition=r["partition"],
                                              pack=str(pack.resolve()), pack_sha256=sha256(pack),
                                              qualified=bool(qualified), pack_audit=audit, teacher_audit=c))
            if sum(p.stat().st_size for p in a.output.rglob("*") if p.is_file()) > 2**30:
                raise RuntimeError("processed1GiB cap exceeded")
            save("EPISODE_COMPLETE")
        passed = [c for c in report["clips"] if c["qualified"]]
        gates = dict(qualified_clips_ge8=len(passed) >= 8,
                     tasks_ge2=len({c["task"] for c in passed}) >= 2,
                     fit_records_ge2=len({c["episode_id"] for c in passed if c["partition"] == "fit_candidate"}) >= 2,
                     dev_records_ge2=len({c["episode_id"] for c in passed if c["partition"] == "dev_candidate"}) >= 2)
        report.update(status="COMPLETED", qualified_clips=len(passed), gates=gates,
                      gate_pass=all(gates.values()), result="PROMISING" if all(gates.values()) else "UNCLEAR")
        save("COMPLETED")
    except Exception as exc:
        report.update(status="FAILED", error=repr(exc), result="UNCLEAR")
        save("FAILED")
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - start
        write_json(a.output / "manifest.json", report)


if __name__ == "__main__":
    main()
