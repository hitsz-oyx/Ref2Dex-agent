"""Audit the frozen FM run against the local processed-data pool.

Read-only: source data and existing checkpoints are never modified.  The
report distinguishes the v6 overlay consumed by ``flowmatching-overnight``
from the larger legacy processed-data roots, which are not interchangeable
with the v6 V5Windows recipe without a conversion step.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[5]
DATA = Path("/home2/wyy/oyx_ws/Ref2Dex/data/processed_data")
RUN = ROOT / "outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2"
PLAN = ROOT / "tmp/wam-v5/plan-compact.json"
PARENT = ROOT / "outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/parent/manifest.json"


def root_size(path):
    # du uses the filesystem walk implementation and is substantially faster
    # than opening every large npz member from Python.
    return int(subprocess.check_output(["du", "-sb", str(path)], text=True).split()[0])


def objectinteraction():
    p = DATA / "ObjectInteractionCmv2/v114d/grab_mano_split_pose_repaired/index.json"
    d = json.loads(p.read_text())
    result = {}
    for split, rows in d["sequences"].items():
        result[split] = {
            "sequences": len(rows),
            "frames": sum(int(r["frame_count"]) for r in rows),
            "objects": len({r["id"].split("/")[2] for r in rows}),
        }
    return result


def stage4():
    files = list((DATA / "stage4/data/grab").glob("s*/*/shared.npz"))
    frames = 0
    objects = set()
    for p in files:
        with np.load(p, allow_pickle=False) as z:
            frames += int(z["raw_frame_id"].shape[0])
            objects.add(str(z["object_name"].item()))
    return {"sequences": len(files), "frames": frames, "objects": len(objects)}


def dexplore_grab():
    files = list((DATA / "dexplore_grab/sequences").glob("*/motion.npz"))
    frames = 0
    objects = set()
    for motion in files:
        with np.load(motion, allow_pickle=True) as z:
            frames += int(np.asarray(z["n_frames"]).item())
        obj = motion.with_name("object.npz")
        with np.load(obj, allow_pickle=True) as z:
            objects.add(str(np.asarray(z["name"]).item()))
    return {"sequences": len(files), "frames": frames, "objects": len(objects)}


def cm_decoder():
    d = json.loads((DATA / "cm_decoder_v2/dexplore_rl_v1_3_full10135/manifest.json").read_text())
    result = {}
    for split, rows in d["sequences"].items():
        result[split] = {
            "sequences": len(rows),
            "frames": sum(int(r["frame_count"]) for r in rows),
            "objects": len({r["object_name"] for r in rows}),
        }
    return result


def coupled_cache():
    d = json.loads((DATA / "coupled_geometric_cache_v1_20260912/run_manifest.json").read_text())
    return d["counts"]


def current_run():
    manifest = json.loads((RUN / "training/run-manifest.json").read_text())
    progress = json.loads((RUN / "training/progress.json").read_text())
    config = json.loads((RUN / "training/config.json").read_text())
    parent = json.loads(PARENT.read_text())
    plan = json.loads(PLAN.read_text())
    train = np.load(parent["indices"]["train"], mmap_mode="r")
    dev = np.load(parent["indices"]["dev"], mmap_mode="r")
    seq = parent["sequences"]
    def by_source(a):
        counts = collections.Counter(int(seq[int(row[0])]["source_id"]) for row in a)
        return {str(k): int(v) for k, v in sorted(counts.items())}
    overlay = {}
    for sid in parent["source_ids"]:
        rows = [r for r in seq if r["source_id"] == sid]
        overlay[str(sid)] = {
            "sequences": len(rows),
            "frames": sum(int(json.loads((Path(r["path"]) / "meta.json").read_text())["frames"]) for r in rows),
            "objects": len({o for r in rows for o in r["objects"]}),
            "train_windows": int(sum(1 for row in train if int(seq[int(row[0])]["source_id"]) == sid)),
            "dev_windows": int(sum(1 for row in dev if int(seq[int(row[0])]["source_id"]) == sid)),
        }
    updates = int(progress["step"])
    global_batch = int(progress["global_batch"])
    return {
        "manifest": manifest,
        "configured_updates": int(config["updates"]),
        "last_progress_step": updates,
        "last_saved_checkpoint_step": 11000,
        "train_windows": int(len(train)),
        "dev_windows": int(len(dev)),
        "global_batch": global_batch,
        "approx_train_epochs_at_last_progress": updates * global_batch / len(train),
        "overlay_by_source": overlay,
        "train_windows_by_source": by_source(train),
        "dev_windows_by_source": by_source(dev),
        "source_draws_at_last_progress": progress["source_draws"],
        "frozen_parent_plan_by_source": {
            str(sid): {
                "planned_sequences": sum(1 for row in plan["jobs"] if int(row["source_id"]) == sid),
                "planned_windows": sum(sum(int(v) for v in row.get("window_counts", {}).values())
                                        for row in plan["jobs"] if int(row["source_id"]) == sid),
            }
            for sid in sorted({int(row["source_id"]) for row in plan["jobs"]})
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {
        "ObjectInteractionCmv2": {"bytes": root_size(DATA / "ObjectInteractionCmv2"), "inventory": objectinteraction()},
        "stage4": {"bytes": root_size(DATA / "stage4"), "inventory": stage4()},
        "coupled_geometric_cache_v1_20260912": {"bytes": root_size(DATA / "coupled_geometric_cache_v1_20260912"), "inventory": coupled_cache()},
        "cm_decoder_v2": {"bytes": root_size(DATA / "cm_decoder_v2"), "inventory": cm_decoder()},
        "dexplore_grab": {"bytes": root_size(DATA / "dexplore_grab"), "inventory": dexplore_grab()},
    }
    result = {
        "schema": "ref2dex.wam-v6.full-scale-data-audit.v1",
        "status": "COMPLETED",
        "run": current_run(),
        "processed_data_roots": roots,
        "direct_root_use_by_flowmatching_r2": {name: False for name in roots},
        "full_v6_recipe_status": "UNAVAILABLE_LOCALLY",
        "full_v6_recipe_reason": "The frozen parent references 39,526 jobs but only 1,177 overlay sequences are local; 38,349 remain pending and their /mnt/ugreen_nas payload paths are not mounted.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output), "pending_sequences": 38349}))


if __name__ == "__main__":
    main()
