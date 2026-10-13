#!/usr/bin/env python3
"""Small real-data probe with preserved depth/track diagnostics on rejection."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/"src"))
import numpy as np
from video_data_preparation.backends import AnnotatedMasks, CoTrackerBackend, OmegaDepthBackend
from video_data_preparation.geometry import resize_intrinsic, calibrate_scale
from video_data_preparation.io import atomic_json
from video_data_preparation.native_hand import read_native
from video_data_preparation.pipeline import process_window, ProcessingConfig
from video_data_preparation.shards import ShardWriter, freeze_snapshot
from video_data_preparation.job_ledger import stable_split
from video_data_preparation.video import VideoFrames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seconds", type=int, default=1200)
    args = parser.parse_args()
    if args.output.exists() or not 30 <= args.seconds <= 1200:
        raise ValueError("fresh probe output and bounded deadline required")
    if ROOT not in args.output.resolve().parents:
        raise ValueError("probe output must be inside this workspace")
    if subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid",
                                "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("assigned GPU occupied")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    scratch = ROOT/"tmp"/args.output.name
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(scratch), TORCH_EXTENSIONS_DIR=str(scratch/"extensions"))
    import torch
    import cv2
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    inputs = json.loads(args.inputs.read_text())
    if not 1 <= len(inputs) <= 3 or sum(r["hdf5_bytes"]+r["video_bytes"] for r in inputs) > 100*2**20:
        raise ValueError("at most three original pairs and 100MiB input bytes")
    args.output.mkdir(parents=True)
    config = ProcessingConfig()
    assets = args.assets.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = dict(status="RUNNING", git_commit=commit, work_version=commit, run_id=args.output.name,
        pid=os.getpid(), gpu=args.gpu, max_seconds=args.seconds, input=str(args.inputs.resolve()),
        annotations=str(args.annotations.resolve()), config=asdict(config), training=False,
        asset_root=str(assets), vendor_commits={n: subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=assets/"vendor"/n, text=True).strip() for n in ["vggt-omega", "co-tracker"]})
    atomic_json(args.output/"manifest.json", manifest)
    started = time.monotonic()
    def event(phase, **values):
        torch.cuda.synchronize()
        row = dict(phase=phase, elapsed_s=time.monotonic()-started,
            allocated_mib=torch.cuda.memory_allocated()/2**20,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, **values)
        atomic_json(args.output/"progress.json", row)
        with (args.output/"events.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+"\n")
        print(json.dumps(row), flush=True)
        if time.monotonic()-started > args.seconds:
            raise TimeoutError("probe deadline reached")
        pids = set(map(int, subprocess.check_output(["nvidia-smi", "-i", str(args.gpu),
            "--query-compute-apps=pid", "--format=csv,noheader"], text=True).split()))
        if pids-{os.getpid()}:
            raise RuntimeError("foreign GPU process")
    depth_model = OmegaDepthBackend(assets/"vendor/vggt-omega", assets/"vggt_omega_1b_512.pt", scratch)
    event("DEPTH_MODEL_LOADED")
    tracker_model = CoTrackerBackend(assets/"vendor/co-tracker", assets/"cotracker3-scaled-offline.pth")
    event("TRACKER_MODEL_LOADED")
    mask = AnnotatedMasks(args.annotations)
    hooks = []
    for block_id, block in enumerate(depth_model.model.aggregator.inter_frame_blocks):
        if (block_id+1) % 8 == 0:
            hooks.append(block.register_forward_hook(lambda module, ins, out, n=block_id+1: event("DEPTH_BLOCK", block=n)))
    rows = []
    try:
        for number, source in enumerate(inputs):
            path = Path(source["local_hdf5"])
            native = read_native(path)
            tick = source["tick"]
            folder = args.output/("episode-"+path.stem)
            folder.mkdir()
            with VideoFrames(path.with_suffix(".mp4"), native.frames) as video:
                rgb = video.get(range(tick-3, tick+25))
                intrinsic = resize_intrinsic(native.intrinsic, video.original_wh, video.size)
            begin = time.monotonic()
            class RecordDepth:
                def predict(self, frames, ids, role):
                    event("DEPTH_START", episode=path.stem, role=role, frames=len(ids))
                    before = time.monotonic()
                    result = depth_model.predict(frames, ids, role)
                    np.savez_compressed(folder/(role+"-depth.npz"), depth=result, frame_ids=ids)
                    event("DEPTH_DONE", episode=path.stem, role=role, seconds=time.monotonic()-before)
                    return result
            class RecordTracker:
                def track(self, frames, uv):
                    event("TRACKING_START", episode=path.stem, queries=len(uv))
                    before = time.monotonic()
                    result = tracker_model.track(frames, uv)
                    np.savez_compressed(folder/"tracks.npz", uv=result[0], visible=result[1], current_uv=uv)
                    event("TRACKING_DONE", episode=path.stem, seconds=time.monotonic()-before)
                    return result
            row = dict(episode=path.stem, source=source["member"], key=source["key"], tick=tick,
                       status="RUNNING", split=stable_split(native.session))
            try:
                sample, meta, audit = process_window(native, source["key"], number, tick, rgb,
                    intrinsic, mask, RecordDepth(), RecordTracker(), config)
                record = mask.meta["episodes"][source["key"]]
                meta.update(raw_hdf5=str(path.resolve()),
                            annotation_limitations=record.get("annotation_source"),
                            training_allowed=bool(record.get("rigid_eligible", False)))
                np.savez_compressed(folder/"sample.npz", **sample)
                np.savez_compressed(folder/"audit.npz", **audit)
                atomic_json(folder/"metadata.json", meta)
                if meta["training_allowed"]:
                    writer = ShardWriter(args.output/"shards"/(path.stem+".h5"), row["split"],
                                         native.session, "real-probe-fixed-profile")
                    writer.append(sample, meta, audit)
                    writer.publish()
                row.update(status="QUALIFIED" if meta["training_allowed"] else "DIAGNOSTIC_ONLY",
                    current_query_count=meta["current_query_count"], endpoint_fraction=meta["endpoint_fraction"],
                    per_horizon_valid=meta["per_horizon_valid"],
                    current_depth_agreement=meta["current_depth_agreement"], timings=meta["timings"],
                    endpoint_displacement_m=float(np.median(np.linalg.norm(sample["scene_target"][0,-1][sample["object_goal_valid"]],axis=-1))))
            except (ValueError, KeyError) as error:
                row.update(status="REJECTED", reason=str(error))
            # Preserve calibration diagnostics even if the pipeline rejected early.
            for role in ("observation", "label"):
                cache = folder/(role+"-depth.npz")
                if cache.exists():
                    with np.load(cache) as data:
                        ids = data["frame_ids"]
                        try:
                            cal = calibrate_scale(data["depth"], native.hand[ids], native.valid[ids],
                                intrinsic, native.camera[ids], min_samples=config.min_scale_samples,
                                max_relative_iqr=config.max_scale_relative_iqr)
                            row[role+"_calibration"] = cal
                        except ValueError as error:
                            row[role+"_calibration_error"] = str(error)
            row["seconds"] = time.monotonic()-begin
            atomic_json(folder/"result.json", row)
            rows.append(row)
            event("WINDOW_DONE", result=row)
        if (args.output/"shards").exists():
            freeze_snapshot(args.output, args.output/"snapshot.json", require_both=False)
        manifest["status"] = "COMPLETED"
    except BaseException as error:
        manifest.update(status="FAILED", error=repr(error))
        raise
    finally:
        for hook in hooks:
            hook.remove()
        manifest.update(elapsed_s=time.monotonic()-started, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        atomic_json(args.output/"manifest.json", manifest)
        atomic_json(args.output/"result.json", dict(status=manifest["status"], windows=rows,
                                                  elapsed_s=manifest["elapsed_s"], scope="local real-data admission Probe; no training"))


if __name__ == "__main__":
    main()
