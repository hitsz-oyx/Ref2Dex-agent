#!/usr/bin/env python3
"""Materialize a tiny, hash-locked NPZ pack from the official PointWorld H5."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np


CLIPS = ("0:11", "10:21", "25:36", "30:41", "35:46", "40:51", "45:56", "5:16", "65:76", "70:81")
CAMERAS = ("camera_20521388_ext", "camera_24259877_ext")
HORIZON = 10


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists: %s" % args.output)
    args.output.mkdir(parents=True)
    manifest = {
        "schema": "ref2dex.official-video-aux-pack.v1",
        "status": "PREPARING",
        "h5": str(args.h5),
        "h5_sha256": sha(args.h5),
        "h5_bytes": args.h5.stat().st_size,
        "points_per_sample": args.points,
        "context_steps": 1,
        "prediction_steps": HORIZON,
        "clips": list(CLIPS),
        "cameras": list(CAMERAS),
        "selection": [],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    with h5py.File(args.h5, "r") as handle:
        for clip_index, clip in enumerate(CLIPS):
            for camera in CAMERAS:
                group = handle[clip][camera]
                points = np.asarray(group["scene_flows"], dtype=np.float32)
                visibility = np.asarray(group["scene_visibility"], dtype=bool)
                depth_valid = np.asarray(group["scene_depth_valid_mask"], dtype=bool)
                colors = np.asarray(group["scene_colors"], dtype=np.float32)
                normals = np.asarray(group["scene_normals"], dtype=np.float32)
                if points.shape[0] != HORIZON + 1 or points.shape[-1] != 3:
                    raise ValueError("unexpected scene flow shape: %s" % (points.shape,))
                valid = visibility & depth_valid & np.isfinite(points).all(-1)
                eligible = np.flatnonzero(valid.all(0))
                if len(eligible) < args.points:
                    raise ValueError("insufficient all-step-valid points: %s/%s" % (clip, camera))
                selected = eligible[np.linspace(0, len(eligible) - 1, args.points, dtype=np.int64)]
                features = np.zeros((args.points, 18), dtype=np.float32)
                features[:, :3] = colors[0, selected] / 255.0
                features[:, 3:6] = normals[0, selected] / 127.0
                sample = {
                    "xyz": points[0, selected],
                    "features": features,
                    "target_flow": (points[1:, selected] - points[0, selected]).transpose(1, 0, 2),
                    "target_valid": valid[1:, selected].transpose(1, 0),
                }
                if not np.isfinite(sample["features"]).all() or not np.isfinite(sample["target_flow"]).all():
                    raise ValueError("nonfinite official sample")
                name = "%02d_%s_%s.npz" % (clip_index, clip.replace(":", "-"), camera)
                np.savez_compressed(args.output / name, **sample)
                manifest["selection"].append({
                    "name": "%s/%s" % (clip, camera),
                    "clip": clip,
                    "camera": camera,
                    "clip_index": clip_index,
                    "split": "train" if clip_index < 8 else "held_clip",
                    "file": name,
                    "selected_track_count": int(len(eligible)),
                    "sha256": sha(args.output / name),
                })
    manifest["status"] = "QUALIFIED_OFFICIAL_VIDEO_AUX_PACK"
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
