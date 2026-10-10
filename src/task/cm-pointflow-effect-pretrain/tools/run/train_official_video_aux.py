#!/usr/bin/env python3
"""Train a random-initialized short-horizon auxiliary model on official H5 data.

This is deliberately separate from the native H=4/K=24 learner.  PointWorld's
released H5 contract is one context scene and ten future scene-flow steps.  The
script consumes that contract directly, keeps only tracks valid for the whole
clip, and never loads a PointWorld checkpoint.  The resulting model is a
learnability Probe; it is not a native Cm transfer claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))
from oakink_wm.video_pointworld import VideoPointWorldWM  # noqa: E402


CLIPS = ("0:11", "10:21", "25:36", "30:41", "35:46", "40:51", "45:56", "5:16", "65:76", "70:81")
CAMERAS = ("camera_20521388_ext", "camera_24259877_ext")
HORIZON = 10


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_samples(pack: Path, points_per_sample: int):
    samples = []
    pack_manifest = json.loads((pack / "manifest.json").read_text())
    if pack_manifest.get("status") != "QUALIFIED_OFFICIAL_VIDEO_AUX_PACK":
        raise ValueError("official auxiliary pack is not qualified")
    if int(pack_manifest["points_per_sample"]) != points_per_sample:
        raise ValueError("pack point count differs from requested count")
    for entry in pack_manifest["selection"]:
        path = pack / entry["file"]
        if sha(path) != entry["sha256"]:
            raise ValueError("official auxiliary pack hash drift: %s" % path)
        with np.load(path, allow_pickle=False) as arrays:
            sample = {key: arrays[key].copy() for key in
                      ("xyz", "features", "target_flow", "target_valid")}
        sample.update({key: entry[key] for key in ("name", "clip_index", "split", "selected_track_count")})
        samples.append(sample)
    return samples


def batch_from(samples, device):
    count = max(len(x["xyz"]) for x in samples)
    xyz = np.zeros((len(samples), count, 3), dtype=np.float32)
    features = np.zeros((len(samples), count, 18), dtype=np.float32)
    target = np.zeros((len(samples), count, HORIZON, 3), dtype=np.float32)
    target_valid = np.zeros((len(samples), count, HORIZON), dtype=bool)
    point_valid = np.zeros((len(samples), count), dtype=bool)
    for i, sample in enumerate(samples):
        n = len(sample["xyz"])
        xyz[i, :n] = sample["xyz"]
        features[i, :n] = sample["features"]
        target[i, :n] = sample["target_flow"]
        target_valid[i, :n] = sample["target_valid"]
        point_valid[i, :n] = True
    return {
        "xyz": torch.from_numpy(xyz).to(device),
        "features": torch.from_numpy(features).to(device),
        "point_valid": torch.from_numpy(point_valid).to(device),
        "target_flow": torch.from_numpy(target).to(device),
        "target_valid": torch.from_numpy(target_valid).to(device),
    }


def aux_loss(prediction, batch):
    mask = batch["target_valid"] & batch["point_valid"][..., None]
    if not bool(mask.any()):
        raise ValueError("official auxiliary batch has no valid targets")
    # Keep the same 1 cm output scale used by the existing video learner.
    return F.huber_loss(prediction[:, :, :HORIZON][mask] / 0.01,
                        batch["target_flow"][mask] / 0.01, reduction="mean")


@torch.no_grad()
def evaluate(model, samples, device):
    model.eval()
    accum = {name: [0.0, 0] for name in ("h1", "h5", "h10")}
    static_accum = {name: [0.0, 0] for name in ("h1", "h5", "h10")}
    per_sample = {}
    for sample in samples:
        batch = batch_from([sample], device)
        prediction = model({key: batch[key] for key in ("xyz", "features", "point_valid")})
        error = torch.linalg.vector_norm(
            prediction[:, :, :HORIZON] - batch["target_flow"], dim=-1
        )[0]
        row = {}
        for horizon, suffix in ((1, "h1"), (5, "h5"), (10, "h10")):
            mask = batch["target_valid"][0, :, :horizon]
            value = error[:, :horizon][mask]
            row[suffix] = float(value.mean()) if bool(mask.any()) else None
            if bool(mask.any()):
                accum[suffix][0] += float(value.sum())
                accum[suffix][1] += int(mask.sum())
                static_value = np.linalg.norm(
                    sample["target_flow"][:, :horizon], axis=-1
                )[sample["target_valid"][:, :horizon]]
                static_accum[suffix][0] += float(static_value.sum())
                static_accum[suffix][1] += int(static_value.size)
        per_sample[sample["name"]] = row
    model.train()
    return {
        "model": {
            key: (value / count if count else None)
            for key, (value, count) in accum.items()
        },
        "static": {
            key: (value / count if count else None)
            for key in ("h1", "h5", "h10")
            for value, count in [static_accum[key]]
        },
        "per_sample": per_sample,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene-stats", type=Path,
                        help="reuse a native train-only scene_mean/std contract")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists: %s" % args.output)
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    samples = load_samples(args.pack, args.points)
    pack_manifest = json.loads((args.pack / "manifest.json").read_text())
    train = [x for x in samples if x["split"] == "train"]
    held = [x for x in samples if x["split"] == "held_clip"]
    feature_matrix = np.concatenate([x["features"] for x in train], axis=0)
    if args.scene_stats:
        scene_stats = json.loads(args.scene_stats.read_text())
        mean = np.asarray(scene_stats["scene_mean"], dtype=np.float32)
        std = np.asarray(scene_stats["scene_std"], dtype=np.float32)
        if mean.shape != (18,) or std.shape != (18,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("native scene stats must contain finite 18-D scene_mean/scene_std")
        if (std <= 0).any():
            raise ValueError("native scene stats must have positive scene_std")
    else:
        mean = feature_matrix.mean(0)
        std = feature_matrix.std(0)
        std = np.where(std < 1e-6, 1.0, np.maximum(std, 1e-3))
    # The source contract is color/normal + explicit zero placeholders; use
    # train-only statistics and leave constant placeholders identity-scaled.
    model = VideoPointWorldWM(mean, std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    manifest = {
        "schema": "ref2dex.official-video-aux-result.v1",
        "status": "RUNNING",
        "h5_sha256": pack_manifest["h5_sha256"],
        "h5_bytes": pack_manifest["h5_bytes"],
        "pack_manifest_sha256": sha(args.pack / "manifest.json"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": args.seed,
        "steps": args.steps,
        "points_per_sample": args.points,
        "clips": list(CLIPS),
        "cameras": list(CAMERAS),
        "train_samples": len(train),
        "held_samples": len(held),
        "protocol": {
            "context_steps": 1,
            "prediction_steps": HORIZON,
            "target": "official_scene_flows[t+1:t+11] - scene_flows[t]",
            "feature_contract": pack_manifest.get("feature_contract"),
            "state_features": bool(pack_manifest.get("state_features", False)),
            "native_contract": bool(pack_manifest.get("native_contract", False)),
            "scene_stats_source": str(args.scene_stats) if args.scene_stats else "train-pack-only",
            "point_selection": "all-step visibility/depth-valid tracks, deterministic linspace",
                "official_checkpoint_loaded": False,
            "zed_used": False,
        },
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
    }
    (args.output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    started = time.monotonic()
    initial = evaluate(model, held, device)
    (args.output / "initial-held.json").write_text(
        json.dumps(initial, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    progress_path = args.output / "progress.jsonl"
    with progress_path.open("w", buffering=1) as log:
        for step in range(1, args.steps + 1):
            indices = np.random.randint(len(train), size=2)
            batch = batch_from([train[int(i)] for i in indices], device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model({key: batch[key] for key in ("xyz", "features", "point_valid")})
            loss = aux_loss(prediction, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite official auxiliary loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite official auxiliary gradient")
            optimizer.step()
            if step == 1 or step % 50 == 0 or step == args.steps:
                elapsed = time.monotonic() - started
                record = {
                    "step": step,
                    "loss": float(loss),
                    "grad_norm": float(norm),
                    "elapsed_seconds": elapsed,
                    "updates_per_second": step / max(elapsed, 1e-6),
                    "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024 ** 2
                    if torch.cuda.is_available() else 0.0,
                }
                if torch.cuda.is_available():
                    record["nvml_gpu_rows"] = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                         "--format=csv,noheader,nounits"], text=True
                    ).strip().splitlines()
                log.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
    final = evaluate(model, held, device)
    torch.save({"model": model.state_dict(), "seed": args.seed, "steps": args.steps,
                "h5_sha256": manifest["h5_sha256"]}, args.output / "final.pt")
    manifest.update({
        "status": "COMPLETED",
        "elapsed_seconds": time.monotonic() - started,
        "initial_held": initial,
        "final_held": final,
        "conclusion": "PROMISING" if (
            final["model"]["h10"] is not None
            and final["model"]["h10"] <= 0.9 * final["static"]["h10"]
        ) else "UNPROMISING",
    })
    (args.output / "result.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
