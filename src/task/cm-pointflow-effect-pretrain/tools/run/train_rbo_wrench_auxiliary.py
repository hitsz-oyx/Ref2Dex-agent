#!/usr/bin/env python3
"""Probe temporal learnability of the bounded RBO measured-wrench adapter.

This is deliberately a small auxiliary forecast, not a CHORD or object-COM
supervision claim.  It consumes one already-adapted interaction, keeps a
chronological tail held out, and compares a short-history MLP with persistence
for both the raw sensor-frame and geometric object-frame wrench layers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


HISTORY = 4
HORIZON = 8
TARGETS = {
    "sensor": "wrench_sensor_myframe",
    "geometric": "wrench_object_geometric",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_features(arrays):
    """Build finite per-frame geometry/state features from the adapter output."""
    points = arrays["world_points"].astype(np.float32)
    valid = arrays["point_valid"].astype(bool)
    pose = arrays["object_pose"].astype(np.float32)
    joint = arrays["joint_position"].astype(np.float32)
    rows = []
    for t in range(len(points)):
        selected = points[t][valid[t]]
        if selected.size == 0 or not np.isfinite(selected).all():
            raise ValueError(f"frame {t} has no finite valid points")
        centroid = selected.mean(0)
        spread = selected.std(0)
        translation = pose[t, :3, 3]
        rotation6 = pose[t, :3, :2].reshape(-1)
        row = np.concatenate((centroid, spread, translation, rotation6,
                              np.asarray([joint[t]], dtype=np.float32)))
        rows.append(row)
    features = np.asarray(rows, dtype=np.float32)
    if features.shape[1] != 16 or not np.isfinite(features).all():
        raise ValueError(f"invalid geometry/state features: {features.shape}")
    return features


def windows(context, target, starts):
    x = np.stack([context[s:s + HISTORY] for s in starts]).astype(np.float32)
    y = np.stack([target[s + HISTORY:s + HISTORY + HORIZON] for s in starts]).astype(np.float32)
    return x, y


class WrenchForecaster(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1)).reshape(x.shape[0], HORIZON, 6)


def metric_rows(pred, target, baseline):
    rows = {}
    for horizon, suffix in ((1, "h1"), (4, "h4"), (8, "h8")):
        p = pred[:, :horizon]
        y = target[:, :horizon]
        b = baseline[:, :horizon]
        rows[suffix] = {
            "model_component_mae": float(np.abs(p - y).mean()),
            "persistence_component_mae": float(np.abs(b - y).mean()),
            "model_force_mae": float(np.abs(p[..., :3] - y[..., :3]).mean()),
            "persistence_force_mae": float(np.abs(b[..., :3] - y[..., :3]).mean()),
            "model_torque_mae": float(np.abs(p[..., 3:] - y[..., 3:]).mean()),
            "persistence_torque_mae": float(np.abs(b[..., 3:] - y[..., 3:]).mean()),
            "model_vector_l2": float(np.linalg.norm(p - y, axis=-1).mean()),
            "persistence_vector_l2": float(np.linalg.norm(b - y, axis=-1).mean()),
        }
        persistence = rows[suffix]["persistence_vector_l2"]
        model = rows[suffix]["model_vector_l2"]
        rows[suffix]["relative_l2"] = model / persistence if persistence > 0 else None
    return rows


def gpu_rows():
    if not torch.cuda.is_available():
        return []
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], text=True,
        ).strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []


def train_target(name, context, target, fit_starts, held_starts, device,
                 seed, steps, output):
    fit_x, fit_y = windows(context, target, fit_starts)
    held_x, held_y = windows(context, target, held_starts)
    context_mean = fit_x.reshape(-1, fit_x.shape[-1]).mean(0)
    context_std = fit_x.reshape(-1, fit_x.shape[-1]).std(0)
    context_std = np.where(context_std < 1e-6, 1.0, np.maximum(context_std, 1e-5))
    target_mean = fit_y.reshape(-1, 6).mean(0)
    target_std = fit_y.reshape(-1, 6).std(0)
    target_std = np.where(target_std < 1e-6, 1.0, np.maximum(target_std, 1e-5))
    fit_x = (fit_x - context_mean) / context_std
    fit_y = (fit_y - target_mean) / target_std
    held_x_norm = (held_x - context_mean) / context_std
    held_y_norm = held_y.copy()
    model = WrenchForecaster(HISTORY * fit_x.shape[-1], HORIZON * 6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    train_x = torch.from_numpy(fit_x).to(device)
    train_y = torch.from_numpy(fit_y).to(device)
    held_x_t = torch.from_numpy(held_x_norm).to(device)
    started = time.monotonic()
    with torch.no_grad():
        initial = model(held_x_t).cpu().numpy() * target_std + target_mean
    # The last observed wrench is the final six channels in the context window.
    last_wrench = held_x[:, -1, -6:]
    baseline = np.repeat(last_wrench[:, None, :], HORIZON, axis=1)
    initial_metrics = metric_rows(initial, held_y, baseline)
    rng = np.random.default_rng(seed)
    progress = []
    model.train()
    for step in range(1, steps + 1):
        indices = rng.integers(0, len(train_x), size=min(32, len(train_x)))
        prediction = model(train_x[indices])
        loss = F.huber_loss(prediction, train_y[indices], reduction="mean")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite {name} loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"nonfinite {name} gradient")
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            record = {
                "target": name,
                "step": step,
                "loss": float(loss),
                "grad_norm": float(norm),
                "elapsed_seconds": time.monotonic() - started,
                "peak_memory_mib": (torch.cuda.max_memory_allocated() / 1024 ** 2
                                     if torch.cuda.is_available() else 0.0),
                "nvml_gpu_rows": gpu_rows(),
            }
            progress.append(record)
            print(json.dumps(record), flush=True)
    model.eval()
    with torch.no_grad():
        final = model(held_x_t).cpu().numpy() * target_std + target_mean
    metrics = metric_rows(final, held_y, baseline)
    torch.save({"model": model.state_dict(), "seed": seed, "target": name},
               output / f"{name}.pt")
    return {
        "target": name,
        "fit_windows": len(fit_starts),
        "held_windows": len(held_starts),
        "feature_mean": context_mean.tolist(),
        "feature_std": context_std.tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "initial_held": initial_metrics,
        "final_held": metrics,
        "progress": progress,
        "elapsed_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--fit-windows", type=int, default=220)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"output exists: {args.output}")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    arrays = dict(np.load(args.input, allow_pickle=False))
    required = {"anchor_timestamps", "object_pose", "joint_position",
                "world_points", "point_valid", *TARGETS.values()}
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"adapter output missing keys: {sorted(missing)}")
    context_features = frame_features(arrays)
    starts = np.arange(0, len(context_features) - HISTORY - HORIZON + 1, dtype=np.int64)
    if args.fit_windows <= 0 or args.fit_windows >= len(starts):
        raise ValueError(f"fit window count must be in (0,{len(starts)})")
    fit_starts = starts[:args.fit_windows]
    held_starts = starts[args.fit_windows:]
    source_manifest = args.input.parent / "manifest.json"
    result = {
        "schema": "ref2dex.rbo-wrench-auxiliary-result.v1",
        "status": "RUNNING",
        "input": str(args.input),
        "input_sha256": sha(args.input),
        "source_manifest_sha256": sha(source_manifest) if source_manifest.exists() else None,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": args.seed,
        "steps": args.steps,
        "history": HISTORY,
        "horizon": HORIZON,
        "fit_windows": len(fit_starts),
        "held_windows": len(held_starts),
        "split": "chronological first fit windows, final tail held out",
        "feature_contract": "world-point centroid+spread, object pose translation+rotation6, joint position, and past target wrench",
        "payload_compensation": False,
        "force_unit_status": "source visualizer+ROS_Wrench; preserve adapter raw values",
        "targets": {},
    }
    (args.output / "input_manifest.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for offset, (name, key) in enumerate(TARGETS.items()):
        target_seed = args.seed + offset
        result["targets"][name] = train_target(
            name, np.concatenate((context_features, arrays[key]), axis=1),
            arrays[key].astype(np.float32), fit_starts, held_starts, device,
            target_seed, args.steps, args.output,
        )
    ratios = [result["targets"][name]["final_held"]["h8"]["relative_l2"]
              for name in TARGETS]
    if all(r is not None and r <= 0.90 for r in ratios):
        conclusion = "PROMISING"
    elif any(r is not None and r <= 0.90 for r in ratios):
        conclusion = "PROMISING_TARGET_SPECIFIC"
    elif all(r is not None and r > 1.00 for r in ratios):
        conclusion = "UNPROMISING"
    else:
        conclusion = "UNCLEAR"
    result.update({"status": "COMPLETED", "conclusion": conclusion})
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
