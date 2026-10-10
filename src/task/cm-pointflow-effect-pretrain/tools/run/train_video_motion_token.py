#!/usr/bin/env python3
"""Matched weak-video Probe for a DynamicWAM-style metric-3D history token."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))
from oakink_wm.video_points import (  # noqa: E402
    MOTION_FEATURES,
    VideoWindows,
    balanced_flow_loss,
    collate_video,
    flow_metrics,
)
from oakink_wm.video_pointworld import VideoPointWorldWM  # noqa: E402
from oakink_wm.pointworld import VENDOR  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    part.replace(path)


def observed(batch: dict) -> dict:
    return {key: batch[key] for key in (
        "xyz", "features", "point_valid", "motion_stats", "motion_valid"
    )}


def device_batch(samples):
    return {key: value.cuda() for key, value in collate_video(samples).items()}


def train_motion_stats(dataset: VideoWindows) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for index in range(len(dataset)):
        sample = dataset[index]
        values.append(sample["motion_stats"][sample["motion_valid"]])
    values = np.concatenate(values, axis=0).astype(np.float32)
    if values.shape[1] != MOTION_FEATURES or not np.isfinite(values).all():
        raise ValueError("motion training statistics are invalid")
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


@torch.no_grad()
def evaluate(model, dataset, amp, horizon, use_motion):
    model.eval()
    totals = {}
    clip_totals = {}
    for index in range(0, len(dataset), 2):
        batch = device_batch([dataset[j] for j in range(index, min(index + 2, len(dataset)))])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            prediction = model(observed(batch), use_motion=use_motion)
        controls = {
            "model": prediction,
            "static": torch.zeros_like(prediction),
            "cv": batch["velocity"][:, :, None] * batch["elapsed"][:, None, :, None],
        }
        for name, pred in controls.items():
            for key, (value, count) in flow_metrics(pred, batch, horizon).items():
                pair = totals.setdefault(name + "/" + key, [0.0, 0])
                pair[0] += value
                pair[1] += count
            for row in range(len(batch["xyz"])):
                scene = dataset.index[index + row][0]
                single = {key: value[row:row + 1] for key, value in batch.items()}
                for key, (value, count) in flow_metrics(
                    pred[row:row + 1], single, horizon
                ).items():
                    pair = clip_totals.setdefault(scene, {}).setdefault(
                        name + "/" + key, [0.0, 0]
                    )
                    pair[0] += value
                    pair[1] += count
    result = {
        key: {"epe_m": value / count if count else None,
              "supported_point_horizons": count}
        for key, (value, count) in totals.items()
    }
    result["_per_clip"] = {
        scene: {
            key: {"epe_m": value / count if count else None,
                  "supported_point_horizons": count}
            for key, (value, count) in metrics.items()
        }
        for scene, metrics in clip_totals.items()
    }
    model.train()
    return result


def run_arm(arm, initial_state, scene_mean, scene_std, motion_mean, motion_std,
            train, dev, args, rng):
    model = VideoPointWorldWM(
        scene_mean, scene_std, motion_mean=motion_mean, motion_std=motion_std,
        motion_token=True,
    ).cuda()
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    amp = torch.cuda.is_bf16_supported()
    use_motion = arm == "motion"
    started = time.monotonic()
    stop = [False]
    progress_path = args.output / f"{arm}-progress.jsonl"
    with progress_path.open("w", buffering=1, encoding="utf-8") as progress:
        initial = evaluate(model, dev, amp, args.horizon, use_motion)
        atomic_json(args.output / f"initial-{arm}.json", initial)
        for step in range(1, args.steps + 1):
            if stop[0] or time.monotonic() - started > args.max_seconds:
                step -= 1
                status = "STOPPED_BUDGET_OR_SIGNAL"
                break
            indices = rng.randint(len(train), size=2)
            batch = device_batch([train[int(index)] for index in indices])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                prediction = model(observed(batch), use_motion=use_motion)
            loss = balanced_flow_loss(prediction, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{arm} nonfinite loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"{arm} nonfinite gradient at step {step}")
            optimizer.step()
            if step == 1 or step % 10 == 0 or step == args.steps:
                elapsed = time.monotonic() - started
                record = {
                    "step": step,
                    "loss": float(loss),
                    "grad_norm": float(grad_norm),
                    "elapsed_seconds": elapsed,
                    "updates_per_second": step / elapsed,
                    "estimated_remaining_seconds": (args.steps - step) * elapsed / step,
                    "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
                }
                if step == 1 or step % 100 == 0 or step == args.steps:
                    record["nvml_gpu_rows"] = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                         "--format=csv,noheader,nounits"], text=True
                    ).strip().splitlines()
                progress.write(json.dumps(record) + "\n")
                print(json.dumps({"arm": arm, **record}), flush=True)
            if not args.smoke and step % 100 == 0:
                atomic_json(
                    args.output / f"development-{arm}-step-{step:04d}.json",
                    evaluate(model, dev, amp, args.horizon, use_motion),
                )
        else:
            status = "COMPLETED"
    final = evaluate(model, dev, amp, args.horizon, use_motion)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "arm": arm}, args.output / f"final-{arm}.pt")
    del model
    torch.cuda.empty_cache()
    return {
        "status": status,
        "steps": step,
        "elapsed_seconds": time.monotonic() - started,
        "metrics": final,
        "use_motion": use_motion,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--horizon", type=int, choices=(8, 24), default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"output exists: {args.output}")
    args.output.mkdir(parents=True)
    random.seed(226)
    np.random.seed(226)
    torch.manual_seed(226)
    torch.cuda.manual_seed_all(226)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol"].get("supervised_horizon", 24) != args.horizon:
        raise ValueError("training horizon differs from corpus qualification")
    for entry in manifest["sequences"]:
        if sha(args.data / entry["file"]) != entry["sha256"]:
            raise ValueError("video source pack hash drift")
    train_videos = {entry["video"] for entry in manifest["sequences"] if entry["split"] == "train"}
    dev_videos = {entry["video"] for entry in manifest["sequences"] if entry["split"] == "dev"}
    if train_videos & dev_videos:
        raise ValueError("source video leakage across partitions")
    train, dev = VideoWindows(args.data, "train"), VideoWindows(args.data, "dev")
    features = np.concatenate([train[i]["features"] for i in range(len(train))])
    scene_mean = features.mean(0)
    scene_std = np.maximum(features.std(0), 1e-3)
    scene_std[features.std(0) < 1e-6] = 1.0
    motion_mean, motion_std = train_motion_stats(train)
    template = VideoPointWorldWM(
        scene_mean, scene_std, motion_mean=motion_mean, motion_std=motion_std,
        motion_token=True,
    ).cuda()
    initial_state = {key: value.detach().cpu().clone()
                     for key, value in template.state_dict().items()}
    del template
    torch.cuda.empty_cache()
    identity = {
        "schema": "ref2dex.video-motion-token-training.v1",
        "seed": 226,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "data_manifest_sha256": sha(args.data / "manifest.json"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(),
        "steps": args.steps,
        "smoke": args.smoke,
        "supervised_horizon": args.horizon,
        "motion_features": MOTION_FEATURES,
        "motion_mean": motion_mean.tolist(),
        "motion_std": motion_std.tolist(),
        "normalization": {"scene_mean": scene_mean.tolist(), "scene_std": scene_std.tolist()},
        "train_windows": len(train),
        "dev_windows": len(dev),
        "implementation_sha256": {
            str(path.relative_to(TASK)): sha(path)
            for path in (
                Path(__file__).resolve(), TASK / "src/oakink_wm/video_points.py",
                TASK / "src/oakink_wm/video_pointworld.py", TASK / "src/oakink_wm/pointworld.py",
                TASK / "src/oakink_wm/model.py",
            )
        },
        "vendor_sources_sha256": {
            str(path.relative_to(VENDOR)): sha(path)
            for path in (VENDOR / "ptv3").rglob("*") if path.suffix in (".py", ".yaml")
        },
    }
    atomic_json(args.output / "input_manifest.json", identity)
    results = {}
    for arm in ("control", "motion"):
        # Reset the sample sequence for a matched comparison.  Both models start
        # from exactly the same full state, including the unused token module.
        rng = np.random.RandomState(226)
        results[arm] = run_arm(
            arm, initial_state, scene_mean, scene_std, motion_mean, motion_std,
            train, dev, args, rng,
        )
    control = results["control"]["metrics"]
    motion = results["motion"]["metrics"]
    qualified = [
        scene for scene, values in motion["_per_clip"].items()
        if values["model/object/h8"]["supported_point_horizons"] >= 16
        and scene in control["_per_clip"]
        and control["_per_clip"][scene]["model/object/h8"]["supported_point_horizons"] >= 16
    ]
    motion_macro = float(np.mean([
        motion["_per_clip"][scene]["model/object/h8"]["epe_m"] for scene in qualified
    ])) if qualified else None
    control_macro = float(np.mean([
        control["_per_clip"][scene]["model/object/h8"]["epe_m"] for scene in qualified
    ])) if qualified else None
    per_clip_ratio = {
        scene: motion["_per_clip"][scene]["model/object/h8"]["epe_m"] /
        max(control["_per_clip"][scene]["model/object/h8"]["epe_m"], 1e-12)
        for scene in qualified
    }
    passed = bool(
        qualified and motion_macro <= 0.9 * control_macro
        and all(ratio <= 1.2 for ratio in per_clip_ratio.values())
        and motion["model/background/h8"]["epe_m"] <=
        1.2 * motion["static/background/h8"]["epe_m"]
    )
    result = {
        "status": "COMPLETED" if all(v["status"] == "COMPLETED" for v in results.values()) else "INCOMPLETE",
        "conclusion": "PROMISING" if passed else ("UNCLEAR" if not qualified else "UNPROMISING"),
        "matched_gate": {
            "qualified_clips": qualified,
            "motion_object_h8_macro_epe_m": motion_macro,
            "control_object_h8_macro_epe_m": control_macro,
            "per_clip_motion_over_control": per_clip_ratio,
            "background_gate_passed": motion["model/background/h8"]["epe_m"] <=
            1.2 * motion["static/background/h8"]["epe_m"],
        },
        "arms": results,
    }
    atomic_json(args.output / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
