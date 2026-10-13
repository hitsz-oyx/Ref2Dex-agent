"""One PointWAM forward with frozen Mosaic3D and Recap-CLIP on one clip.

The source pack stays training_allowed=false. This process only reads it and
writes a new output directory. A finite forecast shows the encoders are wired.
It is not a training result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pointwam_repro.encoders import load_mosaic, load_recap_text, mosaic_point_features, recap_token_features
from pointwam_repro.packs import example_from_pack
from pointwam_repro.pointwam import PointWAMForecaster


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-scene-points", type=int, default=12000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("frozen forward requires CUDA")
    device = torch.device("cuda:0")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with np.load(args.pack, allow_pickle=False) as pack:
        if bool(np.asarray(pack["training_allowed"])):
            raise SystemExit("refusing a pack with training_allowed true")
        language = str(np.asarray(pack["language"]))
        example = example_from_pack(pack, args.max_scene_points)
    mosaic = load_mosaic(device=device)
    text_model, tokenizer = load_recap_text(device=device)
    forecaster = PointWAMForecaster().to(device).eval()
    scene_xyz = torch.as_tensor(example["scene_xyz"][0], device=device)
    scene_rgb = torch.as_tensor(example["scene_rgb"], device=device)
    scene_valid = torch.as_tensor(example["scene_valid"][0], device=device)
    hand_xyz = torch.as_tensor(example["hand_xyz"][0], device=device)
    hand_valid = torch.as_tensor(example["hand_valid"][0], device=device)
    torch.cuda.reset_peak_memory_stats()
    features = mosaic_point_features(mosaic, scene_xyz, scene_rgb)
    tokens = recap_token_features(text_model, tokenizer, language, device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = forecaster.forward_observation(
            scene_xyz, features, scene_valid, hand_xyz, hand_valid, tokens)
    torch.cuda.synchronize()
    scene = prediction["scene_displacement"]
    hand = prediction["hand_displacement"]
    report = {
        "pack": str(args.pack),
        "language": language,
        "scene_points": int(scene_xyz.shape[0]),
        "text_tokens": int(tokens.shape[0]),
        "scene_feature_dim": int(features.shape[-1]),
        "text_feature_dim": int(tokens.shape[-1]),
        "scene_finite": bool(torch.isfinite(scene).all()),
        "hand_finite": bool(torch.isfinite(hand).all()),
        "scene_abs_mean": float(scene.float().abs().mean().cpu()),
        "hand_abs_mean": float(hand.float().abs().mean().cpu()),
        "peak_memory_mib": int(torch.cuda.max_memory_allocated() / (1024 ** 2)),
        "seconds": round(time.perf_counter() - started, 3),
        "encoders": "frozen Mosaic3D sc+ar+sc+++ma+st and Recap-CLIP ln_final tokens",
        "forecaster": "randomly initialized PointWAMForecaster",
        "status": "ok" if torch.isfinite(scene).all() and torch.isfinite(hand).all() else "failed",
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
