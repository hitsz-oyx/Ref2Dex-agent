"""One Table 4 optimizer step with frozen encoders on one stored clip.

The pack stays training_allowed=false. Horizon statistics are fit on this
clip alone, and the forecaster starts from a random initialization. A finite
loss shows the training step is wired. It is not a pre-training result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pointwam_repro.encoders import load_mosaic, load_recap_text, mosaic_point_features, recap_token_features
from pointwam_repro.loss import standardized_joint_loss
from pointwam_repro.packs import example_from_pack
from pointwam_repro.pointwam import PointWAMForecaster, paper_adamw
from pointwam_repro.stats import fit_horizon_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-scene-points", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("frozen step requires CUDA")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with np.load(args.pack, allow_pickle=False) as pack:
        if bool(np.asarray(pack["training_allowed"])):
            raise SystemExit("refusing a pack with training_allowed true")
        language = str(np.asarray(pack["language"]))
        example = example_from_pack(pack, args.max_scene_points)
    scene_xyz = torch.as_tensor(example["scene_xyz"][0], device=device)
    scene_rgb = torch.as_tensor(example["scene_rgb"], device=device)
    scene_valid = torch.as_tensor(example["scene_valid"][0], device=device)
    hand_xyz = torch.as_tensor(example["hand_xyz"][0], device=device)
    hand_valid = torch.as_tensor(example["hand_valid"][0], device=device)
    scene_target = torch.as_tensor(example["scene_displacement"], device=device)
    hand_target = torch.as_tensor(example["hand_displacement"], device=device)
    scene_target_valid = torch.as_tensor(example["scene_target_valid"], device=device)
    hand_target_valid = torch.as_tensor(example["hand_target_valid"], device=device)
    scene_positions = torch.as_tensor(example["scene_xyz"], device=device)
    hand_positions = torch.as_tensor(example["hand_xyz"], device=device)
    stats = fit_horizon_stats([{
        "scene_displacement": example["scene_displacement"],
        "scene_target_valid": example["scene_target_valid"],
        "hand_displacement": example["hand_displacement"],
        "hand_target_valid": example["hand_target_valid"],
    }])
    mosaic = load_mosaic(device=device)
    features = mosaic_point_features(mosaic, scene_xyz, scene_rgb).detach()
    del mosaic
    torch.cuda.empty_cache()
    text_model, tokenizer = load_recap_text(device=device)
    tokens = recap_token_features(text_model, tokenizer, language, device).detach()
    del text_model
    torch.cuda.empty_cache()
    forecaster = PointWAMForecaster().to(device)
    optimizer = paper_adamw(forecaster)
    forecaster.train()
    torch.cuda.reset_peak_memory_stats()
    step_started = time.perf_counter()
    report = {
        "pack": str(args.pack),
        "language": language,
        "scene_points": int(scene_xyz.shape[0]),
        "text_tokens": int(tokens.shape[0]),
        "scene_feature_dim": int(features.shape[-1]),
        "text_feature_dim": int(tokens.shape[-1]),
        "scene_features_finite": bool(torch.isfinite(features).all()),
        "text_features_finite": bool(torch.isfinite(tokens).all()),
        "parameters": sum(parameter.numel() for parameter in forecaster.parameters()),
        "precision": "bf16 autocast, fp32 parameters",
        "optimizer": "AdamW lr 1e-4 weight_decay 0.01 clip 5",
        "stats": "horizon statistics fit on this one clip",
        "encoders": "frozen Mosaic3D sc+ar+sc+++ma+st and Recap-CLIP ln_final tokens",
        "forecaster": "randomly initialized PointWAMForecaster",
        "seed": args.seed,
    }
    try:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = forecaster.forward_observation(
                scene_xyz, features, scene_valid, hand_xyz, hand_valid, tokens)
            loss = standardized_joint_loss(
                prediction["scene_displacement"], prediction["hand_displacement"],
                scene_target, hand_target, scene_target_valid, hand_target_valid,
                scene_positions, hand_positions, stats)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(forecaster.parameters(), 5.0)
        optimizer.step()
        torch.cuda.synchronize()
        report.update({
            "loss": float(loss.detach().cpu()),
            "loss_finite": bool(torch.isfinite(loss.detach())),
            "grad_norm": float(grad_norm.detach().cpu()),
            "grad_finite": bool(torch.isfinite(grad_norm.detach())),
            "status": "ok" if torch.isfinite(loss.detach()) and torch.isfinite(grad_norm.detach()) else "failed",
        })
    except RuntimeError as exc:
        report.update({"status": "failed", "error": str(exc).splitlines()[0]})
    report["peak_memory_mib"] = int(torch.cuda.max_memory_allocated() / (1024 ** 2))
    report["step_seconds"] = round(time.perf_counter() - step_started, 3)
    report["total_seconds"] = round(time.perf_counter() - started, 3)
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
