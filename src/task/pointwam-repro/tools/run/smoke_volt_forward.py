"""One paper-sized Volt-H step on a synthetic observation.

Frozen Mosaic3D and Recap-CLIP weights are not loaded. Scene and text
features are random tensors with the published widths, so a finite loss only
shows that the trainable forecaster fits. It is not a training result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from pointwam_repro.loss import standardized_joint_loss
from pointwam_repro.pointwam import PointWAMForecaster, paper_adamw
from pointwam_repro.stats import fit_horizon_stats


def _observation(points: int, span_m: float, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(11)
    scene_xyz = torch.rand(points, 3, generator=generator) * span_m
    scene_features = torch.randn(points, 768, generator=generator)
    scene_valid = torch.ones(points, dtype=torch.bool)
    hand_xyz = torch.rand(2, 10, 3, generator=generator) * span_m
    hand_valid = torch.ones(2, 10, dtype=torch.bool)
    text = torch.randn(32, 1024, generator=generator)
    displacement = torch.randn(30, points, 3, generator=generator) * 0.01
    hand_displacement = torch.randn(30, 2, 10, 3, generator=generator) * 0.02
    scene_positions = torch.zeros(31, points, 3)
    scene_positions[0] = scene_xyz
    scene_positions[1:] = scene_xyz + displacement.cumsum(dim=0)
    hand_positions = torch.zeros(31, 2, 10, 3)
    hand_positions[0] = hand_xyz
    hand_positions[1:] = hand_xyz + hand_displacement.cumsum(dim=0)
    rows = {
        "scene_xyz": scene_xyz,
        "scene_features": scene_features,
        "scene_valid": scene_valid,
        "hand_xyz": hand_xyz,
        "hand_valid": hand_valid,
        "text_features": text,
        "scene_displacement": displacement,
        "hand_displacement": hand_displacement,
        "scene_target_valid": torch.ones(30, points, dtype=torch.bool),
        "hand_target_valid": torch.ones(30, 2, 10, dtype=torch.bool),
        "scene_positions": scene_positions,
        "hand_positions": hand_positions,
    }
    return {key: value.to(device) for key, value in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=12000)
    parser.add_argument("--span-m", type=float, default=0.8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Volt-H smoke requires CUDA")
    device = torch.device("cuda:0")
    torch.manual_seed(11)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = PointWAMForecaster().to(device)
    example = _observation(args.points, args.span_m, device)
    stats = fit_horizon_stats([{
        "scene_displacement": example["scene_displacement"].cpu(),
        "scene_target_valid": example["scene_target_valid"].cpu(),
        "hand_displacement": example["hand_displacement"].cpu(),
        "hand_target_valid": example["hand_target_valid"].cpu(),
    }])
    optimizer = paper_adamw(model)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    step_started = time.perf_counter()
    report = {
        "model": "PointWAMForecaster",
        "encoders": "random features at Mosaic3D width 768 and Recap-CLIP token width 1024",
        "points": args.points,
        "span_m": args.span_m,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "precision": "bf16 autocast, fp32 parameters",
        "optimizer": "AdamW lr 1e-4 weight_decay 0.01 clip 5",
        "seed": 11,
    }
    try:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model.forward_observation(
                example["scene_xyz"], example["scene_features"], example["scene_valid"],
                example["hand_xyz"], example["hand_valid"], example["text_features"])
            loss = standardized_joint_loss(
                prediction["scene_displacement"], prediction["hand_displacement"],
                example["scene_displacement"], example["hand_displacement"],
                example["scene_target_valid"], example["hand_target_valid"],
                example["scene_positions"], example["hand_positions"], stats)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        torch.cuda.synchronize()
        report.update({
            "loss": float(loss.detach().cpu()),
            "loss_finite": bool(torch.isfinite(loss.detach())),
            "grad_norm": float(grad_norm.detach().cpu()),
            "grad_finite": bool(torch.isfinite(grad_norm.detach())),
            "status": "ok",
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
