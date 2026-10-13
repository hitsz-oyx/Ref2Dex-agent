"""Bounded fit of the reduced patch forecaster on qualified human clips.

Dev clips are evaluation only. The original packs stay training_allowed=false;
this process only reads them and writes a new output directory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pointwam_repro.forecaster import ReducedPatchForecaster
from pointwam_repro.loss import trajectory_loss
from pointwam_repro.packs import example_from_pack


def _load_split(manifest: Path, max_scene_points: int) -> dict[str, list[dict]]:
    document = json.loads(manifest.read_text())
    groups = {"fit_candidate": [], "dev_candidate": []}
    for clip in document["clips"]:
        if not clip.get("qualified"):
            continue
        partition = clip["partition"]
        if partition not in groups:
            raise ValueError(f"unexpected partition {partition}")
        with np.load(clip["pack"], allow_pickle=False) as pack:
            example = example_from_pack(pack, max_scene_points)
        example["episode_id"] = clip["episode_id"]
        groups[partition].append(example)
    if not groups["fit_candidate"] or not groups["dev_candidate"]:
        raise ValueError("both fit and dev qualified clips are required")
    return groups


def _torch_example(example: dict, device: torch.device) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in example.items():
        if key == "episode_id":
            continue
        array = torch.as_tensor(value)
        if array.dtype == torch.bool:
            out[key] = array.to(device)
        else:
            out[key] = array.to(device=device, dtype=torch.float32)
    return out


def _masked_epe_mm(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    valid = valid & torch.isfinite(prediction).all(dim=-1) & torch.isfinite(target).all(dim=-1)
    if not bool(valid.any()):
        return float("nan")
    error = torch.linalg.norm(prediction - target, dim=-1)
    value = (torch.where(valid, error, torch.zeros_like(error))).sum() / valid.sum()
    return float(value.detach().cpu() * 1000)


def _loss(model: ReducedPatchForecaster, example: dict[str, torch.Tensor], balance_static: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = model.forward_clip(
        example["scene_xyz"][0],
        example["scene_rgb"],
        example["scene_valid"][0],
        example["hand_xyz"][0],
        example["hand_valid"][0],
    )
    scene_pred = prediction["scene_displacement"].unsqueeze(0)
    hand_pred = prediction["hand_displacement"].reshape(1, 30, 20, 3)
    scene_target = example["scene_displacement"].unsqueeze(0)
    hand_target = example["hand_displacement"].reshape(1, 30, 20, 3)
    scene_valid = example["scene_target_valid"].unsqueeze(0)
    hand_valid = example["hand_target_valid"].reshape(1, 30, 20)
    scene_positions = example["scene_xyz"].unsqueeze(0)
    hand_positions = example["hand_xyz"].reshape(1, 31, 20, 3)
    scene_loss = trajectory_loss(scene_pred, scene_target, scene_valid, scene_positions)
    if bool(hand_valid.any()):
        hand_loss = trajectory_loss(hand_pred, hand_target, hand_valid, hand_positions)
    else:
        hand_loss = scene_loss.new_zeros(())
    zero = torch.zeros_like(scene_pred)
    if balance_static:
        static_scene = trajectory_loss(zero, scene_target, scene_valid, scene_positions).detach().clamp_min(1e-8)
        static_hand = trajectory_loss(
            torch.zeros_like(hand_pred), hand_target, hand_valid, hand_positions).detach().clamp_min(1e-8)
        total = scene_loss / static_scene + hand_loss / static_hand
    else:
        total = scene_loss + hand_loss
    metrics = {
        "scene_epe_mm": _masked_epe_mm(scene_pred, scene_target, scene_valid),
        "hand_epe_mm": _masked_epe_mm(hand_pred, hand_target, hand_valid),
        "static_scene_epe_mm": _masked_epe_mm(zero, scene_target, scene_valid),
        "static_hand_epe_mm": _masked_epe_mm(torch.zeros_like(hand_pred), hand_target, hand_valid),
    }
    return total, metrics


def _mean(rows: list[dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows if np.isfinite(row[key])]
    if not values:
        return float("nan")
    return float(np.mean(values))


def _evaluate(model: ReducedPatchForecaster, rows: list[dict], balance_static: bool) -> dict[str, float]:
    model.eval()
    collected = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for example in rows:
            _, metrics = _loss(model, _torch_example(example, device), balance_static)
            collected.append(metrics)
    model.train()
    return {key: _mean(collected, key) for key in collected[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-scene-points", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seconds", type=int, default=720)
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--balance-by-static", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this probe requires a visible CUDA device")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    groups = _load_split(args.manifest, args.max_scene_points)
    model = ReducedPatchForecaster().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    initial = {name: _evaluate(model, rows, args.balance_by_static) for name, rows in groups.items()}
    started = time.monotonic()
    step = 0
    first_loss = None
    recent_loss = None
    order = np.arange(len(groups["fit_candidate"]))
    rng = np.random.default_rng(args.seed)
    nonfinite = False
    while step < args.steps and time.monotonic() - started < args.seconds:
        rng.shuffle(order)
        for index in order:
            if step >= args.steps or time.monotonic() - started >= args.seconds:
                break
            batch = _torch_example(groups["fit_candidate"][int(index)], device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _loss(model, batch, args.balance_by_static)
            if not torch.isfinite(loss):
                nonfinite = True
                break
            loss.backward()
            if step == 0:
                scene_grad = model.scene_head.net[-1].weight.grad
                hand_grad = model.hand_head.net[-1].weight.grad
                if scene_grad is None or hand_grad is None or not torch.isfinite(scene_grad).all() or not torch.isfinite(hand_grad).all():
                    raise RuntimeError("scene or hand head received no finite gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            value = float(loss.detach().cpu())
            first_loss = value if first_loss is None else first_loss
            recent_loss = value
            step += 1
            if step == 1 or step % 20 == 0:
                elapsed = time.monotonic() - started
                remaining = args.steps - step
                print(json.dumps({
                    "step": step,
                    "loss": value,
                    "elapsed_s": round(elapsed, 1),
                    "eta_s": round(elapsed / step * remaining, 1),
                    "gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                }), flush=True)
        if nonfinite:
            break
    final = {name: _evaluate(model, rows, args.balance_by_static) for name, rows in groups.items()}
    initial_loss = initial["fit_candidate"]
    final_fit = final["fit_candidate"]
    final_dev = final["dev_candidate"]
    # Eval loss is not stored; compare the reported EPE pair through a second pass
    # only via the saved metrics. The fit gate uses scene and hand EPE jointly
    # against the untrained model on the same fit clips.
    fit_improved = (
        final_fit["scene_epe_mm"] <= 0.9 * initial_loss["scene_epe_mm"]
        and final_fit["hand_epe_mm"] <= 0.9 * initial_loss["hand_epe_mm"]
    )
    dev_ok = (
        final_dev["hand_epe_mm"] <= final_dev["static_hand_epe_mm"]
        and final_dev["scene_epe_mm"] <= 1.05 * final_dev["static_scene_epe_mm"]
    )
    if nonfinite or step < 40:
        status = "UNCLEAR"
    elif fit_improved and dev_ok:
        status = "PROMISING"
    elif fit_improved:
        status = "UNPROMISING"
    else:
        status = "UNCLEAR"
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": status,
        "steps": step,
        "nonfinite": nonfinite,
        "elapsed_s": time.monotonic() - started,
        "max_scene_points": args.max_scene_points,
        "seed": args.seed,
        "balance_by_static": args.balance_by_static,
        "fit_clips": len(groups["fit_candidate"]),
        "dev_clips": len(groups["dev_candidate"]),
        "first_train_loss": first_loss,
        "last_train_loss": recent_loss,
        "gpu_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
        "initial": initial,
        "final": final,
        "fit_epe_improved_10pct": fit_improved,
        "dev_not_worse_than_static": dev_ok,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "steps": step, "seed": args.seed}, args.output / "last.pt")
    print("RESULT " + json.dumps(report), flush=True)
    if nonfinite:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
