#!/usr/bin/env python3
"""Train and benchmark the compact one-step object-effect model."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.task.CmResidual.cmlite import (
    SCHEMA, CmLite, compact_features, local_translation_target,
)


TRANSITION_SCHEMA = "ref2dex.cmlite_transition.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="append", required=True)
    parser.add_argument("--val", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    return parser.parse_args()


def load_transition(path: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != TRANSITION_SCHEMA:
        raise ValueError(f"transition schema mismatch: {path}")
    required = ("q", "action", "object_state", "next_object_state",
                "hand_contact", "object_contact")
    result = {key: payload[key] for key in required}
    size = result["q"].shape[0]
    if any(value.shape[0] != size for value in result.values()):
        raise ValueError(f"inconsistent transition lengths: {path}")
    return result


def prepare(payload: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    features = compact_features(payload["q"].float(), payload["action"].float(),
                                payload["object_state"].float())
    target = local_translation_target(payload["object_state"].float(),
                                      payload["next_object_state"].float())
    contact = (payload["hand_contact"].bool() &
               payload["object_contact"].bool()).float().reshape(-1)
    finite = torch.isfinite(features).all(1) & torch.isfinite(target).all(1)
    return features[finite], target[finite], contact[finite]


def concatenate(items: list[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.cat([item[index] for item in items]) for index in range(3))


@torch.inference_mode()
def metrics(model: CmLite, data: tuple[torch.Tensor, ...], stats: dict[str, torch.Tensor],
            device: torch.device, batch_size: int) -> dict[str, float | int]:
    errors, targets, probabilities, labels = [], [], [], []
    for start in range(0, data[0].shape[0], batch_size):
        features, target, contact = [x[start:start + batch_size].to(device) for x in data]
        output = model((features - stats["feature_mean"]) / stats["feature_std"])
        prediction = (output["delta_local_normalized"] * stats["target_std"] +
                      stats["target_mean"])
        errors.append((prediction - target).norm(dim=-1).cpu())
        targets.append(target.norm(dim=-1).cpu())
        probabilities.append(output["contact_logit"].sigmoid().cpu())
        labels.append(contact.cpu())
    error, motion = torch.cat(errors), torch.cat(targets)
    probability, label = torch.cat(probabilities), torch.cat(labels).bool()
    predicted = probability >= 0.5
    tp = (predicted & label).sum().item()
    fp = (predicted & ~label).sum().item()
    fn = (~predicted & label).sum().item()
    moving = motion > 5e-4
    contact_mask = label
    def selected_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
        return float(value[mask].mean().item() * 1000) if mask.any() else float("nan")
    return {
        "samples": error.numel(),
        "moving_samples": int(moving.sum()),
        "contact_samples": int(label.sum()),
        "epe_mm": float(error.mean().item() * 1000),
        "zero_baseline_mm": float(motion.mean().item() * 1000),
        "moving_epe_mm": selected_mean(error, moving),
        "moving_zero_baseline_mm": selected_mean(motion, moving),
        "contact_epe_mm": selected_mean(error, contact_mask),
        "contact_precision": tp / max(tp + fp, 1),
        "contact_recall": tp / max(tp + fn, 1),
        "contact_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "contact_accuracy": float((predicted == label).float().mean()),
    }


def git_revision() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True,
                          stdout=subprocess.PIPE).stdout.strip()


def main() -> None:
    args = parse_args()
    if not 0 <= args.holdout_fraction < 0.5:
        raise ValueError("holdout fraction must be in [0, 0.5)")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    prepared_train = [prepare(load_transition(path)) for path in args.train]
    source_counts = dict(zip(args.train, [item[0].shape[0] for item in prepared_train]))
    generator = torch.Generator().manual_seed(args.seed)
    train_parts, holdout_parts = [], []
    for item in prepared_train:
        order = torch.randperm(item[0].shape[0], generator=generator)
        count = int(item[0].shape[0] * args.holdout_fraction)
        holdout_parts.append(tuple(value[order[:count]] for value in item))
        train_parts.append(tuple(value[order[count:]] for value in item))
    train = concatenate(train_parts)
    holdout = concatenate(holdout_parts) if args.holdout_fraction else None
    external = [(path, prepare(load_transition(path))) for path in args.val]

    feature_mean, feature_std = train[0].mean(0), train[0].std(0).clamp_min(1e-6)
    target_mean, target_std = train[1].mean(0), train[1].std(0).clamp_min(1e-5)
    stats = {"feature_mean": feature_mean.to(device), "feature_std": feature_std.to(device),
             "target_mean": target_mean.to(device), "target_std": target_std.to(device)}
    model = CmLite(args.width, args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    positives = train[2].sum().item()
    pos_weight = min(max((train[2].numel() - positives) / max(positives, 1), 1.0), 10.0)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "metrics.jsonl"
    best_score = float("inf")
    best_payload = None
    train_device = tuple(value.to(device) for value in train)
    started = time.time()
    with history_path.open("w") as history:
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(train[0].shape[0], device=device)
            total_loss = 0.0
            for start in range(0, order.numel(), args.batch_size):
                index = order[start:start + args.batch_size]
                features, target, contact = [value[index] for value in train_device]
                normalized_target = (target - stats["target_mean"]) / stats["target_std"]
                prediction = model((features - stats["feature_mean"]) / stats["feature_std"])
                important = (target.norm(dim=-1) > 5e-4).float()
                weight = 1 + 4 * contact + 4 * important
                translation = F.smooth_l1_loss(
                    prediction["delta_local_normalized"], normalized_target,
                    reduction="none").mean(1)
                translation = (translation * weight).sum() / weight.sum()
                contact_loss = F.binary_cross_entropy_with_logits(
                    prediction["contact_logit"], contact,
                    pos_weight=torch.tensor(pos_weight, device=device))
                loss = translation + 0.25 * contact_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * index.numel()
            model.eval()
            report = {"epoch": epoch, "train_loss": total_loss / order.numel()}
            if holdout is not None:
                report["holdout"] = metrics(model, holdout, stats, device, args.batch_size)
                score = report["holdout"]["moving_epe_mm"]
            else:
                score = total_loss / order.numel()
            report["external"] = {
                path: metrics(model, data, stats, device, args.batch_size)
                for path, data in external
            }
            history.write(json.dumps(report, allow_nan=True) + "\n")
            history.flush()
            print(json.dumps(report, allow_nan=True), flush=True)
            if score < best_score:
                best_score = score
                best_payload = {key: value.detach().cpu().clone()
                                for key, value in model.state_dict().items()}

    assert best_payload is not None
    model.load_state_dict(best_payload)
    final = {
        "holdout": metrics(model, holdout, stats, device, args.batch_size) if holdout else {},
        "external": {path: metrics(model, data, stats, device, args.batch_size)
                     for path, data in external},
    }
    checkpoint = output / "best.pt"
    torch.save({
        "schema": SCHEMA,
        "model_config": {"width": args.width, "blocks": args.blocks},
        "model": best_payload,
        **{key: value.cpu() for key, value in stats.items()},
        "train_sources": args.train,
        "validation_sources": args.val,
        "source_counts": source_counts,
        "seed": args.seed,
        "git_revision": git_revision(),
        "metrics": final,
    }, checkpoint)

    model.eval()
    sample = ((train_device[0][:64] - stats["feature_mean"]) / stats["feature_std"])
    if device.type == "cuda":
        for _ in range(100):
            model(sample)
        torch.cuda.synchronize(device)
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        for _ in range(1000):
            model(sample)
        end.record()
        torch.cuda.synchronize(device)
        latency_ms = begin.elapsed_time(end) / 1000
    else:
        begin_time = time.perf_counter()
        for _ in range(1000):
            model(sample)
        latency_ms = (time.perf_counter() - begin_time) * 1000 / 1000
    summary = {
        "status": "COMPLETED", "checkpoint": str(checkpoint),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "batch64_latency_ms": latency_ms,
        "elapsed_seconds": time.time() - started,
        "metrics": final,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print("CMLITE_SUMMARY " + json.dumps(summary, allow_nan=True))


if __name__ == "__main__":
    main()
