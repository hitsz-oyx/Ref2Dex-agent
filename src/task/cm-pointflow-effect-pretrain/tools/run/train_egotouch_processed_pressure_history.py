#!/usr/bin/env python3
"""Probe causal processed-pressure history against a persistence baseline.

The released EgoTouch grid is record-normalized.  For this bounded probe each
window is re-scaled by the maximum measured value in its *past* H frames, and
the same scale is applied to the future target and persistence control.  This
keeps the comparison invariant to the recording-wide normalization factor and
does not expose a future maximum to the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


HISTORY = 4
HORIZON = 8
GRID = 21 * 21
TAU = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hand_local_history(hand: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return wrist-relative hand shape with a scale from the past window only."""
    out = np.zeros_like(hand, dtype=np.float32)
    valid = valid.astype(bool, copy=False)
    finite = valid & np.isfinite(hand).all(axis=(1, 2))
    if not finite.any():
        return out, np.zeros(HISTORY, dtype=np.float32)
    centered = hand - hand[:, :1]
    scales = np.linalg.norm(centered[:, 9], axis=-1)
    scale_values = scales[finite]
    scale_values = scale_values[np.isfinite(scale_values) & (scale_values > 1e-6)]
    if not len(scale_values):
        return out, np.zeros(HISTORY, dtype=np.float32)
    scale = float(np.median(scale_values))
    out[finite] = centered[finite] / scale
    return out, finite.astype(np.float32)


class PressureHistoryWindows(Dataset):
    def __init__(self, root: Path, split: str):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "QUALIFIED_PRESSURE_PROBE_ONLY":
            raise ValueError("pressure target pack is not qualified")
        self.records = []
        self.index = []
        for record in manifest["records"]:
            if record["split"] != split:
                continue
            path = self.root / record["file"]
            with np.load(path, allow_pickle=False) as arrays:
                values = {key: arrays[key].copy() for key in (
                    "pressure", "pressure_valid", "hand", "hand_valid", "timestamps")}
            rid = len(self.records)
            self.records.append((record, values))
            for start in range(len(values["pressure"]) - HISTORY - HORIZON + 1):
                for side in range(2):
                    valid = values["pressure_valid"][start:start + HISTORY + HORIZON, side]
                    if valid.any(axis=(1, 2)).all():
                        self.index.append((rid, start, side))
        if not self.index:
            raise ValueError(f"empty {split} pressure split")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        rid, start, side = self.index[index]
        record, a = self.records[rid]
        past = a["pressure"][start:start + HISTORY, side].reshape(HISTORY, GRID)
        past_valid = a["pressure_valid"][start:start + HISTORY, side].reshape(HISTORY, GRID)
        future = a["pressure"][start + HISTORY:start + HISTORY + HORIZON, side].reshape(
            HORIZON, GRID)
        future_valid = a["pressure_valid"][start + HISTORY:start + HISTORY + HORIZON, side].reshape(
            HORIZON, GRID)
        measured_past = past[past_valid]
        if measured_past.size == 0 or not np.isfinite(measured_past).all():
            raise ValueError("past pressure has no finite measured cells")
        # The lower bound is the fixed activation threshold, and is computed
        # from past data only.  It prevents a zero-contact window from creating
        # an unbounded relative scale.
        scale = max(float(measured_past.max()), TAU)
        past_local = np.where(past_valid, past / scale, 0.0).astype(np.float32)
        future_local = np.where(future_valid, future / scale, 0.0).astype(np.float32)
        contact = (future_valid & (future_local >= TAU)).astype(np.float32)
        hand, hand_valid = hand_local_history(
            a["hand"][start:start + HISTORY, side],
            a["hand_valid"][start:start + HISTORY, side],
        )
        return {
            "pressure_history": torch.from_numpy(past_local),
            "pressure_history_valid": torch.from_numpy(past_valid.astype(np.float32)),
            "hand": torch.from_numpy(hand),
            "hand_valid": torch.from_numpy(hand_valid),
            "side": torch.tensor(side, dtype=torch.long),
            "current": torch.from_numpy(past_local[-1]),
            "target": torch.from_numpy(future_local),
            "valid": torch.from_numpy(future_valid),
            "contact": torch.from_numpy(contact),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "record": record["record"],
        }


class ProcessedPressureHead(nn.Module):
    def __init__(self, use_hand: bool):
        super().__init__()
        self.use_hand = bool(use_hand)
        in_dim = HISTORY * GRID * 2 + 2  # values + validity + side id
        if self.use_hand:
            in_dim += HISTORY * (21 * 3 + 1)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU()
        )
        self.pressure = nn.Linear(256, HORIZON * GRID)
        self.contact = nn.Linear(256, HORIZON * GRID)
        nn.init.constant_(self.pressure.bias, -2.0)
        nn.init.constant_(self.contact.bias, -2.0)

    def forward(self, batch):
        parts = [
            batch["pressure_history"].flatten(1),
            batch["pressure_history_valid"].flatten(1),
            F.one_hot(batch["side"], 2).float(),
        ]
        if self.use_hand:
            parts.extend([batch["hand"].flatten(1), batch["hand_valid"]])
        hidden = self.encoder(torch.cat(parts, dim=1))
        return {
            # Relative future pressure can exceed one when it rises above the
            # historical maximum, so use a nonnegative softplus rather than a
            # sigmoid capped at the historical scale.
            "pressure": F.softplus(self.pressure(hidden)).reshape(-1, HORIZON, GRID),
            "contact": self.contact(hidden).reshape(-1, HORIZON, GRID),
        }


def masked_loss(pred, batch):
    valid = batch["valid"]
    if not bool(valid.any()):
        raise ValueError("batch has no measured target cells")
    pressure = F.smooth_l1_loss(pred["pressure"], batch["target"], reduction="none")[valid].mean()
    contact = F.binary_cross_entropy_with_logits(
        pred["contact"][valid], batch["contact"][valid]
    )
    return pressure + 0.2 * contact, pressure.detach(), contact.detach()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    by_record = {}
    for batch in loader:
        names = batch.pop("record")
        device_batch = {key: value.to(device) if torch.is_tensor(value) else value
                        for key, value in batch.items()}
        pred = model(device_batch)
        valid = device_batch["valid"]
        target = device_batch["target"]
        current = device_batch["current"][:, None].expand_as(target)
        predicted_contact = torch.sigmoid(pred["contact"]) >= 0.5
        true_contact = device_batch["contact"] >= 0.5
        for i, name in enumerate(names):
            mask = valid[i]
            row = by_record.setdefault(str(name), {
                "pred_abs": 0.0, "persist_abs": 0.0, "pred_abs_original": 0.0,
                "persist_abs_original": 0.0, "cells": 0, "pred_pos": 0,
                "true_pos": 0, "pred_tp": 0,
            })
            row["pred_abs"] += float(torch.abs(pred["pressure"][i][mask] - target[i][mask]).sum())
            row["persist_abs"] += float(torch.abs(current[i][mask] - target[i][mask]).sum())
            scale = float(device_batch["scale"][i])
            row["pred_abs_original"] += float(
                torch.abs(pred["pressure"][i][mask] - target[i][mask]).sum()
            ) * scale
            row["persist_abs_original"] += float(
                torch.abs(current[i][mask] - target[i][mask]).sum()
            ) * scale
            row["cells"] += int(mask.sum())
            row["pred_pos"] += int(predicted_contact[i][mask].sum())
            row["true_pos"] += int(true_contact[i][mask].sum())
            row["pred_tp"] += int((predicted_contact[i][mask] & true_contact[i][mask]).sum())
    result = {}
    for name, row in by_record.items():
        precision = row["pred_tp"] / max(row["pred_pos"], 1)
        recall = row["pred_tp"] / max(row["true_pos"], 1)
        result[name] = {
            "pressure_mae_relative": row["pred_abs"] / max(row["cells"], 1),
            "persistence_mae_relative": row["persist_abs"] / max(row["cells"], 1),
            "pressure_mae_original_scale": row["pred_abs_original"] / max(row["cells"], 1),
            "persistence_mae_original_scale": row["persist_abs_original"] / max(row["cells"], 1),
            "contact_f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "cells": row["cells"],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=229)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--run-id", default=None)
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
    fit = PressureHistoryWindows(args.pack, "fit")
    held = PressureHistoryWindows(args.pack, "held_task")
    loader = DataLoader(fit, batch_size=args.batch_size, shuffle=True, num_workers=0)
    held_loader = DataLoader(held, batch_size=args.batch_size, shuffle=False, num_workers=0)
    result = {
        "schema": "ref2dex.egotouch-processed-pressure-history-result.v1",
        "status": "RUNNING",
        "run_id": args.run_id or args.output.name,
        "seed": args.seed,
        "device": str(device),
        "steps": args.steps,
        "fit_windows": len(fit),
        "held_windows": len(held),
        "protocol": {
            "history": HISTORY,
            "horizon": HORIZON,
            "past_only_scale_floor": TAU,
            "future_global_max_used_as_input": False,
            "target": "processed_pressure_relative_to_past_window_scale",
            "contact": "relative_pressure_ge_tau",
        },
        "pack_manifest_sha256": sha256(args.pack / "manifest.json"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "arms": {},
    }
    for arm, use_hand in (("pressure", False), ("pressure_hand", True)):
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = ProcessedPressureHead(use_hand).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        losses = []
        started = time.monotonic()
        iterator = iter(loader)
        for step in range(1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {key: value.to(device) if torch.is_tensor(value) else value
                     for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss, pressure_loss, contact_loss = masked_loss(pred, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{arm} step {step} nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step == 1 or step % 100 == 0 or step == args.steps:
                print(json.dumps({
                    "arm": arm, "step": step, "loss": losses[-1],
                    "pressure": float(pressure_loss), "contact": float(contact_loss),
                    "elapsed_s": time.monotonic() - started,
                }), flush=True)
        held_result = evaluate(model, held_loader, device)
        torch.save({"model": model.state_dict(), "use_hand": use_hand,
                    "seed": args.seed, "steps": args.steps}, args.output / f"{arm}.pt")
        result["arms"][arm] = {
            "use_hand": use_hand,
            "final_loss": losses[-1],
            "mean_last50_loss": float(np.mean(losses[-50:])),
            "held": held_result,
            "elapsed_s": time.monotonic() - started,
        }
    result["status"] = "COMPLETED"
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
