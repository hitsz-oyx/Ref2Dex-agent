#!/usr/bin/env python3
"""Small matched pressure/contact prediction probe on prepared EgoTouch packs."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


HISTORY = 4
HORIZON = 8
TAU = 0.05


def _hand_local_history(hand: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wrist-relative hand shape scaled by the median of this past window only."""
    out = np.zeros_like(hand, dtype=np.float32)
    out_valid = valid.astype(np.float32, copy=True)
    finite = valid & np.isfinite(hand).all(axis=(1, 2))
    if not finite.any():
        return out, np.zeros_like(out_valid)
    centered = hand - hand[:, :1]
    scales = np.linalg.norm(centered[:, 9], axis=-1)
    scales = scales[finite]
    scales = scales[np.isfinite(scales) & (scales > 1e-6)]
    if not len(scales):
        return out, np.zeros_like(out_valid)
    scale = float(np.median(scales))
    out[finite] = centered[finite] / scale
    out_valid[~finite] = 0.0
    return out, out_valid


class PressureWindows(Dataset):
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
            arrays = dict(np.load(path, allow_pickle=False))
            rid = len(self.records)
            self.records.append((record, arrays))
            for start in range(len(arrays["pressure"]) - HISTORY - HORIZON + 1):
                for side in range(2):
                    # The preparation script has already validated measured cells.
                    if arrays["pressure_valid"][start:start + HISTORY + HORIZON, side].any(
                        axis=(1, 2)
                    ).all():
                        self.index.append((rid, start, side))
        if not self.index:
            raise ValueError(f"empty {split} pressure split")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        rid, start, side = self.index[index]
        record, a = self.records[rid]
        raw = a["raw_sensor"][start:start + HISTORY, side * 256:(side + 1) * 256] / 255.0
        hand, hand_valid = _hand_local_history(
            a["hand"][start:start + HISTORY, side],
            a["hand_valid"][start:start + HISTORY, side],
        )
        current = a["pressure"][start + HISTORY - 1, side]
        target = a["pressure"][start + HISTORY:start + HISTORY + HORIZON, side]
        valid = a["pressure_valid"][start + HISTORY:start + HISTORY + HORIZON, side]
        contact = a["contact_proxy"][start + HISTORY:start + HISTORY + HORIZON, side]
        # Missing grid cells are zeroed only after preserving valid in the sample.
        target = np.where(valid, target, 0.0).astype(np.float32)
        return {
            "raw": torch.from_numpy(raw.astype(np.float32)),
            "hand": torch.from_numpy(hand.astype(np.float32)),
            "hand_valid": torch.from_numpy(hand_valid.astype(np.float32)),
            "side": torch.tensor(side, dtype=torch.long),
            "current": torch.from_numpy(current.reshape(-1).astype(np.float32)),
            "target": torch.from_numpy(target.reshape(HORIZON, -1)),
            "valid": torch.from_numpy(valid.reshape(HORIZON, -1)),
            "contact": torch.from_numpy(contact.reshape(HORIZON, -1).astype(np.float32)),
            "record": record["record"],
        }


class PressureHead(nn.Module):
    def __init__(self, use_hand: bool):
        super().__init__()
        self.use_hand = bool(use_hand)
        in_dim = HISTORY * 256 + 2
        if self.use_hand:
            in_dim += HISTORY * (21 * 3 + 1)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU()
        )
        self.pressure = nn.Linear(256, HORIZON * 21 * 21)
        self.contact = nn.Linear(256, HORIZON * 21 * 21)
        # Start close to a low-pressure prediction while retaining a learnable head.
        nn.init.constant_(self.pressure.bias, -3.0)
        nn.init.constant_(self.contact.bias, -2.0)

    def forward(self, batch):
        parts = [batch["raw"].flatten(1), torch.nn.functional.one_hot(batch["side"], 2).float()]
        if self.use_hand:
            parts.extend([batch["hand"].flatten(1), batch["hand_valid"]])
        x = self.encoder(torch.cat(parts, dim=1))
        return {
            "pressure": torch.sigmoid(self.pressure(x)).reshape(-1, HORIZON, 441),
            "contact": self.contact(x).reshape(-1, HORIZON, 441),
        }


def masked_loss(pred, batch):
    valid = batch["valid"]
    if not bool(valid.any()):
        raise ValueError("batch has no measured target cells")
    pressure = torch.nn.functional.smooth_l1_loss(
        pred["pressure"], batch["target"], reduction="none"
    )
    pressure = pressure[valid].mean()
    contact = torch.nn.functional.binary_cross_entropy_with_logits(
        pred["contact"][valid], batch["contact"][valid]
    )
    return pressure + 0.2 * contact, pressure.detach(), contact.detach()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    by_record = {}
    for batch in loader:
        record_names = batch.pop("record")
        tensor_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(tensor_batch)
        valid = tensor_batch["valid"]
        target = tensor_batch["target"]
        current = tensor_batch["current"][:, None].expand_as(target)
        pred = out["pressure"]
        pred_contact = (torch.sigmoid(out["contact"]) >= 0.5)
        true_contact = tensor_batch["contact"] >= 0.5
        for i, name in enumerate(record_names):
            mask = valid[i]
            key = str(name)
            row = by_record.setdefault(key, {"pred_abs": 0.0, "persist_abs": 0.0,
                                              "cells": 0, "pred_pos": 0, "true_pos": 0,
                                              "pred_tp": 0})
            row["pred_abs"] += float(torch.abs(pred[i][mask] - target[i][mask]).sum())
            row["persist_abs"] += float(torch.abs(current[i][mask] - target[i][mask]).sum())
            row["cells"] += int(mask.sum())
            row["pred_pos"] += int(pred_contact[i][mask].sum())
            row["true_pos"] += int(true_contact[i][mask].sum())
            row["pred_tp"] += int((pred_contact[i][mask] & true_contact[i][mask]).sum())
    rows = {}
    for name, row in by_record.items():
        precision = row["pred_tp"] / max(row["pred_pos"], 1)
        recall = row["pred_tp"] / max(row["true_pos"], 1)
        rows[name] = {
            "pressure_mae": row["pred_abs"] / max(row["cells"], 1),
            "persistence_mae": row["persist_abs"] / max(row["cells"], 1),
            "contact_f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "cells": row["cells"],
        }
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=229)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"output exists: {args.output}")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fit = PressureWindows(args.pack, "fit")
    held = PressureWindows(args.pack, "held_task")
    loader = DataLoader(fit, batch_size=args.batch_size, shuffle=True, num_workers=0)
    held_loader = DataLoader(held, batch_size=args.batch_size, shuffle=False, num_workers=0)
    result = {
        "schema": "ref2dex.egotouch-pressure-auxiliary-result.v1",
        "status": "RUNNING",
        "run_id": "egotouch-pressure-20261011-r1",
        "seed": args.seed,
        "device": str(device),
        "steps": args.steps,
        "fit_windows": len(fit),
        "held_windows": len(held),
        "arms": {},
    }
    for arm, use_hand in (("raw", False), ("raw_hand", True)):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model = PressureHead(use_hand).to(device)
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
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss, pressure_loss, contact_loss = masked_loss(out, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{arm} step {step} nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step == 1 or step % 100 == 0 or step == args.steps:
                print(json.dumps({"arm": arm, "step": step, "loss": losses[-1],
                                  "pressure": float(pressure_loss), "contact": float(contact_loss),
                                  "elapsed_s": time.monotonic() - started}), flush=True)
        rows = evaluate(model, held_loader, device)
        torch.save({"model": model.state_dict(), "use_hand": use_hand,
                    "seed": args.seed, "steps": args.steps}, args.output / f"{arm}.pt")
        result["arms"][arm] = {
            "use_hand": use_hand,
            "final_loss": losses[-1],
            "mean_last50_loss": float(np.mean(losses[-50:])),
            "held": rows,
            "elapsed_s": time.monotonic() - started,
        }
    result["status"] = "COMPLETED"
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
