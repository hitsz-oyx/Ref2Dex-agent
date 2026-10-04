#!/usr/bin/env python3
"""Single-step Cmv2 point-flow feasibility probe on the read-only GRAB/MANO cache.

This is intentionally separate from the Gate 2 action/value dataset.  The cache
contains the spatial hand/object streams used by ObjectInteractionCmv2 and the
next-frame object flow label, so the probe asks only whether the original
point-stream architecture learns a useful one-step E-only signal.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task.ObjectInteractionCmv2.grab import GrabManoTransitions
from src.task.ObjectInteractionCmv2.model import (
    ObjectInteractionCmv2V13Model,
    _axis_angle_matrix_stable,
    object_interaction_v13_loss,
)


def _tensor_batch(samples):
    return {key: torch.stack([sample[key] for sample in samples], dim=0)
            for key in samples[0] if torch.is_tensor(samples[0][key])}


def _sequence_indices(dataset, transitions_per_sequence: int, seed: int):
    rng = np.random.default_rng(seed)
    chosen = []
    sequence_ids = []
    for sequence_index, (sequence_id, _path, valid) in enumerate(dataset.sequences):
        start, stop = dataset.starts[sequence_index], dataset.starts[sequence_index + 1]
        count = stop - start
        take = min(count, transitions_per_sequence)
        local = np.sort(rng.choice(count, size=take, replace=False))
        chosen.extend((start + local).tolist())
        sequence_ids.append(sequence_id)
    return chosen, sequence_ids


def _metrics(model, loader, device):
    model.eval()
    flow_errors, translation_errors, rotation_errors = [], [], []
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            output = model(batch)
            losses = object_interaction_v13_loss(output, batch)
            gt_flow = batch["obj_flow_gt"]
            flow_errors.append(torch.linalg.vector_norm(output["obj_flow_pred"] - gt_flow, dim=-1).mean().cpu())
            translation_errors.append(torch.linalg.vector_norm(
                output["delta_xi_root"][:, :3] - batch["delta_translation_gt"], dim=-1).mean().cpu())
            predicted_rotation = _axis_angle_matrix_stable(output["delta_xi_root"][:, 3:])
            relative = batch["delta_rotation_gt"].transpose(1, 2) @ predicted_rotation
            cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
            rotation_errors.append(torch.rad2deg(torch.acos(cosine)).mean().cpu())
            total_loss += float(losses["total"]) * gt_flow.shape[0]
            count += gt_flow.shape[0]
    return {
        "flow_epe_m": float(torch.stack(flow_errors).mean()),
        "flow_epe_mm": float(torch.stack(flow_errors).mean() * 1000.0),
        "root_translation_mae_m": float(torch.stack(translation_errors).mean()),
        "root_translation_mae_mm": float(torch.stack(translation_errors).mean() * 1000.0),
        "root_rotation_mae_deg": float(torch.stack(rotation_errors).mean()),
        "loss": total_loss / max(count, 1),
    }


def _zero_metrics(loader):
    errors, translation, rotation = [], [], []
    for batch in loader:
        errors.append(torch.linalg.vector_norm(batch["obj_flow_gt"], dim=-1).mean())
        translation.append(torch.linalg.vector_norm(batch["delta_translation_gt"], dim=-1).mean())
        cosine = ((batch["delta_rotation_gt"].diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
        rotation.append(torch.rad2deg(torch.acos(cosine)).mean())
    epe = torch.stack(errors).mean()
    trans = torch.stack(translation).mean()
    return {"flow_epe_m": float(epe), "flow_epe_mm": float(epe * 1000.0),
            "root_translation_mae_m": float(trans), "root_translation_mae_mm": float(trans * 1000.0),
            "root_rotation_mae_deg": float(torch.stack(rotation).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-sequences", type=int, default=8)
    parser.add_argument("--val-sequences", type=int, default=2)
    parser.add_argument("--transitions-per-sequence", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.train_sequences <= 0 or args.val_sequences <= 0 or args.transitions_per_sequence <= 0:
        raise ValueError("sequence and transition limits must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    train_data = GrabManoTransitions(args.index, args.manifest, "train", max_sequences=args.train_sequences,
                                     allow_completed_cache=True, direct_pose_gt=True)
    val_data = GrabManoTransitions(args.index, args.manifest, "val", max_sequences=args.val_sequences,
                                   allow_completed_cache=True, direct_pose_gt=True)
    train_indices, train_sequences = _sequence_indices(train_data, args.transitions_per_sequence, args.seed)
    val_indices, val_sequences = _sequence_indices(val_data, args.transitions_per_sequence, args.seed + 1)
    train_loader = DataLoader(Subset(train_data, train_indices), batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=_tensor_batch)
    val_loader = DataLoader(Subset(val_data, val_indices), batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=_tensor_batch)
    config = SimpleNamespace(hidden_width=args.hidden_width, num_tokens=args.num_tokens, use_residual=False,
                             interaction_mode="swept", feature_scale_m=0.02, knn_k=args.knn_k,
                             interaction_radius_m=0.02, frame_dt_s=1 / 30)
    model = ObjectInteractionCmv2V13Model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    history = []
    for epoch in range(args.epochs):
        model.train()
        running, count = 0.0, 0
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            output = model(batch)
            loss = object_interaction_v13_loss(output, batch)["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * batch["obj_points"].shape[0]
            count += batch["obj_points"].shape[0]
        metrics = _metrics(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": running / max(count, 1), **metrics})
        print(json.dumps(history[-1]), flush=True)
    zero = _zero_metrics(val_loader)
    learned = _metrics(model, val_loader, device)
    report = {
        "schema": "ref2dex.cmv2_pointflow_single_probe.v1",
        "run_id": "P-20261004-cmv2-pointflow-single",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "device": str(device), "seed": args.seed,
        "data": {"index": str(args.index.resolve()), "manifest": str(args.manifest.resolve()),
                 "train_sequences": train_sequences, "val_sequences": val_sequences,
                 "train_transitions": len(train_indices), "val_transitions": len(val_indices)},
        "architecture": {"hidden_width": args.hidden_width, "num_tokens": args.num_tokens,
                          "knn_k": args.knn_k, "interaction_mode": "swept",
                          "feature_scale_m": 0.02, "residual": False},
        "zero_flow": zero, "learned": learned,
        "relative_flow_epe_reduction": float(1.0 - learned["flow_epe_m"] / max(zero["flow_epe_m"], 1e-12)),
        "history": history,
        "decision": "PROMISING if relative_flow_epe_reduction >= 0.10; otherwise do not extend to chunk",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"schema": report["schema"], "config": vars(args), "model": model.state_dict()},
               args.output.with_suffix(".pt"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
