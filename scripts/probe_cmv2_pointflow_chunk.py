#!/usr/bin/env python3
"""Matched chunk extension for the Cmv2 point-flow feasibility probe.

The chunk is a diagnostic of temporal aggregation, not a deployment contract:
the read-only cache supplies cumulative MANO hand flow for each future horizon.
No future object geometry or object-flow label is passed to the model.
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
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task.ObjectInteractionCmv2.grab import GrabManoTransitions, _arrays
from src.task.ObjectInteractionCmv2.model import (
    ObjectInteractionCmv2V13Model,
    _axis_angle_matrix_stable,
    object_interaction_v13_loss,
)


class Cmv2ChunkDataset(Dataset):
    def __init__(self, transitions, chunk_length: int, chunks_per_sequence: int, seed: int):
        self.chunk_length = chunk_length
        self.point_indices = transitions.point_indices
        rng = np.random.default_rng(seed)
        self.records = []
        self.sequence_ids = []
        for sequence_index, (sequence_id, path, valid) in enumerate(transitions.sequences):
            starts = [int(value) for value in valid
                      if all(int(value + offset) in set(valid.tolist()) for offset in range(chunk_length))]
            if not starts:
                continue
            take = min(len(starts), chunks_per_sequence)
            chosen = sorted(rng.choice(len(starts), size=take, replace=False).tolist())
            self.sequence_ids.append(sequence_id)
            self.records.extend((sequence_id, path, starts[index]) for index in chosen)
        if not self.records:
            raise ValueError("No contiguous point-flow chunks available")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        sequence_id, path, t = self.records[index]
        arrays = _arrays(path)
        selected = self.point_indices
        pose0 = np.asarray(arrays["obj_pose_world"][t], dtype=np.float32)
        rotation0, translation0 = pose0[:3, :3], pose0[:3, 3]

        def local(world):
            return (np.asarray(world, dtype=np.float32) - translation0) @ rotation0

        obj_points = local(arrays["obj_points_pool_world"][t, selected])
        obj_normals = np.asarray(arrays["obj_normals_pool_world"][t, selected], dtype=np.float32) @ rotation0
        hand_points = local(arrays["knn_hand_points_world"][t])
        hand_normals = np.asarray(arrays["knn_hand_normals_world"][t], dtype=np.float32) @ rotation0
        hand_flow, obj_flow, delta_translation, delta_rotation, delta_time = [], [], [], [], []
        for offset in range(self.chunk_length):
            frame = t + offset + 1
            hand_flow.append(local(arrays["knn_hand_points_world"][frame]) - local(arrays["knn_hand_points_world"][t]))
            obj_flow.append(local(arrays["obj_points_pool_world"][frame, selected]) - obj_points)
            pose = np.asarray(arrays["obj_pose_world"][frame], dtype=np.float32)
            delta_rotation.append(rotation0.T @ pose[:3, :3])
            delta_translation.append((pose[:3, 3] - translation0) @ rotation0)
            delta_time.append(float(np.float64(arrays["frame_time"][frame]) - np.float64(arrays["frame_time"][t])))
        values = {
            "obj_points": obj_points,
            "obj_normals": obj_normals,
            "hand_points": hand_points,
            "hand_normals": hand_normals,
            "hand_flow": np.asarray(hand_flow, dtype=np.float32),
            "obj_flow_gt": np.asarray(obj_flow, dtype=np.float32),
            "delta_translation_gt": np.asarray(delta_translation, dtype=np.float32),
            "delta_rotation_gt": np.asarray(delta_rotation, dtype=np.float32),
            "delta_time_s": np.asarray(delta_time, dtype=np.float32),
            "hand_valid_mask": np.ones((4096,), dtype=bool),
        }
        if not all(np.isfinite(value).all() for value in values.values()):
            raise ValueError(f"Nonfinite point-flow chunk: {sequence_id} frame {t}")
        return {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in values.items()}


def _collate(samples):
    return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in samples[0]}


class Cmv2ChunkModel(nn.Module):
    def __init__(self, config, chunk_length: int):
        super().__init__()
        self.chunk_length = chunk_length
        self.base = ObjectInteractionCmv2V13Model(config)
        self.temporal = nn.GRU(config.hidden_width, config.hidden_width, batch_first=True)
        self.root_head = nn.Linear(config.hidden_width, 6)

    def forward(self, batch):
        bsz, steps = batch["hand_flow"].shape[:2]
        repeat = lambda value: value[:, None].expand(-1, steps, *value.shape[1:]).reshape(bsz * steps, *value.shape[1:])
        flat = {
            "obj_points": repeat(batch["obj_points"]),
            "obj_normals": repeat(batch["obj_normals"]),
            "hand_points": repeat(batch["hand_points"]),
            "hand_normals": repeat(batch["hand_normals"]),
            "hand_flow": batch["hand_flow"].reshape(bsz * steps, *batch["hand_flow"].shape[2:]),
            "hand_valid_mask": repeat(batch["hand_valid_mask"]),
            "delta_time_s": batch["delta_time_s"].reshape(-1),
        }
        base = self.base(flat)
        spatial = base["fused_feature"].reshape(bsz, steps, -1)
        temporal, _ = self.temporal(spatial)
        delta_xi = self.root_head(temporal)
        rotation = _axis_angle_matrix_stable(delta_xi[..., 3:].reshape(-1, 3)).reshape(bsz, steps, 3, 3)
        points = batch["obj_points"][:, None].expand(-1, steps, -1, -1)
        flow = torch.matmul(points, rotation.transpose(-1, -2)) + delta_xi[..., None, :3] - points
        return {"delta_xi_root": delta_xi, "obj_flow_pred": flow, "obj_flow_structured": flow}


def _loss(output, batch, scale_m=0.02):
    bsz, steps = batch["hand_flow"].shape[:2]
    flat_output = {key: value.reshape(bsz * steps, *value.shape[2:]) for key, value in output.items()}
    flat_batch = {}
    for key in ("obj_points", "obj_flow_gt", "delta_translation_gt", "delta_rotation_gt"):
        value = batch[key]
        if key == "obj_points":
            value = value[:, None].expand(-1, steps, -1, -1)
        flat_batch[key] = value.reshape(bsz * steps, *value.shape[2:])
    return object_interaction_v13_loss(flat_output, flat_batch, scale_m)


def _metrics(model, loader, device):
    model.eval()
    flow, translation, rotation = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(batch)
            flow.append(torch.linalg.vector_norm(output["obj_flow_pred"] - batch["obj_flow_gt"], dim=-1).mean((0, 2)).cpu())
            translation.append(torch.linalg.vector_norm(output["delta_xi_root"][..., :3] - batch["delta_translation_gt"], dim=-1).mean(0).cpu())
            relative = batch["delta_rotation_gt"].transpose(-1, -2) @ _axis_angle_matrix_stable(output["delta_xi_root"][..., 3:].reshape(-1, 3)).reshape_as(batch["delta_rotation_gt"])
            cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
            rotation.append(torch.rad2deg(torch.acos(cosine)).mean(0).cpu())
    return {"flow_epe_mm_by_horizon": (torch.stack(flow).mean(0) * 1000.0).tolist(),
            "root_translation_mae_mm_by_horizon": (torch.stack(translation).mean(0) * 1000.0).tolist(),
            "root_rotation_mae_deg_by_horizon": torch.stack(rotation).mean(0).tolist()}


def _single_baseline(base, loader, device):
    base.eval()
    flow = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            bsz, steps = batch["hand_flow"].shape[:2]
            flat = {"obj_points": batch["obj_points"][:, None].expand(-1, steps, -1, -1).reshape(bsz * steps, -1, 3),
                    "obj_normals": batch["obj_normals"][:, None].expand(-1, steps, -1, -1).reshape(bsz * steps, -1, 3),
                    "hand_points": batch["hand_points"][:, None].expand(-1, steps, -1, -1).reshape(bsz * steps, -1, 3),
                    "hand_normals": batch["hand_normals"][:, None].expand(-1, steps, -1, -1).reshape(bsz * steps, -1, 3),
                    "hand_flow": batch["hand_flow"].reshape(bsz * steps, -1, 3),
                    "hand_valid_mask": batch["hand_valid_mask"][:, None].expand(-1, steps, -1).reshape(bsz * steps, -1),
                    "delta_time_s": batch["delta_time_s"].reshape(-1)}
            pred = base(flat)["obj_flow_pred"].reshape(bsz, steps, -1, 3)
            flow.append(torch.linalg.vector_norm(pred - batch["obj_flow_gt"], dim=-1).mean((0, 2)).cpu())
    return {"flow_epe_mm_by_horizon": (torch.stack(flow).mean(0) * 1000.0).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--single-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-length", type=int, default=4)
    parser.add_argument("--train-sequences", type=int, default=8)
    parser.add_argument("--val-sequences", type=int, default=2)
    parser.add_argument("--chunks-per-sequence", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.chunk_length < 2:
        raise ValueError("chunk length must be at least 2")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    train_transitions = GrabManoTransitions(args.index, args.manifest, "train", max_sequences=args.train_sequences,
                                            allow_completed_cache=True)
    val_transitions = GrabManoTransitions(args.index, args.manifest, "val", max_sequences=args.val_sequences,
                                          allow_completed_cache=True)
    train_data = Cmv2ChunkDataset(train_transitions, args.chunk_length, args.chunks_per_sequence, args.seed)
    val_data = Cmv2ChunkDataset(val_transitions, args.chunk_length, args.chunks_per_sequence, args.seed + 1)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0, collate_fn=_collate)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0, collate_fn=_collate)
    config = SimpleNamespace(hidden_width=args.hidden_width, num_tokens=args.num_tokens, use_residual=False,
                             interaction_mode="swept", feature_scale_m=0.02, knn_k=args.knn_k,
                             interaction_radius_m=0.02, frame_dt_s=1 / 30)
    base = ObjectInteractionCmv2V13Model(config)
    checkpoint = torch.load(args.single_checkpoint, map_location="cpu", weights_only=False)
    base.load_state_dict(checkpoint["model"], strict=True)
    baseline = _single_baseline(base.to(device), val_loader, device)
    model = Cmv2ChunkModel(config, args.chunk_length).to(device)
    model.base.load_state_dict(checkpoint["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    history = []
    for epoch in range(args.epochs):
        model.train(); running = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = _loss(model(batch), batch)["total"]
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            running += float(loss.detach())
        metrics = _metrics(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss_mean": running / max(len(train_loader), 1), **metrics})
        print(json.dumps(history[-1]), flush=True)
    learned = _metrics(model, val_loader, device)
    report = {
        "schema": "ref2dex.cmv2_pointflow_chunk_probe.v1",
        "run_id": "P-20261004-cmv2-pointflow-chunk-k%d" % args.chunk_length,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "device": str(device), "seed": args.seed,
        "data": {"train_sequences": train_data.sequence_ids, "val_sequences": val_data.sequence_ids,
                 "train_chunks": len(train_data), "val_chunks": len(val_data), "chunk_length": args.chunk_length},
        "architecture": {"hidden_width": args.hidden_width, "num_tokens": args.num_tokens, "knn_k": args.knn_k,
                          "temporal": "GRU over per-horizon spatial fused features", "residual": False},
        "single_step_baseline": baseline, "chunk_model": learned, "history": history,
        "decision": "PROMISING if mean chunk EPE beats matched single-step baseline by >= 5%; otherwise stop chunk route",
        "caveat": "Cumulative recorded MANO hand flow is a privileged diagnostic input; this is not yet an action-deployable Gate 2 model.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"schema": report["schema"], "config": vars(args), "model": model.state_dict()}, args.output.with_suffix(".pt"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
