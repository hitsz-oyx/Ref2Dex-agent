#!/usr/bin/env python3
"""K=1 hand-agnostic object-surface interaction-token probe.

This is an independent follow-up to ``probe_cmv2_unified_ei.py``.  The old
13-D interaction vector pooled an entire object into one global statistic.  The
target here is one surface token with five invariant quantities:

``[contact_mass, hand_object_distance, normal_velocity, tangential_velocity,
contact_change]``.

The target is computed from unordered hand surface samples and object anchors
in the future object frame.  No hand link, finger, joint, or topology ID is
used.  The Cmv2 route loads the previous K=1 checkpoint and freezes its
spatial E encoder; only a new token head is fitted.  A state/action MLP+GRU is
the matched direct baseline.
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

from src.task.CmResidual.dexplore_cm_geometry import (  # noqa: E402
    dexplore_action_to_native_targets,
    dexplore_root_pose,
    native_joint_limits,
)
from src.task.CmResidual.surface_execution import area_hand_samples  # noqa: E402
from src.task.CmResidual.v118_planner import TorchInspireKinematics  # noqa: E402
from src.task.ObjectInteractionCmv2.model import ObjectInteractionCmv2V13Model  # noqa: E402

DT = 1.0 / 30.0
TOKEN_DIM = 5


def read_obj(path: Path):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] == "f":
            faces.append([int(x.split("/")[0]) - 1 for x in fields[1:4]])
    vertices = np.asarray(vertices, np.float32)
    faces = np.asarray(faces, np.int64)
    normals = np.zeros_like(vertices)
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for col in range(3):
        np.add.at(normals, faces[:, col], face_normals)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-8)
    return vertices, normals


def axis_angle_from_rotation(rotation):
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    skew = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        -1,
    )
    axis = skew / (2.0 * angle.sin().unsqueeze(-1).clamp_min(1e-6))
    small = angle < 1e-4
    return torch.where(small.unsqueeze(-1), 0.5 * skew, axis * angle.unsqueeze(-1))


def surface_token_label(object_points, object_normals, hand_now_world,
                        hand_future_world, object_now, object_future):
    """Return one future-object-frame surface token for each sample."""
    future_inv = object_future[:, :3, :3].transpose(-1, -2)
    hand_future = torch.einsum(
        "bij,bnj->bni", future_inv,
        hand_future_world - object_future[:, None, :3, 3],
    )
    hand_now = torch.einsum(
        "bij,bnj->bni", future_inv,
        hand_now_world - object_future[:, None, :3, 3],
    )
    future_distance = torch.cdist(object_points, hand_future).amin(-1)
    now_distance = torch.cdist(object_points, hand_now).amin(-1)
    future_contact = torch.sigmoid((0.02 - future_distance) / 0.005)
    now_contact = torch.sigmoid((0.02 - now_distance) / 0.005)
    weights = future_contact / future_contact.sum(-1, keepdim=True).clamp_min(1e-6)

    future_nearest = torch.cdist(object_points, hand_future).argmin(-1)
    now_nearest = torch.cdist(object_points, hand_now).argmin(-1)
    hf = hand_future.gather(1, future_nearest[..., None].expand(-1, -1, 3))
    h0 = hand_now.gather(1, now_nearest[..., None].expand(-1, -1, 3))
    relative_velocity = (hf - h0) / DT
    normals = object_normals / object_normals.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    normal_velocity = (relative_velocity * normals).sum(-1)
    tangential_velocity = (
        relative_velocity - normal_velocity[..., None] * normals
    ).norm(dim=-1)
    return torch.cat(
        (
            future_contact.mean(-1, keepdim=True),
            (weights * future_distance).sum(-1, keepdim=True),
            (weights * normal_velocity).sum(-1, keepdim=True),
            (weights * tangential_velocity).sum(-1, keepdim=True),
            (future_contact - now_contact).mean(-1, keepdim=True),
        ),
        -1,
    )


@torch.no_grad()
def materialize(rows, trace, object_points, object_normals, hand_surface,
                hand_normals, hand_links, fk, lower, upper, device,
                chunk_size=64):
    """Vectorized geometry construction with the corrected actor/link axes."""
    if len(rows) > chunk_size:
        pieces = [
            materialize(
                rows[start:start + chunk_size], trace, object_points,
                object_normals, hand_surface, hand_normals, hand_links, fk,
                lower, upper, device, chunk_size,
            )
            for start in range(0, len(rows), chunk_size)
        ]
        return {key: torch.cat([part[key] for part in pieces], 0) for key in pieces[0]}

    n = len(rows)
    object_points = object_points.to(device)
    object_normals = object_normals.to(device)
    hand_surface = hand_surface.to(device)
    hand_normals = hand_normals.to(device)
    hand_links = hand_links.to(device)
    times = torch.tensor([row[1] for row in rows], dtype=torch.long, device=device)
    envs = torch.tensor([row[2] for row in rows], dtype=torch.long, device=device)
    q_all = trace["q"].to(device)
    root_all = trace["root"].to(device)
    action_all = trace["action"].to(device)
    state_all = trace["state"].to(device)
    current_q = q_all[times, envs]
    current_roots = root_all[times, envs]
    current_obj = dexplore_root_pose(current_roots[:, 2])
    current_hand_root = current_roots[:, 0]
    actions = torch.stack([action_all[times + 1, envs]], 1)
    future_q = torch.stack([q_all[times + 1, envs]], 1)
    future_roots = torch.stack([root_all[times + 1, envs]], 1)

    def surface(links, roots):
        base = dexplore_root_pose(roots.reshape(-1, 13)).reshape(*roots.shape[:-1], 4, 4)
        world_links = torch.matmul(base.unsqueeze(-3), links.unsqueeze(-4))
        rotation = world_links[..., hand_links, :3, :3]
        translation = world_links[..., hand_links, :3, 3]
        points = torch.einsum("...pij,pj->...pi", rotation, hand_surface) + translation
        normals = torch.einsum("...pij,pj->...pi", rotation, hand_normals)
        return points, normals

    current_links = fk.forward(current_q[:, None])[:, 0]
    current_world, current_world_normals = surface(current_links, current_hand_root[:, None])
    current_world, current_world_normals = current_world[:, 0], current_world_normals[:, 0]
    object_rotation = current_obj[:, :3, :3]
    current_hand = torch.einsum(
        "bij,bpj->bpi", object_rotation.transpose(-1, -2),
        current_world - current_obj[:, None, :3, 3],
    )
    current_hand_normals = torch.einsum(
        "bij,bpj->bpi", object_rotation.transpose(-1, -2), current_world_normals,
    )

    nominal_q = current_q
    nominal_q = dexplore_action_to_native_targets(actions[:, 0], nominal_q, lower, upper)
    nominal_links = fk.forward(nominal_q[:, None])[:, 0]
    nominal_world, _ = surface(nominal_links, current_hand_root[:, None])
    nominal_world = nominal_world[:, 0]
    predicted_flow = torch.einsum(
        "bij,bpj->bpi", object_rotation.transpose(-1, -2), nominal_world - current_world,
    )[:, None]

    future_links = fk.forward(future_q[:, 0:1])[:, 0]
    actual_future, _ = surface(future_links, future_roots[:, 0, 0:1])
    actual_future = actual_future[:, 0]
    future_obj = dexplore_root_pose(future_roots[:, 0, 2])
    effect_translation = torch.einsum(
        "bij,bj->bi", object_rotation.transpose(-1, -2),
        future_obj[:, :3, 3] - current_obj[:, :3, 3],
    )
    effect_rotation = object_rotation.transpose(-1, -2) @ future_obj[:, :3, :3]
    effect = torch.cat((effect_translation, axis_angle_from_rotation(effect_rotation)), -1)[:, None]
    token = surface_token_label(
        object_points[None].expand(n, -1, -1), object_normals[None].expand(n, -1, -1),
        current_world, actual_future, current_obj, future_obj,
    )[:, None]
    return {
        "state": state_all[times, envs].float().cpu(),
        "action": actions.float().cpu(),
        "obj_points": object_points.float().expand(n, -1, -1).cpu(),
        "obj_normals": object_normals.float().expand(n, -1, -1).cpu(),
        "hand_points": current_hand.float().cpu(),
        "hand_normals": current_hand_normals.float().cpu(),
        "hand_flow": predicted_flow.float().cpu(),
        "effect": effect.float().cpu(),
        "token": token.float().cpu(),
    }


class CachedDataset(Dataset):
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return self.values["state"].shape[0]

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.values.items()}


class ActionTokenModel(nn.Module):
    def __init__(self, state_dim=55, hidden=128):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(state_dim + 18, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, TOKEN_DIM)

    def forward(self, state, action):
        x = self.input(torch.cat((state[:, None].expand(-1, action.shape[1], -1), action), -1))
        x, _ = self.gru(x)
        return self.head(x)


class FrozenCmv2TokenModel(nn.Module):
    def __init__(self, cfg, checkpoint):
        super().__init__()
        self.base = ObjectInteractionCmv2V13Model(cfg)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        base_state = {key[len("base."):]: value for key, value in state.items() if key.startswith("base.")}
        missing, unexpected = self.base.load_state_dict(base_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"base checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_width + 32, cfg.hidden_width), nn.SiLU(),
            nn.Linear(cfg.hidden_width, TOKEN_DIM),
        )

    def forward(self, batch):
        bsz, steps = batch["hand_flow"].shape[:2]
        repeat = lambda value: value[:, None].expand(-1, steps, *value.shape[1:]).reshape(bsz * steps, *value.shape[1:])
        flat = {
            "obj_points": repeat(batch["obj_points"]),
            "obj_normals": repeat(batch["obj_normals"]),
            "hand_points": repeat(batch["hand_points"]),
            "hand_normals": repeat(batch["hand_normals"]),
            "hand_flow": batch["hand_flow"].reshape(bsz * steps, -1, 3),
            "hand_valid_mask": torch.ones(
                (bsz * steps, batch["hand_points"].shape[1]), dtype=torch.bool,
                device=batch["hand_flow"].device,
            ),
            "delta_time_s": torch.full((bsz * steps,), DT, device=batch["hand_flow"].device),
        }
        with torch.no_grad():
            output = self.base(flat)
        features = torch.cat((output["fused_feature"], output["cm_tokens"].mean(1)), -1)
        return self.head(features).reshape(bsz, steps, TOKEN_DIM)


def collate(samples):
    return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}


@torch.no_grad()
def evaluate(model, loader, device, kind, mean, scale):
    model.eval()
    errors, normalized = [], []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        prediction = model(batch["state"], batch["action"]) if kind == "action" else model(batch)
        target = batch["token"]
        prediction = prediction * scale + mean
        errors.append(prediction - target)
        normalized.append((prediction - target) / scale)
    error = torch.cat(errors, 0)
    zerror = torch.cat(normalized, 0)
    return {
        "token_mae": float(error.abs().mean().cpu()),
        "token_rmse": float(error.square().mean().sqrt().cpu()),
        "token_normalized_rmse": float(zerror.square().mean().sqrt().cpu()),
        "per_dim_rmse": error.square().mean((0, 1)).sqrt().cpu().tolist(),
        "contact_mass_rmse": float(error[..., 0].square().mean().sqrt().cpu()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "tmp/P-20261004-cmv2-unified-ei-k1-v7.best1.pt")
    parser.add_argument("--train-envs", type=int, default=16)
    parser.add_argument("--val-envs", type=int, default=8)
    parser.add_argument("--rows-per-env", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    raw = torch.load(args.trace, map_location="cpu", weights_only=False)
    tlen, env_count = raw["action"].shape[:2]
    trace = {
        "action": raw["action"],
        "state": raw["state_after"],
        "q": raw["dof_after"].reshape(tlen, env_count, 18, 2)[..., 0],
        "root": raw["root_after"].reshape(tlen, env_count, 3, 13),
    }
    asset = ROOT / "third_party/DExplore/dexplore/data/assets"
    urdf = asset / "inspire_hand_new/inspire_hand_right.urdf"
    obj_path = asset / "mjcf/objects/airplane/airplane.obj"
    vertices, normals = read_obj(obj_path)
    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(len(vertices), 256, replace=False))
    object_points = torch.from_numpy(vertices[indices])
    object_normals = torch.from_numpy(normals[indices])
    hand = area_hand_samples(urdf, count=512, seed=args.seed)
    hand_surface = torch.from_numpy(hand["points"])
    hand_normals = torch.from_numpy(hand["normals"])
    hand_links = torch.from_numpy(hand["links"]).long()
    fk = TorchInspireKinematics(urdf, device)
    lower, upper = native_joint_limits(urdf, device)
    max_start = tlen - 3
    train_env = np.arange(min(args.train_envs, env_count))
    val_env = np.arange(min(args.train_envs, env_count), min(args.train_envs + args.val_envs, env_count))

    def rows(envs):
        records = []
        for env in envs:
            starts = np.linspace(0, max_start, min(args.rows_per_env, max_start + 1), dtype=int)
            records.extend((trace, int(start), int(env)) for start in starts)
        return records

    train_values = materialize(rows(train_env), trace, object_points, object_normals,
                               hand_surface, hand_normals, hand_links, fk, lower, upper, device)
    val_values = materialize(rows(val_env), trace, object_points, object_normals,
                             hand_surface, hand_normals, hand_links, fk, lower, upper, device)
    train = CachedDataset(train_values)
    val = CachedDataset(val_values)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    mean = train_values["token"].mean((0, 1)).to(device)
    scale = train_values["token"].std((0, 1)).clamp_min(1e-4).to(device)
    action_model = ActionTokenModel().to(device)
    cfg = SimpleNamespace(hidden_width=64, num_tokens=8, use_residual=False,
                          interaction_mode="swept", feature_scale_m=0.02,
                          knn_k=16, interaction_radius_m=0.02, frame_dt_s=DT)
    cm_model = FrozenCmv2TokenModel(cfg, args.checkpoint).to(device)
    action_optimizer = torch.optim.AdamW(action_model.parameters(), lr=3e-4)
    cm_optimizer = torch.optim.AdamW([p for p in cm_model.parameters() if p.requires_grad], lr=3e-4)
    history = []
    for epoch in range(args.epochs):
        for model, optimizer, kind in ((action_model, action_optimizer, "action"),
                                       (cm_model, cm_optimizer, "cmv2")):
            model.train()
            for batch in train_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                prediction = model(batch["state"], batch["action"]) if kind == "action" else model(batch)
                target = (batch["token"] - mean) / scale
                loss = (prediction - target).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        action_metrics = evaluate(action_model, val_loader, device, "action", mean, scale)
        cm_metrics = evaluate(cm_model, val_loader, device, "cmv2", mean, scale)
        history.append({"epoch": epoch + 1, "action": action_metrics, "cmv2_frozen_e": cm_metrics})
        print(json.dumps(history[-1]), flush=True)

    report = {
        "schema": "ref2dex.surface_token_i_probe.v1",
        "run_id": "P-20261004-surface-token-i-k1",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "trace": str(args.trace.resolve()),
        "device": str(device),
        "seed": args.seed,
        "split": {"train_envs": train_env.tolist(), "val_envs": val_env.tolist(),
                  "rows_per_env": args.rows_per_env, "train_rows": len(train), "val_rows": len(val)},
        "target": {
            "name": "K1 object-surface interaction token",
            "components": ["contact_mass", "hand_object_distance", "normal_velocity",
                            "tangential_velocity", "contact_change"],
            "frame": "future object frame",
            "hand_identity_free": True,
            "legacy_13d_reference": {
                "k1_direct_rmse_seed1": 0.106682,
                "k1_cmv2_rmse_seed1": 0.106254,
                "source": "P-20261004-cmv2-unified-ei k1 v7; different target dimensionality",
            },
        },
        "encoder": {"checkpoint": str(args.checkpoint.resolve()), "frozen": True,
                    "base": "ObjectInteractionCmv2V13Model v1_3_rigid_only"},
        "normalization": {"mean": mean.cpu().tolist(), "scale": scale.cpu().tolist()},
        "history": history,
        "best": {"action": history[-1]["action"], "cmv2_frozen_e": history[-1]["cmv2_frozen_e"]},
        "decision": "PROMISING only if frozen-E surface token beats direct token baseline; GT-I independent value requires the separate G bridge probe",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
