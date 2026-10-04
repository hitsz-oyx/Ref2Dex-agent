#!/usr/bin/env python3
"""Matched Inspire action/Cmv2 probe with a hand-identity-free interaction field.

The I target is a pooled object-surface field:
contact mass, contact centroid, weighted covariance, normal approach speed,
tangential speed, and its 90th percentile.  It uses only unordered hand
surface points and object anchors, so it is compatible with MANO or Inspire
geometry.  Future geometry is used only to form labels; Cmv2 hand flow is
generated from the current state and the recorded 18-D command through the
exact DExplore action-to-target mapping and FK.
"""
from __future__ import annotations

import argparse, json, random, subprocess, sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.task.CmResidual.dexplore_cm_geometry import dexplore_action_to_native_targets, dexplore_root_pose, native_joint_limits
from src.task.CmResidual.surface_execution import area_hand_samples
from src.task.CmResidual.v118_planner import TorchInspireKinematics
from src.task.ObjectInteractionCmv2.model import ObjectInteractionCmv2V13Model, _axis_angle_matrix_stable

DT = 1.0 / 30.0
I_DIM = 13


def read_obj(path: Path):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields: continue
        if fields[0] == "v": vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] == "f": faces.append([int(x.split('/')[0]) - 1 for x in fields[1:4]])
    v = np.asarray(vertices, np.float32); f = np.asarray(faces, np.int64)
    normals = np.zeros_like(v)
    tri = v[f]; fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for col in range(3): np.add.at(normals, f[:, col], fn)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-8)
    return v, normals


def axis_angle_from_rotation(rotation):
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    skew = torch.stack((rotation[..., 2, 1] - rotation[..., 1, 2],
                        rotation[..., 0, 2] - rotation[..., 2, 0],
                        rotation[..., 1, 0] - rotation[..., 0, 1]), -1)
    axis = skew / (2.0 * angle.sin().unsqueeze(-1).clamp_min(1e-6))
    small = angle < 1e-4
    return torch.where(small.unsqueeze(-1), 0.5 * skew, axis * angle.unsqueeze(-1))


def field_label(object_points, object_normals, hand_now, hand_future, object_now, object_future):
    """Build 13-D object-surface I in the future object frame."""
    # object_points are canonical local anchors; move hand to future object frame.
    inv = object_future[:, :3, :3].transpose(-1, -2)
    hand = torch.einsum("bij,bnj->bni", inv, hand_future - object_future[:, None, :3, 3])
    dist = torch.cdist(object_points, hand)
    nearest = dist.argmin(-1)
    d = dist.gather(-1, nearest[..., None])[..., 0]
    contact = torch.sigmoid((0.02 - d) / 0.005)
    w = contact / contact.sum(-1, keepdim=True).clamp_min(1e-6)
    centroid = (w[..., None] * object_points).sum(1)
    centered = object_points - centroid[:, None]
    cov = (w[..., None, None] * centered[..., :, None] * centered[..., None, :]).sum(1)
    cov6 = torch.stack((cov[:, 0, 0], cov[:, 1, 1], cov[:, 2, 2], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 2]), -1)
    # Relative contact motion: nearest current/future hand points, object-frame corrected.
    inv_now = object_now[:, :3, :3].transpose(-1, -2)
    hand0_now = torch.einsum("bij,bnj->bni", inv_now, hand_now - object_now[:, None, :3, 3])
    nearest0 = torch.cdist(object_points, hand0_now).argmin(-1)
    # Select the current contact in the current object frame, then express its
    # world position in the same future object frame as ``hand``.
    h0_world = hand_now.gather(1, nearest0[..., None].expand(-1, -1, 3))
    h0 = torch.einsum("bij,bnj->bni", inv, h0_world - object_future[:, None, :3, 3])
    hf = hand.gather(1, nearest[..., None].expand(-1, -1, 3))
    rel = (hf - h0) / DT
    normals = object_normals / object_normals.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    vn = (rel * normals).sum(-1)
    vt = (rel - vn[..., None] * normals).norm(dim=-1)
    weighted_vn = (w * vn).sum(-1, keepdim=True)
    weighted_vt = (w * vt).sum(-1, keepdim=True)
    q90 = torch.quantile(vt, 0.9, dim=-1, keepdim=True)
    return torch.cat((contact.mean(-1, keepdim=True), centroid, cov6, weighted_vn, weighted_vt, q90), -1)


class UnifiedDataset(Dataset):
    def __init__(self, rows, k, object_points, object_normals, hand_surface, hand_normals, hand_links, fk, lower, upper, device):
        self.k = k; self.rows = rows; self.device = device
        self.object_points = object_points.to(device); self.object_normals = object_normals.to(device)
        self.hand_surface = hand_surface.to(device); self.hand_normals = hand_normals.to(device); self.hand_links = hand_links.to(device); self.fk = fk
        self.lower, self.upper, self.device = lower, upper, device
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        run, t, env = self.rows[i]
        trace = run
        qall = trace["q"][:, env].to(self.device); roots = trace["root"][:, env].reshape(trace["root"].shape[0], 3, 13).to(self.device)
        # DExplore actor order is humanoid hand=0, table=1, object=2.
        current_q = qall[t]; current_hand_root = roots[t, 0]; current_obj = roots[t, 2]
        current_obj_pose = dexplore_root_pose(current_obj[None])[0]
        obj = self.object_points
        # Current actual hand geometry is an observation; future hand geometry is label only.
        links = self.fk.forward(current_q[None, None])[0, 0]
        hp_world = self._surface(links, current_hand_root, self.hand_surface, self.hand_normals)[0]
        actions, hand_flows, effects, labels = [], [], [], []
        nominal_q = current_q.clone()
        for h in range(1, self.k + 1):
            action = trace["action"][t + h, env].to(self.device)
            actions.append(action)
            nominal_q = dexplore_action_to_native_targets(action[None], nominal_q[None], self.lower, self.upper)[0]
            nominal_links = self.fk.forward(nominal_q[None, None])[0, 0]
            pred_hp = self._surface(nominal_links, current_hand_root, self.hand_surface, self.hand_normals)[0]
            hand_flows.append(torch.einsum("ij,nj->ni", current_obj_pose[:3, :3].T, pred_hp - hp_world))
            future_q = qall[t + h]; future_roots = roots[t + h]
            future_obj = dexplore_root_pose(future_roots[2][None])[0]
            delta_t = current_obj_pose[:3, :3].T @ (future_obj[:3, 3] - current_obj_pose[:3, 3])
            delta_r = current_obj_pose[:3, :3].T @ future_obj[:3, :3]
            effects.append(torch.cat((delta_t, axis_angle_from_rotation(delta_r[None])[0])))
            future_links = self.fk.forward(future_q[None, None])[0, 0]
            future_hp = self._surface(future_links, future_roots[0], self.hand_surface, self.hand_normals)[0]
            labels.append(field_label(obj[None], self.object_normals[None], hp_world[None], future_hp[None], current_obj_pose[None], future_obj[None])[0])
        state = trace["state"][t, env].to(self.device)
        result = {"state": state.float(), "action": torch.stack(actions), "obj_points": obj.float(),
                "obj_normals": self.object_normals.float(), "hand_points": hp_world.float(),
                "hand_normals": self._surface(links, current_hand_root, self.hand_surface, self.hand_normals)[1].float(),
                "hand_flow": torch.stack(hand_flows), "effect": torch.stack(effects), "interaction": torch.stack(labels)}
        return {key: value.cpu() for key, value in result.items()}
    def _surface(self, links, root, points, normals):
        base = dexplore_root_pose(root[None])[0]
        links = torch.matmul(base, links)
        rot = links[self.hand_links, :3, :3]; trans = links[self.hand_links, :3, 3]
        world = torch.einsum("pij,pj->pi", rot, points) + trans
        normal = torch.einsum("pij,pj->pi", rot, normals)
        return world, normal


@torch.no_grad()
def materialize(rows, k, trace, object_points, object_normals, hand_surface, hand_normals,
                hand_links, fk, lower, upper, device, chunk_size=64):
    """Vectorized replacement for per-sample FK/nearest-neighbor construction."""
    n = len(rows)
    if n > chunk_size:
        parts = [materialize(rows[start:start + chunk_size], k, trace, object_points, object_normals,
                             hand_surface, hand_normals, hand_links, fk, lower, upper, device, chunk_size)
                 for start in range(0, n, chunk_size)]
        return {key: torch.cat([part[key] for part in parts], dim=0) for key in parts[0]}
    object_points = object_points.to(device); object_normals = object_normals.to(device)
    hand_surface = hand_surface.to(device); hand_normals = hand_normals.to(device); hand_links = hand_links.to(device)
    times = torch.tensor([row[1] for row in rows], dtype=torch.long)
    envs = torch.tensor([row[2] for row in rows], dtype=torch.long)
    q_all = trace["q"].to(device); root_all = trace["root"].to(device)
    action_all = trace["action"].to(device); state_all = trace["state"].to(device)
    current_q = q_all[times, envs]
    current_roots = root_all[times, envs]
    # DExplore actor order is humanoid hand=0, table=1, object=2.
    current_obj = dexplore_root_pose(current_roots[:, 2])
    current_hand_root = current_roots[:, 0]
    actions = torch.stack([action_all[times + h, envs] for h in range(1, k + 1)], 1)
    future_q = torch.stack([q_all[times + h, envs] for h in range(1, k + 1)], 1)
    future_roots = torch.stack([root_all[times + h, envs] for h in range(1, k + 1)], 1)

    def surface(links, roots):
        base = dexplore_root_pose(roots.reshape(-1, 13)).reshape(*roots.shape[:-1], 4, 4)
        # Keep the sample and link axes separate; otherwise matmul broadcasts
        # the leading sample axis across other samples when n > 1.
        world_links = torch.matmul(base.unsqueeze(-3), links.unsqueeze(-4))
        rot = world_links[..., hand_links, :3, :3]
        trans = world_links[..., hand_links, :3, 3]
        points = torch.einsum("...pij,pj->...pi", rot, hand_surface) + trans
        normals = torch.einsum("...pij,pj->...pi", rot, hand_normals)
        return points, normals

    current_links = fk.forward(current_q[:, None])[:, 0]
    current_world, current_world_normals = surface(current_links, current_hand_root[:, None])
    current_world, current_world_normals = current_world[:, 0], current_world_normals[:, 0]
    obj_rot = current_obj[:, :3, :3]
    current_hand = torch.einsum("bij,bpj->bpi", obj_rot.transpose(-1, -2), current_world - current_obj[:, None, :3, 3])
    current_hand_normals = torch.einsum("bij,bpj->bpi", obj_rot.transpose(-1, -2), current_world_normals)

    nominal_q = current_q
    predicted_flow = []
    for h in range(k):
        nominal_q = dexplore_action_to_native_targets(actions[:, h], nominal_q, lower, upper)
        nominal_links = fk.forward(nominal_q[:, None])[:, 0]
        nominal_world, _ = surface(nominal_links, current_hand_root[:, None])
        nominal_world = nominal_world[:, 0]
        predicted_flow.append(torch.einsum("bij,bpj->bpi", obj_rot.transpose(-1, -2), nominal_world - current_world))
    predicted_flow = torch.stack(predicted_flow, 1)

    actual_future = []
    for h in range(k):
        links = fk.forward(future_q[:, h:h + 1])[:, 0]
        points, _ = surface(links, future_roots[:, h, 0:1])
        actual_future.append(points[:, 0])
    actual_future = torch.stack(actual_future, 1)
    future_obj = dexplore_root_pose(future_roots[..., 2, :].reshape(-1, 13)).reshape(n, k, 4, 4)
    effects, labels = [], []
    for h in range(k):
        delta_t = obj_rot.transpose(-1, -2) @ (future_obj[:, h, :3, 3] - current_obj[:, :3, 3]).unsqueeze(-1)
        delta_r = obj_rot.transpose(-1, -2) @ future_obj[:, h, :3, :3]
        effects.append(torch.cat((delta_t[..., 0], axis_angle_from_rotation(delta_r)), -1))
        labels.append(field_label(object_points[None].expand(n, -1, -1), object_normals[None].expand(n, -1, -1), current_world, actual_future[:, h], current_obj, future_obj[:, h]))
    labels = torch.stack(labels, 1)
    result = {"state": state_all[times, envs].float(), "action": actions.float(),
              "obj_points": object_points.float().expand(n, -1, -1),
              "obj_normals": object_normals.float().expand(n, -1, -1),
              "hand_points": current_hand.float(), "hand_normals": current_hand_normals.float(),
              "hand_flow": predicted_flow.float(), "effect": torch.stack(effects, 1).float(),
              "interaction": labels.float()}
    return {key: value.cpu() for key, value in result.items()}


class CachedDataset(Dataset):
    def __init__(self, values): self.values = values
    def __len__(self): return self.values["state"].shape[0]
    def __getitem__(self, index): return {key: value[index] for key, value in self.values.items()}


def derived_interaction(effect, batch):
    """Geometric I obtained from predicted E and the same command-derived hand flow."""
    b, k = effect.shape[:2]
    rotation = _axis_angle_matrix_stable(effect[..., 3:].reshape(-1, 3)).reshape(b, k, 3, 3)
    future_pose = torch.eye(4, device=effect.device, dtype=effect.dtype).expand(b, k, 4, 4).clone()
    future_pose[..., :3, :3] = rotation
    future_pose[..., :3, 3] = effect[..., :3]
    current_pose = torch.eye(4, device=effect.device, dtype=effect.dtype).expand(b, 4, 4)
    values = []
    for h in range(k):
        values.append(field_label(batch["obj_points"], batch["obj_normals"], batch["hand_points"],
                                   batch["hand_points"] + batch["hand_flow"][:, h], current_pose,
                                   future_pose[:, h]))
    return torch.stack(values, 1)


class ActionModel(nn.Module):
    def __init__(self, state_dim, k, hidden=128):
        super().__init__(); self.k = k; self.inp = nn.Sequential(nn.Linear(state_dim + 18, hidden), nn.SiLU(), nn.Linear(hidden, hidden)); self.gru = nn.GRU(hidden, hidden, batch_first=True); self.head = nn.Linear(hidden, 6 + I_DIM)
    def forward(self, state, action):
        h = self.inp(torch.cat((state[:, None].expand(-1, action.shape[1], -1), action), -1)); h, _ = self.gru(h); y = self.head(h); return y[..., :6], y[..., 6:]


class Cmv2Model(nn.Module):
    def __init__(self, cfg, k):
        super().__init__(); self.k = k; self.base = ObjectInteractionCmv2V13Model(cfg); self.gru = nn.GRU(cfg.hidden_width, cfg.hidden_width, batch_first=True); self.context = nn.Sequential(nn.Linear(55 + 18, cfg.hidden_width), nn.SiLU(), nn.Linear(cfg.hidden_width, cfg.hidden_width)); self.field_context = nn.Sequential(nn.Linear(15, cfg.hidden_width), nn.SiLU(), nn.Linear(cfg.hidden_width, cfg.hidden_width)); self.i_head = nn.Linear(cfg.hidden_width * 3, I_DIM)
    def forward(self, batch):
        b, k = batch["hand_flow"].shape[:2]
        rep = lambda x: x[:, None].expand(-1, k, *x.shape[1:]).reshape(b*k, *x.shape[1:])
        flat = {"obj_points": rep(batch["obj_points"]), "obj_normals": rep(batch["obj_normals"]), "hand_points": rep(batch["hand_points"]), "hand_normals": rep(batch["hand_normals"]), "hand_flow": batch["hand_flow"].reshape(b*k, -1, 3), "hand_valid_mask": torch.ones((b*k, batch["hand_points"].shape[1]), dtype=torch.bool, device=batch["hand_flow"].device), "delta_time_s": torch.full((b*k,), DT, device=batch["hand_flow"].device)}
        out = self.base(flat); z, _ = self.gru(out["fused_feature"].reshape(b, k, -1)); context = self.context(torch.cat((batch["state"][:, None].expand(-1, k, -1), batch["action"]), -1))
        object_points = flat["obj_points"]; object_normals = flat["obj_normals"]; hand_points = flat["hand_points"]; hand_flow = flat["hand_flow"]
        distances = torch.cdist(object_points, hand_points); nearest = distances.argmin(-1); distance = distances.gather(-1, nearest[..., None])[..., 0]
        contact = torch.sigmoid((0.02 - distance) / 0.005); weights = contact / contact.sum(-1, keepdim=True).clamp_min(1e-6)
        centroid = (weights[..., None] * object_points).sum(1); flow = hand_flow.gather(1, nearest[..., None].expand(-1, -1, 3)); flow_mean = (weights[..., None] * flow).sum(1)
        normals = F.normalize(object_normals, dim=-1, eps=1e-8); normal_speed = (flow * normals).sum(-1); tangent_speed = (flow - normal_speed[..., None] * normals).norm(dim=-1)
        field = torch.cat((contact.mean(-1, keepdim=True), centroid, flow_mean, (weights * normal_speed).sum(-1, keepdim=True), (weights * tangent_speed).sum(-1, keepdim=True), out["delta_xi_root"]), -1).reshape(b, k, -1)
        field = self.field_context(field.reshape(b * k, -1)).reshape(b, k, -1)
        return out["delta_xi_root"].reshape(b, k, 6), self.i_head(torch.cat((z, context, field), -1))


def evaluate(model, loader, device, kind, interaction_mean, interaction_scale):
    model.eval(); sums = torch.zeros(4); derived_sum = torch.zeros(1); count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}; ep, it_norm = model(batch["state"], batch["action"]) if kind == "action" else model(batch)
            it = it_norm * interaction_scale + interaction_mean
            derived = derived_interaction(ep, batch)
            sums[0] += (ep - batch["effect"]).abs().mean().cpu(); sums[1] += (it - batch["interaction"]).abs().mean().cpu(); sums[2] += (ep - batch["effect"]).pow(2).mean().sqrt().cpu(); sums[3] += (it - batch["interaction"]).pow(2).mean().sqrt().cpu(); count += 1
            derived_sum += (derived - batch["interaction"]).pow(2).mean().sqrt().cpu()
    return dict(effect_mae=float(sums[0]/count), interaction_mae=float(sums[1]/count), effect_rmse=float(sums[2]/count), interaction_rmse=float(sums[3]/count), derived_interaction_rmse=float(derived_sum/count))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--trace", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--k", type=int, default=1); p.add_argument("--train-envs", type=int, default=48); p.add_argument("--val-envs", type=int, default=24); p.add_argument("--rows-per-env", type=int, default=128); p.add_argument("--epochs", type=int, default=8); p.add_argument("--device", default="cuda:1"); p.add_argument("--seed", type=int, default=20261004)
    a = p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if a.output.exists(): raise FileExistsError(a.output)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    raw = torch.load(a.trace, map_location="cpu", weights_only=False); tlen, envs = raw["action"].shape[:2]
    q = raw["dof_after"].reshape(tlen, envs, 18, 2)[..., 0]; root = raw["root_after"].reshape(tlen, envs, 3, 13)
    trace = {"action": raw["action"], "state": raw["state_after"], "q": q, "root": root}
    asset = ROOT / "third_party/DExplore/dexplore/data/assets"; urdf = asset / "inspire_hand_new/inspire_hand_right.urdf"; obj_path = asset / "mjcf/objects/airplane/airplane.obj"
    verts, norms = read_obj(obj_path); rng = np.random.default_rng(a.seed); ids = np.sort(rng.choice(len(verts), 256, replace=False)); obj = torch.from_numpy(verts[ids]); objn = torch.from_numpy(norms[ids])
    surface = area_hand_samples(urdf, count=512, seed=a.seed); hp = torch.from_numpy(surface["points"]); hn = torch.from_numpy(surface["normals"])
    fk = TorchInspireKinematics(urdf, device); lower, upper = native_joint_limits(urdf, device)
    max_start = tlen - a.k - 2; rng = np.random.default_rng(a.seed); train_env = np.arange(min(a.train_envs, envs)); val_env = np.arange(min(a.train_envs, envs), min(a.train_envs + a.val_envs, envs));
    def rows(env_ids):
        out=[]
        for e in env_ids:
            starts = np.linspace(0, max_start, min(a.rows_per_env, max_start + 1), dtype=int)
            out.extend((trace, int(t), int(e)) for t in starts)
        return out
    hand_links = torch.from_numpy(surface["links"]).long()
    train_values = materialize(rows(train_env), a.k, trace, obj, objn, hp, hn, hand_links, fk, lower, upper, device)
    val_values = materialize(rows(val_env), a.k, trace, obj, objn, hp, hn, hand_links, fk, lower, upper, device)
    train = CachedDataset(train_values); val = CachedDataset(val_values)
    def collate(samples): return {k: torch.stack([x[k] for x in samples]) for k in samples[0]}
    train_loader = DataLoader(train, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate); val_loader = DataLoader(val, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate)
    interaction_mean = train_values["interaction"].mean(dim=(0, 1)).to(device)
    interaction_scale = train_values["interaction"].std(dim=(0, 1)).clamp_min(1e-4).to(device)
    action_model = ActionModel(55, a.k).to(device); cfg = SimpleNamespace(hidden_width=64, num_tokens=8, use_residual=False, interaction_mode="swept", feature_scale_m=0.02, knn_k=16, interaction_radius_m=0.02, frame_dt_s=DT); cm_model = Cmv2Model(cfg, a.k).to(device)
    opts = [torch.optim.AdamW(action_model.parameters(), lr=3e-4), torch.optim.AdamW(cm_model.parameters(), lr=2e-4)]; best = [{"score": float("inf")}, {"score": float("inf")}]
    history=[]
    for epoch in range(a.epochs):
        for model, opt, kind in ((action_model, opts[0], "action"), (cm_model, opts[1], "cm")):
            model.train()
            for batch in train_loader:
                batch = {k: v.to(device) for k,v in batch.items()}; ep,it_norm = model(batch["state"], batch["action"]) if kind=="action" else model(batch); interaction_target = (batch["interaction"] - interaction_mean) / interaction_scale; loss = (ep-batch["effect"]).pow(2).mean() + 0.5*(it_norm-interaction_target).pow(2).mean(); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        ar=evaluate(action_model,val_loader,device,"action",interaction_mean,interaction_scale); cr=evaluate(cm_model,val_loader,device,"cm",interaction_mean,interaction_scale); history.append({"epoch":epoch+1,"action":ar,"cmv2":cr}); print(json.dumps(history[-1]),flush=True)
        for model, metrics, slot in ((action_model,ar,0),(cm_model,cr,1)):
            score=metrics["effect_rmse"]+metrics["interaction_rmse"]
            if score < best[slot]["score"]: best[slot]={"score":score,"epoch":epoch+1}; torch.save(model.state_dict(), a.output.with_suffix(f".best{slot}.pt"))
    report={"schema":"ref2dex.cmv2_unified_ei_probe.v1","run_id":f"P-20261004-cmv2-unified-ei-k{a.k}","code_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"trace":str(a.trace.resolve()),"device":str(device),"k":a.k,"i_definition":"object surface pooled [contact_mass, centroid3, covariance6, normal_approach, tangent_speed, q90_tangent_speed]","interaction_training_normalization":{"mean":interaction_mean.cpu().tolist(),"scale":interaction_scale.cpu().tolist()},"train_envs":train_env.tolist(),"val_envs":val_env.tolist(),"best":best,"history":history,"status":"PROMISING only if Cmv2 improves interaction RMSE over action baseline; command-proxy, not a final policy claim"}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__ == "__main__": main()
