#!/usr/bin/env python3
"""Bounded GT surface-field information probe; no deployable predictor inputs.

Eight fixed mesh patches retain object spatial topology. Each patch stores
soft proximity mass, weighted nearest surface distance, relative normal and
tangential velocities, and proximity-mass change over one simulator step.
These are geometric surface proxies, not measured contact forces.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from probe_ref4_g_bridge import Bridge, _split, _standardize, _bootstrap
from probe_surface_token_i import read_obj, axis_angle_from_rotation
from assemble_gate1_dataset_v2 import _quat_rotate, _quat_conjugate, _quat_normalize
from src.task.CmResidual.surface_execution import area_hand_samples
from src.task.CmResidual.v118_planner import TorchInspireKinematics, QUERY_LINKS
from src.task.CmResidual.dexplore_cm_geometry import dexplore_root_pose
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_pointflow_g import pose_effect


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patch_assignment(points, count=8):
    centers = [int(points.square().sum(-1).argmax())]
    distances = (points - points[centers[0]]).square().sum(-1)
    for _ in range(count - 1):
        centers.append(int(distances.argmax()))
        distances = torch.minimum(distances, (points - points[centers[-1]]).square().sum(-1))
    labels = torch.cdist(points, points[centers]).argmin(-1)
    return labels, centers


def field(points, normals, hand0, hand1, obj0, obj1, labels, dt):
    # Compare the same material hand sample in contemporaneous object frames.
    def local(hand, obj):
        inverse = _quat_conjugate(_quat_normalize(obj[:, 3:7]))
        return _quat_rotate(inverse[:, None].expand(-1, hand.shape[1], -1), hand - obj[:, None, :3])
    h0, h1 = local(hand0, obj0), local(hand1, obj1)
    distances1 = torch.cdist(points[None].expand(len(hand1), -1, -1), h1)
    nearest = distances1.argmin(-1)
    d1 = distances1.amin(-1)
    d0 = torch.cdist(points[None].expand(len(hand0), -1, -1), h0).amin(-1)
    gather = nearest[..., None].expand(-1, -1, 3)
    velocity = (h1.gather(1, gather) - h0.gather(1, gather)) / dt
    vn = (velocity * normals[None]).sum(-1)
    vt = (velocity - vn[..., None] * normals[None]).norm(dim=-1)
    c0, c1 = torch.sigmoid((0.02 - d0) / 0.005), torch.sigmoid((0.02 - d1) / 0.005)
    patches = []
    for patch in range(8):
        mask = labels == patch
        weights = c1[:, mask] / c1[:, mask].sum(-1, keepdim=True).clamp_min(1e-6)
        patches.append(torch.stack((c1[:, mask].mean(-1), (weights * d1[:, mask]).sum(-1),
                                    (weights * vn[:, mask]).sum(-1), (weights * vt[:, mask]).sum(-1),
                                    (c1[:, mask] - c0[:, mask]).mean(-1)), -1))
    return torch.stack(patches, 1)


def prepare(path, device, max_per_episode):
    source = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    groups = sorted(set(zip(source["source_run"].tolist(), source["episode_id"].tolist())))
    selected = []
    for run, episode in groups:
        idx = ((source["source_run"] == run) & (source["episode_id"] == episode)).nonzero().flatten()
        idx = idx[torch.argsort(source["step"][idx])]
        selected.extend(idx[torch.linspace(0, len(idx) - 1, min(max_per_episode, len(idx))).round().long()].tolist())
    subset = torch.tensor(selected)
    raw = {"H": torch.cat([source[k][subset].float() for k in
                          ("history_state", "history_previous_action", "history_context", "history_progress")], -1),
           "_namespace": torch.zeros(len(subset), dtype=torch.long),
           "_run": source["source_run"][subset], "_episode": source["episode_id"][subset]}
    asset = ROOT / "third_party/DExplore/dexplore/data/assets"
    urdf = asset / "inspire_hand_new/inspire_hand_right.urdf"
    mesh = asset / "mjcf/objects/airplane/airplane.obj"
    vertices, normals = read_obj(mesh)
    anchors = np.sort(np.random.default_rng(20261004).choice(len(vertices), 256, replace=False))
    points, normals = torch.tensor(vertices[anchors], device=device), torch.tensor(normals[anchors], device=device)
    labels, centers = patch_assignment(points)
    hand = area_hand_samples(urdf, count=512, seed=20261004)
    hand_points = torch.tensor(hand["points"], device=device)
    links = torch.tensor(hand["links"], device=device).long()
    fk = TorchInspireKinematics(urdf, device)
    contact_links = [QUERY_LINKS.index(n) for n in
                     ("index_intermediate", "middle_intermediate", "pinky_intermediate", "ring_intermediate", "thumb_distal")]
    interaction, effects = torch.empty(len(subset), 8, 5), torch.empty(len(subset), 1, 6)
    audit = {"fk_position_max_error_m": 0., "state_alignment_max_error": 0.,
             "inferred_hand_root_translation_max_m": 0., "inferred_hand_root_rotation_from_identity_max": 0.,
             "state_object_root_max_error": 0., "reconstructed_E_max_error": 0.,
             "next_state_alignment_max_error": 0., "shards": [], "patch_anchor_counts": torch.bincount(labels).tolist(),
             "patch_center_anchor_indices": centers, "urdf_sha256": sha(urdf), "mesh_sha256": sha(mesh)}
    for info in source["metadata"]["runs"]:
        shards = sorted((ROOT / info["run_dir"]).glob("transitions_*.pt"))
        hashes = [sha(p) for p in shards]
        if hashes != info["shard_sha256"]:
            raise RuntimeError("input shards changed")
        parts = [torch.load(p, map_location="cpu", weights_only=False, mmap=True) for p in shards]
        merged = {k: torch.cat([part[k] for part in parts]) for k in
                  ("state", "next_state", "object_root", "hand_body_position", "hand_body_quaternion", "episode_id", "step")}
        # Historical file predates hash-sorted source_run IDs. Infer its run ID
        # from globally unique recorded episode IDs, then verify every state.
        shard_episodes = set(merged["episode_id"].tolist())
        matching = torch.tensor([int(ep) in shard_episodes for ep in source["episode_id"]])
        matching_runs = source["source_run"][matching].unique()
        if len(matching_runs) != 1:
            raise RuntimeError("ambiguous source run / episode identity")
        run_id = int(matching_runs[0])
        lookup = {(int(ep), int(step)): i for i, (ep, step) in enumerate(zip(merged["episode_id"], merged["step"]))}
        out_idx = (raw["_run"] == run_id).nonzero().flatten()
        current = torch.tensor([lookup[(int(raw["_episode"][i]), int(source["step"][subset[i]]))] for i in out_idx])
        future = torch.tensor([lookup[(int(raw["_episode"][i]), int(source["step"][subset[i]]) + 1)] for i in out_idx])
        state_error = (merged["state"][current] - source["state"][subset[out_idx]]).abs().max().item()
        next_error = (merged["next_state"][current] - merged["state"][future]).abs().max().item()
        audit["state_alignment_max_error"] = max(audit["state_alignment_max_error"], state_error)
        audit["next_state_alignment_max_error"] = max(audit["next_state_alignment_max_error"], next_error)
        state_object = merged["state"][current, 36:49].clone()
        shard_object = merged["object_root"][current].clone()
        state_object[:, 3:7] = _quat_normalize(state_object[:, 3:7])
        shard_object[:, 3:7] = _quat_normalize(shard_object[:, 3:7])
        shard_object[:, 3:7] *= torch.where((state_object[:, 3:7] * shard_object[:, 3:7]).sum(-1, keepdim=True) < 0, -1., 1.)
        object_error = (state_object - shard_object).abs().max().item()
        audit["state_object_root_max_error"] = max(audit["state_object_root_max_error"], object_error)
        if state_error > 1e-5:
            raise RuntimeError("assembled decision and shard state mismatch")
        for start in range(0, len(out_idx), 64):
            where = out_idx[start:start + 64]
            i0, i1 = current[start:start + 64], future[start:start + 64]
            q0, q1 = merged["state"][i0, :18].to(device), merged["state"][i1, :18].to(device)
            transforms0, transforms1 = fk.forward(q0[:, None])[:, 0], fk.forward(q1[:, None])[:, 0]
            # Free actor root is absent from state55 and can move. Recover a
            # common rigid root from one measured body solely for GT labels,
            # then verify it against the other four independently stored bodies.
            def correct_root(transforms, ii):
                body = torch.cat((merged["hand_body_position"][ii, 0], merged["hand_body_quaternion"][ii, 0],
                                  torch.zeros(len(ii), 6)), -1).to(device)
                measured = dexplore_root_pose(body)
                fk_body = transforms[:, contact_links[0]]
                correction = torch.eye(4, device=device).expand(len(ii), 4, 4).clone()
                correction[:, :3, :3] = measured[:, :3, :3] @ fk_body[:, :3, :3].transpose(-1, -2)
                correction[:, :3, 3] = measured[:, :3, 3] - torch.einsum("bij,bj->bi", correction[:, :3, :3], fk_body[:, :3, 3])
                audit["inferred_hand_root_translation_max_m"] = max(audit["inferred_hand_root_translation_max_m"], correction[:, :3, 3].norm(dim=-1).max().item())
                audit["inferred_hand_root_rotation_from_identity_max"] = max(audit["inferred_hand_root_rotation_from_identity_max"],
                                                                           (correction[:, :3, :3] - torch.eye(3, device=device)).abs().max().item())
                return correction[:, None] @ transforms
            transforms0, transforms1 = correct_root(transforms0, i0), correct_root(transforms1, i1)
            for transforms, ii in ((transforms0, i0), (transforms1, i1)):
                position_error = (transforms[:, contact_links, :3, 3].cpu() - merged["hand_body_position"][ii]).norm(dim=-1)
                error = position_error.max().item()
                if error > audit["fk_position_max_error_m"]:
                    flat = int(position_error.argmax())
                    bad = int(ii[flat // 5])
                    audit["fk_worst"] = {"run_dir": info["run_dir"], "episode_id": int(merged["episode_id"][bad]),
                                         "step": int(merged["step"][bad]), "contact_body_index": flat % 5,
                                         "actual": merged["hand_body_position"][bad, flat % 5].tolist(),
                                         "predicted": transforms[flat // 5, contact_links[flat % 5], :3, 3].cpu().tolist()}
                audit["fk_position_max_error_m"] = max(audit["fk_position_max_error_m"], error)
            if audit["fk_position_max_error_m"] > 0.001:
                raise RuntimeError(json.dumps(audit))
            def surface(transforms):
                return torch.einsum("bpij,pj->bpi", transforms[:, links, :3, :3], hand_points) + transforms[:, links, :3, 3]
            obj0, obj1 = merged["object_root"][i0].to(device), merged["object_root"][i1].to(device)
            interaction[where] = field(points, normals, surface(transforms0), surface(transforms1), obj0, obj1,
                                       labels, float(info["control_dt"])).cpu()
            translation = _quat_rotate(_quat_conjugate(_quat_normalize(obj0[:, 3:7])), obj1[:, :3] - obj0[:, :3])
            from assemble_gate1_dataset_v2 import _relative_quaternion
            relative = _relative_quaternion(obj0[:, 3:7], obj1[:, 3:7])
            reconstructed = pose_effect(torch.cat((translation, relative), -1)).cpu()
            source_effect = pose_effect(source["effect"][subset[where], :1])[:, 0]
            audit["reconstructed_E_max_error"] = max(audit["reconstructed_E_max_error"], (reconstructed - source_effect).abs().max().item())
            effects[where, 0] = source_effect
        audit["shards"].extend({"path": str(p), "sha256": h} for p, h in zip(shards, hashes))
    raw["E"], raw["I"] = effects, interaction.reshape(len(subset), 1, 40)
    audit["selected_rows"] = len(subset)
    audit["selected_index_sha256"] = hashlib.sha256(subset.numpy().tobytes()).hexdigest()
    audit["I_shape"] = list(interaction.shape)
    audit["I_std"] = interaction.std((0, 1)).tolist()
    audit["positive_softcontact_fraction_gt_0_5"] = float((interaction[..., 0] > .5).float().mean())
    if not all(torch.isfinite(v).all() for v in (interaction, effects, raw["H"])):
        raise RuntimeError("nonfinite features")
    return raw, source["return_to_go"][subset], audit


def fit(blocks, y, train, test, test_groups, device, epochs, seed, input_i):
    torch.manual_seed(seed)
    model = Bridge({k: v.shape[-1] for k, v in blocks.items() if k in ("H", "E", "I")}, ["H", "E", "I"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    target_mean, target_scale = y[train].mean(), y[train].std(unbiased=False).clamp_min(1e-6)
    target = ((y - target_mean) / target_scale).to(device)
    inputs = {k: blocks[k].to(device) for k in ("H", "E", "I")}
    inputs["I"] = input_i.to(device)
    indices = train.nonzero().flatten().to(device)
    losses = []
    for epoch in range(epochs):
        order = indices[torch.randperm(len(indices), device=device)]
        total = 0.
        for idx in order.split(512):
            prediction = model({k: v[idx] for k, v in inputs.items()})
            loss = nn.functional.mse_loss(prediction, target[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        losses.append(total / len(indices))
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model({k: v[s:s + 512] for k, v in inputs.items()}).cpu()
                          for s in range(0, len(y), 512)]) * target_scale + target_mean
    keys = list(zip(blocks["_namespace"][test].tolist(), blocks["_run"][test].tolist(), blocks["_episode"][test].tolist()))
    error = (pred[test] - y[test]).abs()
    group_mae = [float(error[torch.tensor([tuple(k) == g for k in keys])].mean()) for g in test_groups]
    return {"parameter_count": sum(p.numel() for p in model.parameters()), "train_mae": float((pred[train] - y[train]).abs().mean()),
            "test_mae": float(error.mean()), "test_episode_balanced_mae": float(np.mean(group_mae)),
            "test_group_mae": group_mae, "training_loss": losses,
            "_predictions": pred[test], "_targets": y[test], "_groups": keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "tmp/e260_all4_h16_histfix.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--model-seed", type=int, default=202)
    parser.add_argument("--rows-per-episode", type=int, default=64)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    raw, y, audit = prepare(args.input, device, args.rows_per_episode)
    report = {"schema": "ref2dex.gt_surface_i_value.v1", "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "input_sha256": sha(args.input), "script_sha256": sha(Path(__file__)), "argv": sys.argv,
              "model_seed": args.model_seed, "split_seed": 20261004, "geometry_seed": 20261004,
              "audit": audit, "consequence_horizon": 1, "effect_dim": 6,
              "target": "stored exact Monte Carlo simulator reward return; no policy utility claim",
              "I_contract": "8 spatial surface patches x 5 geometric quantities; GT future native q/object pose and common hand-root reconstructed from measured body pose only diagnostic labels",
              "decision_rule": ">=3% episode-balanced gain vs capacity-matched zero-I and positive bootstrap delta; otherwise do not train an I predictor"}
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.audit_only:
        print(json.dumps(report, indent=2), flush=True)
        return
    train, test, train_groups, test_groups = _split(raw, 20261004)
    blocks = {k: _standardize(raw[k], train) for k in ("H", "E", "I")}
    blocks.update({k: raw[k] for k in ("_namespace", "_run", "_episode")})
    pooled = raw["I"].reshape(-1, 1, 8, 5).mean(2, keepdim=True).expand(-1, -1, 8, -1).reshape(-1, 1, 40)
    controls = {"HE_zero_I": torch.zeros_like(blocks["I"]), "HE_surface_I_GT": blocks["I"],
                "HE_pooled_I_GT": _standardize(pooled, train)}
    fitted = {}
    for name, values in controls.items():
        fitted[name] = fit(blocks, y, train, test, test_groups, device, args.epochs, args.model_seed, values)
        print(json.dumps({"variant": name, "mae": fitted[name]["test_episode_balanced_mae"]}), flush=True)
    report.update(train_groups=train_groups, test_groups=test_groups, epochs=args.epochs,
                  variants={k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in fitted.items()},
                  comparison_surface_vs_zero=_bootstrap(fitted["HE_zero_I"], fitted["HE_surface_I_GT"], 20261004),
                  comparison_surface_vs_pooled=_bootstrap(fitted["HE_pooled_I_GT"], fitted["HE_surface_I_GT"], 20261004),
                  elapsed_seconds=time.monotonic() - started)
    comparison = report["comparison_surface_vs_zero"]
    report["status"] = "PROMISING" if comparison["relative_mae_reduction"] >= .03 and comparison["episode_bootstrap_delta_ci95"][0] > 0 else "UNPROMISING"
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "comparison_surface_vs_zero", "comparison_surface_vs_pooled", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
