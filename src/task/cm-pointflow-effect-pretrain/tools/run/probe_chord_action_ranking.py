#!/usr/bin/env python3
"""Probe whether current simulator wrench context improves action ranking.

This is a bounded, data-only/geometry-only simulator probe.  A read-only
DExplore checkout supplies the Inspire simulator and one already materialized
interaction trajectory.  At several reference frames the same state is
replayed across candidate actions.  A ridge effect model ranks candidates from
state/action features (E-only) or from those features plus the current
contact-force/wrench-support context (E+C).  No actor or Cm checkpoint is
loaded.  The experiment is a ranking/evidence probe, not a policy claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int


PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_EXTERNAL = Path("/home2/wyy/oyx_ws/Ref2Dex/third_party/DExplore")
DEFAULT_MOTION = Path(
    "/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/"
    "inspire_geometric_dexplore_coupled_v1_20260912/s1_airplane_lift"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--motion-root", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--fit-frames", type=int, nargs="+", default=[48, 56, 64, 72])
    parser.add_argument("--held-frames", type=int, nargs="+", default=[80, 88, 96, 104])
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def build_external_task(args):
    external = args.external_root.resolve()
    if not external.is_dir():
        raise FileNotFoundError(external)
    if not args.motion_root.is_dir():
        raise FileNotFoundError(args.motion_root)
    if args.num_envs < 8:
        raise ValueError("at least eight candidate environments are required")
    os.chdir(external)
    sys.path.insert(0, str(external / "dexplore"))
    sys.path.insert(0, str(PROJECT_ROOT / "src/task/cm-pointflow-effect-pretrain/src"))

    # Isaac Gym must be imported before torch.
    from isaacgym import gymapi  # noqa: F401
    import torch
    from isaacgym.torch_utils import quat_rotate_inverse
    from utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed
    from utils.parse_task import parse_task
    from oakink_wm.chord_wrench import contact_wrench_rays, support_vector

    set_np_formatting()
    sys.argv = [
        "probe_chord_action_ranking.py",
        "--task", "Dexplore_Inspire",
        "--cfg_env", "dexplore/data/cfg/inspire.yaml",
        "--cfg_train", "dexplore/data/cfg/train/rlg/inspire.yaml",
        "--num_envs", str(args.num_envs),
        "--episode_length", "160",
        "--headless",
        "--seed", str(args.seed),
        "--sim_device", f"cuda:{args.device_id}",
        "--pipeline", "gpu",
    ]
    dex_args = get_args()
    cfg, cfg_train, _ = load_cfg(dex_args)
    cfg["env"]["motion_file"] = str(args.motion_root.resolve())
    cfg["env"]["enableEarlyTermination"] = False
    cfg["env"]["stateInit"] = "Start"
    cfg_train["params"]["seed"] = set_seed(
        cfg_train["params"].get("seed", -1), False
    )
    sim_params = parse_sim_params(dex_args, cfg, cfg_train)
    task, env = parse_task(dex_args, cfg, cfg_train, sim_params)
    task._enable_early_termination = False
    return task, env, torch, quat_rotate_inverse, contact_wrench_rays, support_vector


def reference_action(task, desired_frame, torch):
    frame = min(int(desired_frame), int(task.max_episode_length[0].detach().cpu()) - 1)
    desired = task.hoi_refs[0, 0, frame, 119:119 + task.num_dof]
    current = task._dof_pos
    action = torch.zeros((task.num_envs, task.num_dof), device=task.device)
    action[:, :6] = (desired[:6].unsqueeze(0) - current[:, :6]) / task._pd_action_scale[:6].clamp_min(1e-6)
    action[:, 6:] = 2.0 * desired[6:].unsqueeze(0) / task._pd_action_scale[6:].clamp_min(1e-6) - 1.0
    return action.clamp(-1.0, 1.0)


def wrench_context(task, torch, quat_rotate_inverse, contact_wrench_rays, support_vector):
    force = task._contact_forces[:, task._contact_body_ids, :].detach()
    body_pos = task._rigid_body_pos[:, task._contact_body_ids, :].detach()
    object_pos = task._target_states[:, :3].detach()
    object_quat = task._target_states[:, 3:7].detach()
    object_quat_body = object_quat.unsqueeze(1).expand(-1, body_pos.shape[1], -1)
    points_object = quat_rotate_inverse(
        object_quat_body.reshape(-1, 4),
        (body_pos - object_pos.unsqueeze(1)).reshape(-1, 3),
    ).reshape_as(body_pos)
    normals_object = quat_rotate_inverse(
        object_quat_body.reshape(-1, 4), force.reshape(-1, 3)
    ).reshape_as(force)
    mask = force.norm(dim=-1) > 1e-5
    rays, valid = contact_wrench_rays(
        points_object, normals_object, mask,
        friction=0.5, ray_count=8, body_radius=0.05,
    )
    basis = torch.cat((torch.eye(6, device=task.device), -torch.eye(6, device=task.device)), dim=1)
    support = support_vector(rays, valid, basis)
    object_force = task._tar_contact_forces.detach()
    context = torch.cat((force.flatten(1), object_force, support), dim=-1)
    return context, valid.sum(dim=1)


def candidate_deltas(torch, device):
    """Small, fixed action perturbations around the reference action."""
    zero = torch.zeros(18, device=device)
    rows = [zero]
    for index, amount in ((0, 0.10), (0, -0.10), (2, 0.10), (2, -0.10),
                          (6, 0.10), (6, -0.10), (10, 0.10), (10, -0.10),
                          (14, 0.10), (14, -0.10), (15, 0.10)):
        row = zero.clone()
        row[index] = amount
        rows.append(row)
    return torch.stack(rows, dim=0)


def collect(args, task, env, torch, quat_rotate_inverse, contact_wrench_rays, support_vector):
    deltas = candidate_deltas(torch, task.device)
    if task.num_envs < len(deltas):
        raise ValueError(f"num_envs={task.num_envs} < candidates={len(deltas)}")
    deltas = deltas[:task.num_envs]
    rows = []
    frame_sets = [("fit", int(x)) for x in args.fit_frames] + [("held", int(x)) for x in args.held_frames]
    motion_len = int(task.max_episode_length[0].detach().cpu())
    for split, frame in frame_sets:
        if frame < 1 or frame + args.horizon >= motion_len:
            raise ValueError(f"frame {frame} outside motion horizon {motion_len}")
        task.reset()
        for warm_frame in range(frame):
            env.step(reference_action(task, warm_frame + 1, torch))
        task._refresh_sim_tensors()
        state = torch.cat((task._dof_pos.detach(), task._dof_vel.detach(), task._target_states.detach()), dim=-1)
        context, valid_rays = wrench_context(task, torch, quat_rotate_inverse, contact_wrench_rays, support_vector)
        base = reference_action(task, frame + 1, torch)
        action = (base + deltas).clamp(-1.0, 1.0)
        env.step(action)
        for future_frame in range(frame + 2, frame + args.horizon + 1):
            env.step(reference_action(task, future_frame, torch))
        task._refresh_sim_tensors()
        actual = task._target_states[:, :3].detach()
        reference = task.hoi_refs[0, 0, frame + args.horizon, 106:109].unsqueeze(0)
        target_error = (actual - reference).norm(dim=-1)
        # One row per candidate; all candidate features share the pre-action
        # state/context, while action and future outcome differ.
        for i in range(task.num_envs):
            rows.append({
                "split": split,
                "frame": frame,
                "candidate": i,
                "state": state[i].detach().cpu().numpy().astype(np.float64),
                "action": action[i].detach().cpu().numpy().astype(np.float64),
                "context": context[i].detach().cpu().numpy().astype(np.float64),
                "valid_rays": int(valid_rays[i].detach().cpu()),
                "target": float(-target_error[i].detach().cpu()),
            })
    return rows


def fit_ridge(rows, feature_key, lam=1e-2):
    fit = [r for r in rows if r["split"] == "fit"]
    x = np.stack([r[feature_key] for r in fit])
    y = np.asarray([r["target"] for r in fit], dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    z = np.concatenate([z, np.ones((len(z), 1))], axis=1)
    reg = np.eye(z.shape[1], dtype=np.float64) * lam
    reg[-1, -1] = 0.0
    weight = np.linalg.solve(z.T @ z + reg, z.T @ y)
    return mean, scale, weight


def rank_eval(rows, feature_key, params):
    mean, scale, weight = params
    groups = defaultdict(list)
    for r in rows:
        if r["split"] == "held":
            groups[r["frame"]].append(r)
    details = []
    for frame, group in sorted(groups.items()):
        x = np.stack([r[feature_key] for r in group])
        z = (x - mean) / scale
        pred = np.concatenate([z, np.ones((len(z), 1))], axis=1) @ weight
        actual = np.asarray([r["target"] for r in group])
        chosen = int(np.argmax(pred))
        best = int(np.argmax(actual))
        details.append({
            "frame": frame,
            "chosen_candidate": chosen,
            "best_candidate": best,
            "top1_hit": chosen == best,
            "regret": float(actual[best] - actual[chosen]),
            "predicted": pred.tolist(),
            "actual": actual.tolist(),
        })
    return {
        "frames": len(details),
        "top1_hit_rate": float(np.mean([x["top1_hit"] for x in details])) if details else 0.0,
        "mean_regret": float(np.mean([x["regret"] for x in details])) if details else 0.0,
        "details": details,
    }


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    task, env, torch, quat_rotate_inverse, contact_wrench_rays, support_vector = build_external_task(args)
    rows = collect(args, task, env, torch, quat_rotate_inverse, contact_wrench_rays, support_vector)
    for row in rows:
        row["effect_features"] = np.concatenate([row["state"], row["action"]])
        # Include action/context products so E+C can express that a proposed
        # action acts differently under an existing contact wrench.
        # The last 12 context entries are the CHORD support coordinates.  Use
        # those for the action/support interaction; the preceding entries are
        # raw body/object forces and remain available in the main context.
        interaction = np.outer(row["action"], row["context"][-12:]).reshape(-1)
        row["effect_contact_features"] = np.concatenate([
            row["state"], row["action"], row["context"], interaction,
        ])
    e_only = fit_ridge(rows, "effect_features")
    e_contact = fit_ridge(rows, "effect_contact_features")
    result = {
        "schema": "ref2dex.chord-action-ranking.v1",
        "status": "COMPLETED",
        "external_repo": str(args.external_root.resolve()),
        "motion_root": str(args.motion_root.resolve()),
        "motion_sha256": sha256(args.motion_root / "interaction_hand_inspire.pt"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "protocol": {
            "fit_frames": args.fit_frames,
            "held_frames": args.held_frames,
            "horizon": args.horizon,
            "candidate_count": int(task.num_envs),
            "target": "negative_future_object_reference_error",
            "e_only": "state+dof_velocity+object_state+candidate_action",
            "e_plus_contact": "e_only+current_body_force+object_force+CHORD_support+action_support_interaction",
            "normal_source": "current simulator net-force direction (approximate; not per-contact normal)",
            "official_weights_used": False,
        },
        "rows": len(rows),
        "fit_valid_ray_rows": int(sum(r["valid_rays"] > 0 for r in rows if r["split"] == "fit")),
        "held_valid_ray_rows": int(sum(r["valid_rays"] > 0 for r in rows if r["split"] == "held")),
        "e_only": rank_eval(rows, "effect_features", e_only),
        "e_plus_contact": rank_eval(rows, "effect_contact_features", e_contact),
    }
    result["decision"] = (
        "PROMISING" if result["e_plus_contact"]["mean_regret"] + 1e-9 < result["e_only"]["mean_regret"] * 0.8
        and result["e_plus_contact"]["top1_hit_rate"] >= result["e_only"]["top1_hit_rate"]
        else "UNPROMISING"
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
