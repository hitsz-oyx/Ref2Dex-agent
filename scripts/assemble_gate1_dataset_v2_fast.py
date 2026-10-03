#!/usr/bin/env python3
"""Vectorized v2 assembler; same contract as assemble_gate1_dataset_v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from assemble_gate1_dataset_v2 import (
    _episode_summaries,
    _iter_episode_indices,
    _load_shards,
    _quat_conjugate,
    _canonicalize_quaternion_sequence,
    _quat_normalize,
    _quat_rotate,
    _relative_quaternion,
)


def assemble(run_dirs: List[Path], horizon: int, history_length: int = 10):
    if horizon < 1 or history_length < 1:
        raise ValueError("horizon and history_length must be positive")
    chunks: Dict[str, List[torch.Tensor]] = {key: [] for key in (
        "state", "previous_action", "context", "history_state", "history_previous_action",
        "history_context", "history_progress", "action", "reward", "reward_components",
        "return_to_go", "effect", "interaction", "future_valid_mask", "episode_id",
        "future_action",
        "step", "motion_id", "noise_std", "source_run", "done_at_decision",
        "episode_auxiliary")}
    metadata: Dict[str, object] = {"horizon": horizon, "history_length": history_length, "runs": []}
    for run_index, run_dir in enumerate(run_dirs):
        merged, run_meta, shard_hashes = _load_shards(run_dir)
        summaries = _episode_summaries(run_dir)
        gamma = float(run_meta.get("gamma", 0.99))
        control_dt = float(run_meta.get("control_dt", 1.0))
        if "effect_definition" not in run_meta or "interaction_definition" not in run_meta:
            raise ValueError(f"explicit E/I definitions missing in {run_dir}")
        metadata["runs"].append({
            "run_dir": str(run_dir), "shard_sha256": shard_hashes,
            "rows": int(merged["reward"].numel()), "gamma": gamma, "control_dt": control_dt,
            **{k: run_meta[k] for k in ("schema", "source_sha256", "physical_timing", "effect_definition", "interaction_definition") if k in run_meta},
        })
        physical_timing = str(run_meta.get("physical_timing", "post_env_step_legacy"))
        for episode, idx in _iter_episode_indices(merged):
            n = int(idx.numel())
            count = n - horizon - history_length + 1
            if count <= 0:
                continue
            if physical_timing == "pre_env_step":
                start = history_length - 1
            elif physical_timing == "post_env_step_legacy":
                start = history_length
            else:
                raise ValueError(f"unknown physical_timing={physical_timing}")
            starts = torch.arange(start, n - horizon, dtype=torch.long)
            rows = int(starts.numel())
            if rows <= 0:
                continue
            history_pos = starts[:, None] - history_length + 1 + torch.arange(history_length)[None, :]
            future_pos = starts[:, None] + 1 + torch.arange(horizon)[None, :]
            full_pos = starts[:, None] + torch.arange(horizon + 1)[None, :]
            physical_idx = idx
            if physical_timing == "post_env_step_legacy":
                physical_idx = idx.clone()
                physical_idx[1:] = idx[:-1]
            current_pos = idx[starts]
            history_idx = idx[history_pos]
            future_idx = physical_idx[future_pos]
            # Actions remain decision-row indexed even for legacy shards;
            # only the measured physical tensors receive the post-step shift.
            future_action_idx = idx[future_pos]
            full_idx = physical_idx[full_pos]
            reward = merged["reward"][idx].float()
            returns = torch.zeros_like(reward)
            running = torch.tensor(0.0)
            for pos in range(n - 1, -1, -1):
                running = reward[pos] + gamma * running
                returns[pos] = running
            current_object = merged["object_root"][physical_idx[starts]]
            future_object = merged["object_root"][future_idx]
            object_window_raw = merged["object_root"][full_idx]
            object_window_q = _canonicalize_quaternion_sequence(object_window_raw[..., 3:7], time_dim=1)
            current_object = object_window_raw[:, 0].clone()
            current_object[:, 3:7] = object_window_q[:, 0]
            future_object = object_window_raw[:, 1:].clone()
            future_object[:, :, 3:7] = object_window_q[:, 1:]
            current_q = _quat_normalize(current_object[:, 3:7])
            inverse_current_q = _quat_conjugate(current_q)
            inverse_current_q_h = inverse_current_q[:, None, :].expand(-1, horizon, -1)
            effect = torch.cat((
                _quat_rotate(inverse_current_q_h, future_object[..., :3] - current_object[:, None, :3]),
                _canonicalize_quaternion_sequence(
                    _relative_quaternion(current_q[:, None, :].expand(-1, horizon, -1), future_object[..., 3:7]),
                    time_dim=1,
                ),
                _quat_rotate(inverse_current_q_h, future_object[..., 7:10]),
                _quat_rotate(inverse_current_q_h, future_object[..., 10:13]),
            ), dim=-1)
            object_window = merged["object_root"][full_idx]
            object_q = object_window_q
            object_inv = _quat_conjugate(object_q)
            hand_position = merged["hand_body_position"][full_idx]
            hand_quaternion = _canonicalize_quaternion_sequence(
                merged["hand_body_quaternion"][full_idx], time_dim=1)
            frame_q = object_inv[:, :, None, :].expand(-1, -1, hand_position.shape[2], -1)
            relative_position = _quat_rotate(frame_q, hand_position - object_window[..., None, :3])
            relative_quaternion = _relative_quaternion(
                object_q[:, :, None, :].expand(-1, -1, hand_quaternion.shape[2], -1),
                hand_quaternion)
            relative_quaternion = _canonicalize_quaternion_sequence(relative_quaternion, time_dim=1)
            relative_velocity = (relative_position[:, 1:] - relative_position[:, :-1]) / control_dt
            hand_force_world = merged["hand_force"][future_idx]
            object_force_world = merged["object_force"][future_idx]
            future_object_inv = object_inv[:, 1:]
            hand_force = _quat_rotate(
                future_object_inv[:, :, None, :].expand(-1, -1, hand_force_world.shape[2], -1),
                hand_force_world)
            object_force = _quat_rotate(future_object_inv, object_force_world)
            hand_force_norm = hand_force.norm(dim=-1)
            object_force_norm = object_force.norm(dim=-1, keepdim=True)
            interaction = torch.cat((
                relative_position[:, 1:].flatten(2), relative_quaternion[:, 1:].flatten(2),
                relative_velocity.flatten(2), hand_force.flatten(2), object_force,
                hand_force_norm, object_force_norm,
                (hand_force_norm > 0.1).float(), (object_force_norm > 0.1).float()), dim=-1)
            summary = summaries.get(episode, {})
            aux = torch.tensor([
                float(bool(summary.get("stable_success", False))),
                float(bool(summary.get("drop_after_success", False))),
                float(summary.get("max_hold_seconds", 0.0)),
                float(summary.get("mean_lift_meters", 0.0)),
                float(summary.get("contact_fraction", 0.0)),
            ], dtype=torch.float32).expand(rows, -1)
            chunks["state"].append(merged["state"][current_pos].float())
            chunks["previous_action"].append(merged["previous_action"][current_pos].float())
            chunks["context"].append(merged["context"][current_pos].float())
            chunks["history_state"].append(merged["state"][history_idx].float())
            chunks["history_previous_action"].append(
                merged["previous_action"][history_idx].float())
            chunks["history_context"].append(merged["context"][history_idx].float())
            chunks["history_progress"].append(merged["progress"][history_idx].float().unsqueeze(-1))
            chunks["action"].append(merged["action"][current_pos].float())
            chunks["reward"].append(reward[starts])
            chunks["reward_components"].append(merged["reward_components"][current_pos].float())
            chunks["return_to_go"].append(returns[starts])
            chunks["effect"].append(effect.float())
            chunks["interaction"].append(interaction.float())
            chunks["future_action"].append(merged["action"][future_action_idx].float())
            chunks["future_valid_mask"].append(torch.ones(rows, horizon, dtype=torch.bool))
            chunks["episode_id"].append(torch.full((rows,), episode, dtype=torch.long))
            chunks["step"].append(merged["step"][current_pos].long())
            chunks["motion_id"].append(merged["motion_id"][current_pos].long())
            chunks["noise_std"].append(merged["noise_std"][current_pos].float())
            chunks["source_run"].append(torch.full((rows,), run_index, dtype=torch.long))
            chunks["done_at_decision"].append(merged["done"][current_pos].bool())
            chunks["episode_auxiliary"].append(aux)
    if not chunks["episode_id"]:
        raise ValueError("no complete v2 windows")
    dataset = {key: torch.cat(values, dim=0) for key, values in chunks.items()}
    dataset["metadata"] = {
        **metadata,
        "effect_layout": "future decision-object-frame [relative_xyz, relative_quaternion_xyzw, linear_velocity_xyz, angular_velocity_xyz]",
        "interaction_layout": "future contemporaneous-object-frame [hand_relative_xyz, hand_relative_quaternion_xyzw, relative_velocity_xyz, hand_force_xyz, object_force_xyz, hand_force_norm, object_force_norm, hand_contact_mask, object_contact_mask] per contact body",
        "history_layout": "past contiguous [state, factual preceding action, context, progress]; episode-initial preceding action remains the collector value",
        "target_definition": "exact Monte Carlo return-to-go from recorded simulator reward; no bootstrap",
        "split_unit": "(source_run, episode_id)",
        "future_action_layout": "on-policy actions at t+1:t+H; diagnostic control only",
        "assembly_impl": "vectorized_v2",
    }
    report = {"runs": len(run_dirs), "samples": int(dataset["episode_id"].numel()),
              "episodes": len(set(zip(dataset["source_run"].tolist(), dataset["episode_id"].tolist()))), "horizon": horizon,
              "history_length": history_length, "effect_dim": int(dataset["effect"].shape[-1]),
              "interaction_dim": int(dataset["interaction"].shape[-1]),
              "target_mean": float(dataset["return_to_go"].mean()),
              "target_std": float(dataset["return_to_go"].std(unbiased=False))}
    return dataset, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--history-length", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset, report = assemble(args.input, args.horizon, args.history_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
