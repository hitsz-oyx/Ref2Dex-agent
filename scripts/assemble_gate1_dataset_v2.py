#!/usr/bin/env python3
"""Assemble the corrected Gate 1 v2 history/consequence windows.

This is an offline data transformation.  It never bootstraps the value target:
``return_to_go`` is the exact discounted sum of recorded simulator rewards up
to the recorded terminal transition.  Samples are kept only when a complete
future window is available, so no padding is silently treated as data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


REQUIRED = {
    "state", "context", "previous_action", "action", "reward", "reward_components", "done",
    "episode_id", "step", "progress", "motion_id", "noise_std", "object_root",
    "hand_body_position", "hand_body_quaternion", "hand_force", "object_force",
}


def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., :3] *= -1
    return out


def _quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack((
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ), dim=-1)


def _quat_rotate(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(vector[..., :1])
    vector_quaternion = torch.cat((vector, zeros), dim=-1)
    return _quat_multiply(_quat_multiply(q, vector_quaternion), _quat_conjugate(q))[..., :3]


def _canonicalize_quaternion_sequence(q: torch.Tensor, time_dim: int = 0) -> torch.Tensor:
    """Make equivalent q/-q samples deterministic and continuous in time.

    The first sample is sign-fixed by its largest-magnitude component before
    continuity is propagated. This makes overlapping windows agree on the
    sign of an equivalent orientation instead of inheriting each window's raw
    quaternion sign.
    """
    out = _quat_normalize(q).movedim(time_dim, 0).contiguous()
    pivot = out[0].abs().argmax(dim=-1, keepdim=True)
    sign = torch.where(out[0].gather(-1, pivot) < 0, -1.0, 1.0)
    out[0] = out[0] * sign
    for step in range(1, out.shape[0]):
        dot = (out[step - 1] * out[step]).sum(dim=-1, keepdim=True)
        out[step] = out[step] * torch.where(dot < 0, -1.0, 1.0)
    return out.movedim(0, time_dim)


def _relative_quaternion(reference: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ref = _quat_normalize(reference)
    tar = _quat_normalize(target)
    return _quat_normalize(_quat_multiply(_quat_conjugate(ref), tar))


def _load_shards(run_dir: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, object], List[str]]:
    shards = sorted(run_dir.glob("transitions_*.pt"))
    if not shards:
        raise FileNotFoundError(f"no transition shards in {run_dir}")
    parts = []
    metadata: Dict[str, object] = {}
    shard_hashes = []
    timing_presence = []
    timing_values = []
    source_presence = []
    source_values = []
    for path in shards:
        raw = path.read_bytes()
        shard_hashes.append(hashlib.sha256(raw).hexdigest())
        item = torch.load(path, map_location="cpu")
        missing = REQUIRED.difference(item)
        if missing:
            raise ValueError(f"{path} missing required keys: {sorted(missing)}")
        timing_presence.append("physical_timing" in item)
        if "physical_timing" in item:
            timing_values.append(item["physical_timing"])
        source_presence.append("source_sha256" in item)
        if "source_sha256" in item:
            source_values.append(item["source_sha256"])
        for key in ("schema", "gamma", "control_dt", "source_sha256", "physical_timing",
                    "effect_definition", "interaction_definition"):
            if key in item:
                if key in metadata and metadata[key] != item[key]:
                    raise ValueError(f"metadata mismatch for {key} in {path}")
                metadata[key] = item[key]
        parts.append(item)
    if any(timing_presence) and not all(timing_presence):
        raise ValueError(
            f"physical_timing must be present in every shard or absent from every shard in {run_dir}"
        )
    if timing_values and any(value != timing_values[0] for value in timing_values[1:]):
        raise ValueError(f"physical_timing mismatch across shards in {run_dir}")
    if any(source_presence) and not all(source_presence):
        raise ValueError(
            f"source_sha256 must be present in every shard or absent from every shard in {run_dir}"
        )
    if not all(source_presence):
        raise ValueError(f"source_sha256 missing in {run_dir}; actor validation requires checkpoint identity")
    if not source_values or any(value != source_values[0] for value in source_values[1:]):
        raise ValueError(f"source_sha256 mismatch across shards in {run_dir}")
    if not isinstance(source_values[0], str) or not source_values[0]:
        raise ValueError(f"source_sha256 must be a non-empty string in {run_dir}")
    keys = sorted(REQUIRED)
    merged = {key: torch.cat([part[key] for part in parts], dim=0) for key in keys}
    return merged, metadata, shard_hashes


def _episode_summaries(run_dir: Path) -> Dict[int, Dict[str, object]]:
    result_path = run_dir / "results.json"
    if not result_path.exists():
        return {}
    data = json.loads(result_path.read_text())
    return {int(row["episode_id"]): row for row in data.get("per_episode", [])}


def _iter_episode_indices(merged: Dict[str, torch.Tensor]) -> Iterable[Tuple[int, torch.Tensor]]:
    episodes = merged["episode_id"].tolist()
    by_episode: Dict[int, List[int]] = {}
    for index, episode in enumerate(episodes):
        by_episode.setdefault(int(episode), []).append(index)
    for episode, indices in sorted(by_episode.items()):
        idx = torch.tensor(indices, dtype=torch.long)
        order = torch.argsort(merged["step"][idx], stable=True)
        idx = idx[order]
        steps = merged["step"][idx].tolist()
        if len(set(steps)) != len(steps):
            raise ValueError(f"duplicate step in episode {episode}")
        if steps != list(range(min(steps), max(steps) + 1)):
            raise ValueError(f"non-contiguous steps in episode {episode}: {steps[:5]}...{steps[-5:]}")
        yield episode, idx


def assemble(run_dirs: List[Path], horizon: int, history_length: int = 10,
             allow_legacy_timing_inference: bool = False) -> Tuple[Dict[str, object], Dict[str, object]]:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if history_length < 1:
        raise ValueError("history_length must be positive")
    sample_episode: List[int] = []
    sample_step: List[int] = []
    sample_motion: List[int] = []
    sample_noise: List[float] = []
    sample_run: List[int] = []
    sample_action: List[torch.Tensor] = []
    sample_state: List[torch.Tensor] = []
    sample_previous_action: List[torch.Tensor] = []
    sample_context: List[torch.Tensor] = []
    sample_history_state: List[torch.Tensor] = []
    sample_history_action: List[torch.Tensor] = []
    sample_history_context: List[torch.Tensor] = []
    sample_history_progress: List[torch.Tensor] = []
    sample_reward: List[torch.Tensor] = []
    sample_reward_components: List[torch.Tensor] = []
    sample_return: List[torch.Tensor] = []
    sample_effect: List[torch.Tensor] = []
    sample_interaction: List[torch.Tensor] = []
    sample_future_action: List[torch.Tensor] = []
    sample_mask: List[torch.Tensor] = []
    sample_done: List[bool] = []
    sample_aux: List[List[float]] = []
    sample_namespace: List[int] = []
    metadata: Dict[str, object] = {"horizon": horizon, "history_length": history_length, "runs": []}
    namespace_ids: Dict[str, int] = {}
    namespace_keys: List[str] = []

    run_infos = []
    for run_dir in run_dirs:
        merged, run_meta, shard_hashes = _load_shards(run_dir)
        run_infos.append((run_dir, merged, run_meta, shard_hashes))
    namespace_keys = sorted({str(run_meta["source_sha256"]) for _, _, run_meta, _ in run_infos})
    namespace_ids = {key: index for index, key in enumerate(namespace_keys)}

    for run_index, (run_dir, merged, run_meta, shard_hashes) in enumerate(run_infos):
        summaries = _episode_summaries(run_dir)
        gamma = float(run_meta.get("gamma", 0.99))
        control_dt = float(run_meta.get("control_dt", 1.0))
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"invalid gamma {gamma} in {run_dir}")
        if "effect_definition" not in run_meta or "interaction_definition" not in run_meta:
            raise ValueError(f"explicit E/I definitions missing in {run_dir}")
        timing_inferred = "physical_timing" not in run_meta
        if timing_inferred and not allow_legacy_timing_inference:
            raise ValueError(
                f"physical_timing missing in {run_dir}; pass "
                "allow_legacy_timing_inference=True only for audited legacy shards"
            )
        namespace_key = str(run_meta["source_sha256"])
        namespace_id = namespace_ids[namespace_key]
        metadata["runs"].append({
            "run_dir": str(run_dir),
            "shard_sha256": shard_hashes,
            "rows": int(merged["reward"].numel()),
            "gamma": gamma,
            "control_dt": control_dt,
            "physical_timing_inferred": timing_inferred,
            "source_namespace": namespace_key,
            **{k: run_meta[k] for k in ("schema", "source_sha256", "physical_timing", "effect_definition", "interaction_definition")
               if k in run_meta},
        })

        physical_timing = str(run_meta.get("physical_timing", "post_env_step_legacy"))
        for episode, idx in _iter_episode_indices(merged):
            n = idx.numel()
            if n <= horizon + history_length - 1:
                continue
            reward = merged["reward"][idx].float()
            returns = torch.zeros_like(reward)
            running = torch.tensor(0.0)
            for pos in range(n - 1, -1, -1):
                running = reward[pos] + gamma * running
                returns[pos] = running
            summary = summaries.get(episode, {})
            aux = [
                float(bool(summary.get("stable_success", False))),
                float(bool(summary.get("drop_after_success", False))),
                float(summary.get("max_hold_seconds", 0.0)),
                float(summary.get("mean_lift_meters", 0.0)),
                float(summary.get("contact_fraction", 0.0)),
            ]
            # Pre-step collectors align row t with (s_t, a_t).  Older shards
            # captured after env_step; shift their physical arrays back one
            # row so the same assembler contract remains valid.
            physical_idx = idx
            if physical_timing == "post_env_step_legacy":
                if n <= history_length:
                    continue
                physical_idx = idx.clone()
                physical_idx[1:] = idx[:-1]
                start_pos = history_length
            elif physical_timing == "pre_env_step":
                start_pos = history_length - 1
            else:
                raise ValueError(f"unknown physical_timing={physical_timing}")
            for pos in range(start_pos, n - horizon):
                current = idx[pos]
                history = idx[pos - history_length + 1:pos + 1]
                future = idx[pos + 1:pos + horizon + 1]
                current_physical = physical_idx[pos]
                future_physical = physical_idx[pos + 1:pos + horizon + 1]
                current_object = merged["object_root"][current_physical]
                future_object = merged["object_root"][future_physical]
                current_position = current_object[:3]
                current_quaternion = current_object[3:7].expand(horizon, -1)
                inverse_current_quaternion = _quat_conjugate(_quat_normalize(current_quaternion))
                effect_position = _quat_rotate(
                    inverse_current_quaternion,
                    future_object[:, :3] - current_position,
                )
                object_window = merged["object_root"][physical_idx[pos:pos + horizon + 1]]
                object_window = object_window.clone()
                object_window[:, 3:7] = _canonicalize_quaternion_sequence(object_window[:, 3:7])
                current_quaternion = object_window[0, 3:7].expand(horizon, -1)
                future_object = object_window[1:]
                effect_position = _quat_rotate(
                    _quat_conjugate(_quat_normalize(current_quaternion)),
                    future_object[:, :3] - object_window[0, :3],
                )
                effect_quaternion = _canonicalize_quaternion_sequence(
                    _relative_quaternion(current_quaternion, future_object[:, 3:7]),
                    time_dim=0,
                )
                object_linear_velocity = _quat_rotate(inverse_current_quaternion, future_object[:, 7:10])
                object_angular_velocity = _quat_rotate(inverse_current_quaternion, future_object[:, 10:13])
                effect = torch.cat((
                    effect_position,
                    effect_quaternion,
                    object_linear_velocity,
                    object_angular_velocity,
                ), dim=-1)

                full_window = physical_idx[pos:pos + horizon + 1]
                object_window = merged["object_root"][full_window]
                object_window_quaternion = _canonicalize_quaternion_sequence(object_window[:, 3:7])
                object_window_inverse = _quat_conjugate(object_window_quaternion)
                hand_position_window = merged["hand_body_position"][full_window]
                hand_quaternion_window = _canonicalize_quaternion_sequence(
                    merged["hand_body_quaternion"][full_window], time_dim=0)
                object_position_window = object_window[:, :3]
                frame_quaternion = object_window_inverse[:, None, :].expand(
                    -1, hand_position_window.shape[1], -1)
                relative_position_window = _quat_rotate(
                    frame_quaternion,
                    hand_position_window - object_position_window[:, None, :],
                )
                relative_quaternion_window = _relative_quaternion(
                    object_window_quaternion[:, None, :].expand(
                        -1, hand_quaternion_window.shape[1], -1), hand_quaternion_window)
                relative_quaternion_window = _canonicalize_quaternion_sequence(
                    relative_quaternion_window, time_dim=0)
                relative_hand_position = relative_position_window[1:]
                relative_hand_quaternion = relative_quaternion_window[1:]
                relative_hand_velocity = (
                    relative_position_window[1:] - relative_position_window[:-1]
                ) / control_dt
                hand_force_world = merged["hand_force"][physical_idx[pos + 1:pos + horizon + 1]]
                object_force_world = merged["object_force"][physical_idx[pos + 1:pos + horizon + 1]]
                future_object_inverse = object_window_inverse[1:]
                hand_force = _quat_rotate(
                    future_object_inverse[:, None, :].expand(-1, hand_force_world.shape[1], -1),
                    hand_force_world)
                object_force = _quat_rotate(future_object_inverse, object_force_world)
                hand_force_norm = hand_force.norm(dim=-1)
                object_force_norm = object_force.norm(dim=-1, keepdim=True)
                # Match the live reward/contact predicate in physical_value_live.
                hand_contact = (hand_force_norm > 0.1).float()
                object_contact = (object_force_norm > 0.1).float()
                interaction = torch.cat((
                    relative_hand_position.flatten(1),
                    relative_hand_quaternion.flatten(1),
                    relative_hand_velocity.flatten(1),
                    hand_force.flatten(1),
                    object_force,
                    hand_force_norm,
                    object_force_norm,
                    hand_contact,
                    object_contact,
                ), dim=-1)

                sample_episode.append(episode)
                sample_step.append(int(merged["step"][current]))
                sample_motion.append(int(merged["motion_id"][current]))
                sample_noise.append(float(merged["noise_std"][current]))
                sample_run.append(run_index)
                sample_namespace.append(namespace_id)
                sample_action.append(merged["action"][current].float())
                sample_state.append(merged["state"][current].float())
                sample_previous_action.append(merged["previous_action"][current].float())
                sample_context.append(merged["context"][current].float())
                sample_history_state.append(merged["state"][history].float())
                sample_history_action.append(merged["previous_action"][history].float())
                sample_history_context.append(merged["context"][history].float())
                sample_history_progress.append(merged["progress"][history].float().unsqueeze(-1))
                sample_reward.append(reward[pos])
                sample_reward_components.append(merged["reward_components"][current].float())
                sample_return.append(returns[pos])
                sample_effect.append(effect)
                sample_interaction.append(interaction)
                sample_future_action.append(merged["action"][future].float())
                sample_mask.append(torch.ones(horizon, dtype=torch.bool))
                sample_done.append(bool(merged["done"][current]))
                sample_aux.append(aux)

    if not sample_episode:
        raise ValueError("no complete future windows were assembled")
    dataset: Dict[str, object] = {
        "state": torch.stack(sample_state),
        "previous_action": torch.stack(sample_previous_action),
        "context": torch.stack(sample_context),
        "history_state": torch.stack(sample_history_state),
        "history_previous_action": torch.stack(sample_history_action),
        "history_context": torch.stack(sample_history_context),
        "history_progress": torch.stack(sample_history_progress),
        "action": torch.stack(sample_action),
        "reward": torch.stack(sample_reward),
        "reward_components": torch.stack(sample_reward_components),
        "return_to_go": torch.stack(sample_return),
        "effect": torch.stack(sample_effect),
        "interaction": torch.stack(sample_interaction),
        "future_action": torch.stack(sample_future_action),
        "future_valid_mask": torch.stack(sample_mask),
        "episode_id": torch.tensor(sample_episode, dtype=torch.long),
        "step": torch.tensor(sample_step, dtype=torch.long),
        "motion_id": torch.tensor(sample_motion, dtype=torch.long),
        "noise_std": torch.tensor(sample_noise, dtype=torch.float32),
        "source_run": torch.tensor(sample_run, dtype=torch.long),
        "source_namespace": torch.tensor(sample_namespace, dtype=torch.long),
        "done_at_decision": torch.tensor(sample_done, dtype=torch.bool),
        "episode_auxiliary": torch.tensor(sample_aux, dtype=torch.float32),
        "metadata": {
            **metadata,
            "effect_layout": "future decision-object-frame [relative_xyz, relative_quaternion_xyzw, linear_velocity_xyz, angular_velocity_xyz]",
            "interaction_layout": "future contemporaneous-object-frame [hand_relative_xyz, hand_relative_quaternion_xyzw, relative_velocity_xyz, hand_force_xyz, object_force_xyz, hand_force_norm, object_force_norm, hand_contact_mask, object_contact_mask] per contact body",
            "history_layout": "past contiguous [state, factual preceding action, context, progress]; episode-initial preceding action remains the collector value",
            "target_definition": "exact Monte Carlo return-to-go from recorded simulator reward; no bootstrap",
            "split_unit": "(source_namespace, source_run, episode_id)",
            "source_namespace_keys": namespace_keys,
            "future_action_layout": "on-policy actions at t+1:t+H; diagnostic control only",
            "assembly_impl": "reference_v2",
        },
    }
    report = {
        "runs": len(run_dirs),
        "samples": len(sample_episode),
        "episodes": len(set(zip(sample_run, sample_episode))),
        "horizon": horizon,
        "history_length": history_length,
        "effect_dim": int(dataset["effect"].shape[-1]),
        "interaction_dim": int(dataset["interaction"].shape[-1]),
        "target_mean": float(dataset["return_to_go"].mean()),
        "target_std": float(dataset["return_to_go"].std(unbiased=False)),
    }
    return dataset, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--allow-legacy-timing-inference", action="store_true",
                        help="allow missing physical_timing as audited legacy post-step data")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    dataset, report = assemble(args.input, args.horizon, args.history_length,
                               args.allow_legacy_timing_inference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, args.output)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
