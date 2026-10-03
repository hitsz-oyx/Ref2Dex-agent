#!/usr/bin/env python3
"""Assemble episode-grouped Gate 1 (E, I) -> G windows from physical rollouts.

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
    "episode_id", "step", "motion_id", "noise_std", "object_root",
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
    for path in shards:
        raw = path.read_bytes()
        shard_hashes.append(hashlib.sha256(raw).hexdigest())
        item = torch.load(path, map_location="cpu")
        missing = REQUIRED.difference(item)
        if missing:
            raise ValueError(f"{path} missing required keys: {sorted(missing)}")
        for key in ("schema", "gamma", "control_dt", "source_sha256",
                    "effect_definition", "interaction_definition"):
            if key in item:
                if key in metadata and metadata[key] != item[key]:
                    raise ValueError(f"metadata mismatch for {key} in {path}")
                metadata[key] = item[key]
        parts.append(item)
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


def assemble(run_dirs: List[Path], horizon: int) -> Tuple[Dict[str, object], Dict[str, object]]:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    all_rows: Dict[str, List[torch.Tensor]] = {}
    sample_episode: List[int] = []
    sample_step: List[int] = []
    sample_motion: List[int] = []
    sample_noise: List[float] = []
    sample_run: List[int] = []
    sample_action: List[torch.Tensor] = []
    sample_state: List[torch.Tensor] = []
    sample_previous_action: List[torch.Tensor] = []
    sample_context: List[torch.Tensor] = []
    sample_reward: List[torch.Tensor] = []
    sample_reward_components: List[torch.Tensor] = []
    sample_return: List[torch.Tensor] = []
    sample_effect: List[torch.Tensor] = []
    sample_interaction: List[torch.Tensor] = []
    sample_mask: List[torch.Tensor] = []
    sample_done: List[bool] = []
    sample_aux: List[List[float]] = []
    metadata: Dict[str, object] = {"horizon": horizon, "runs": []}

    for run_index, run_dir in enumerate(run_dirs):
        merged, run_meta, shard_hashes = _load_shards(run_dir)
        summaries = _episode_summaries(run_dir)
        gamma = float(run_meta.get("gamma", 0.99))
        control_dt = float(run_meta.get("control_dt", 1.0))
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"invalid gamma {gamma} in {run_dir}")
        if "effect_definition" not in run_meta or "interaction_definition" not in run_meta:
            raise ValueError(f"explicit E/I definitions missing in {run_dir}")
        metadata["runs"].append({
            "run_dir": str(run_dir),
            "shard_sha256": shard_hashes,
            "rows": int(merged["reward"].numel()),
            "gamma": gamma,
            "control_dt": control_dt,
            **{k: run_meta[k] for k in ("schema", "source_sha256", "effect_definition", "interaction_definition")
               if k in run_meta},
        })

        for episode, idx in _iter_episode_indices(merged):
            n = idx.numel()
            if n <= horizon:
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
            for pos in range(n - horizon):
                current = idx[pos]
                future = idx[pos + 1:pos + horizon + 1]
                current_object = merged["object_root"][current]
                future_object = merged["object_root"][future]
                current_position = current_object[:3]
                current_quaternion = current_object[3:7].expand(horizon, -1)
                inverse_current_quaternion = _quat_conjugate(_quat_normalize(current_quaternion))
                effect_position = _quat_rotate(
                    inverse_current_quaternion,
                    future_object[:, :3] - current_position,
                )
                effect_quaternion = _relative_quaternion(current_quaternion, future_object[:, 3:7])
                object_linear_velocity = _quat_rotate(inverse_current_quaternion, future_object[:, 7:10])
                object_angular_velocity = _quat_rotate(inverse_current_quaternion, future_object[:, 10:13])
                effect = torch.cat((
                    effect_position,
                    effect_quaternion,
                    object_linear_velocity,
                    object_angular_velocity,
                ), dim=-1)

                future_hand_position = merged["hand_body_position"][future]
                future_hand_quaternion = merged["hand_body_quaternion"][future]
                relative_hand_position = _quat_rotate(
                    inverse_current_quaternion.view(horizon, 1, 4).expand(-1, future_hand_position.shape[1], -1),
                    future_hand_position - current_position.view(1, 1, 3),
                )
                object_quaternion = current_quaternion.view(horizon, 1, 4).expand(-1, future_hand_quaternion.shape[1], -1)
                relative_hand_quaternion = _relative_quaternion(object_quaternion, future_hand_quaternion)
                relative_hand_velocity = torch.empty_like(relative_hand_position)
                current_hand_position = _quat_rotate(
                    _quat_conjugate(_quat_normalize(current_object[3:7])).view(1, 4).expand(future_hand_position.shape[1], -1),
                    merged["hand_body_position"][current] - current_position.view(1, 3),
                )
                relative_hand_velocity[0] = (relative_hand_position[0] - current_hand_position) / control_dt
                if horizon > 1:
                    relative_hand_velocity[1:] = (
                        relative_hand_position[1:] - relative_hand_position[:-1]
                    ) / control_dt
                hand_force = merged["hand_force"][future]
                object_force = merged["object_force"][future]
                interaction = torch.cat((
                    relative_hand_position.flatten(1),
                    relative_hand_quaternion.flatten(1),
                    relative_hand_velocity.flatten(1),
                    hand_force.flatten(1),
                    object_force,
                ), dim=-1)

                sample_episode.append(episode)
                sample_step.append(int(merged["step"][current]))
                sample_motion.append(int(merged["motion_id"][current]))
                sample_noise.append(float(merged["noise_std"][current]))
                sample_run.append(run_index)
                sample_action.append(merged["action"][current].float())
                sample_state.append(merged["state"][current].float())
                sample_previous_action.append(merged["previous_action"][current].float())
                sample_context.append(merged["context"][current].float())
                sample_reward.append(reward[pos])
                sample_reward_components.append(merged["reward_components"][current].float())
                sample_return.append(returns[pos])
                sample_effect.append(effect)
                sample_interaction.append(interaction)
                sample_mask.append(torch.ones(horizon, dtype=torch.bool))
                sample_done.append(bool(merged["done"][current]))
                sample_aux.append(aux)

    if not sample_episode:
        raise ValueError("no complete future windows were assembled")
    dataset: Dict[str, object] = {
        "state": torch.stack(sample_state),
        "previous_action": torch.stack(sample_previous_action),
        "context": torch.stack(sample_context),
        "action": torch.stack(sample_action),
        "reward": torch.stack(sample_reward),
        "reward_components": torch.stack(sample_reward_components),
        "return_to_go": torch.stack(sample_return),
        "effect": torch.stack(sample_effect),
        "interaction": torch.stack(sample_interaction),
        "future_valid_mask": torch.stack(sample_mask),
        "episode_id": torch.tensor(sample_episode, dtype=torch.long),
        "step": torch.tensor(sample_step, dtype=torch.long),
        "motion_id": torch.tensor(sample_motion, dtype=torch.long),
        "noise_std": torch.tensor(sample_noise, dtype=torch.float32),
        "source_run": torch.tensor(sample_run, dtype=torch.long),
        "done_at_decision": torch.tensor(sample_done, dtype=torch.bool),
        "episode_auxiliary": torch.tensor(sample_aux, dtype=torch.float32),
        "metadata": {
            **metadata,
            "effect_layout": "future object-frame [relative_xyz, relative_quaternion_xyzw, linear_velocity_xyz, angular_velocity_xyz]",
            "interaction_layout": "future object-frame [hand_relative_xyz, hand_relative_quaternion_xyzw, hand_relative_linear_velocity_xyz, hand_force, object_force] per contact body",
            "target_definition": "exact Monte Carlo return-to-go from recorded simulator reward; no bootstrap",
            "split_unit": "episode_id",
        },
    }
    report = {
        "runs": len(run_dirs),
        "samples": len(sample_episode),
        "episodes": len(set(sample_episode)),
        "horizon": horizon,
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
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    dataset, report = assemble(args.input, args.horizon)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, args.output)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
