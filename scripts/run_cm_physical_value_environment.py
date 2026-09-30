#!/usr/bin/env python3
"""Collect complete physical transitions or evaluate actor-only stable grasp."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/DExplore/dexplore"))
sys.path.insert(0, str(ROOT))
from isaacgym import gymapi  # before torch
import torch
import evaluate as original
from src.task.CmResidual.physical_value_contract import SCHEMA, HoldTracker, validate_rows
from src.task.CmResidual.physical_value_live import snapshot, context, contacts, shared_reward
from src.task.CmResidual.dexplore_cm_geometry import DExploreCmv2GeometryBridge
from src.task.CmResidual.dexplore_approach import ApproachConfig, sampled_surface_gap

ARGS = None
SOURCE_SHA = "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PhysicalPlayer(original.EvalPlayer):
    @torch.no_grad()
    def run(self):
        task = self.env.task
        task._enable_early_termination = False
        task._adaptive_kappa_enabled = False
        task._hybrid_init_prob = .5 if ARGS.mode == "collect" else 1.0
        if abs(task.dt - 1 / 30) > 1e-8:
            raise ValueError("control interval drift")
        tracker = HoldTracker(task.num_envs, self.device)
        asset = ROOT / "third_party/DExplore/dexplore/data/assets"
        bridge = (DExploreCmv2GeometryBridge(hand_urdf=asset / "inspire_hand_new/inspire_hand_right.urdf",
                                            object_urdf=asset / "mjcf/airplane.urdf", device=task._dof_pos.device)
                  if ARGS.mode == "collect" else None)
        approach = ApproachConfig()
        noise_rng = torch.Generator(device=self.device).manual_seed(ARGS.assignment_seed)
        assignment_rng = torch.Generator(device=self.device).manual_seed(ARGS.assignment_seed + 1)
        num = task.num_envs
        obs = self.env_reset([])
        if self.get_batch_size(obs["obs"], 1) != num:
            raise ValueError("player batch size differs from simulator")
        if self.is_rnn:
            self.init_rnn()
        ids = torch.arange(num, device=self.device)
        tracker.reset(ids, task._target_states[:, 2])
        start_frame = task.start_times.clone()
        motion = task.data_id.clone()
        episode = ids.clone() + ARGS.seed_namespace * 10000000
        next_episode = int(episode.max()) + 1
        step = torch.zeros(num, dtype=torch.long, device=self.device)
        noise = torch.tensor((0., .05, .10, .20), device=self.device)[torch.randint(4, (num,), generator=assignment_rng, device=self.device)]
        finished = torch.zeros(num, dtype=torch.bool, device=self.device)
        ending = False
        completed_episodes = []
        parts = {}
        shard_paths = []
        rows = 0
        shard_rows = 0
        previous_action = torch.zeros(num, 18, device=self.device)
        started = time.monotonic()
        done_indices = []
        gamma = float(self.config.get("gamma", .99))

        def gap():
            geometry = bridge.current(task._dof_pos, task._target_states)
            return sampled_surface_gap(geometry.hand_points, geometry.object_points, approach)

        def flush():
            nonlocal parts, shard_rows
            if not parts:
                return
            payload = {key: torch.cat(value) for key, value in parts.items()}
            payload.update(schema=SCHEMA, gamma=gamma, control_dt=task.dt,
                           motion_names=list(task.motion_file), source_sha256=ARGS.checkpoint_sha256)
            validate_rows(payload)
            path = ARGS.run_dir / ("transitions_%03d.pt" % len(shard_paths))
            torch.save(payload, path)
            shard_paths.append(dict(path=str(path), sha256=sha(path), rows=len(payload["state"])))
            parts, shard_rows = {}, 0

        for tick in range(ARGS.max_steps):
            if time.monotonic() - started > ARGS.wall_seconds:
                raise TimeoutError("environment phase wall budget")
            # DExplore's reset adapter also packages obs/AMP in a dictionary
            # on non-reset steps; env_step itself returns a bare tensor.
            obs = self.env_reset(done_indices)
            if done_indices:
                # Player reset consumes only completed environments, before the
                # next state snapshot; terminal snapshots above are preserved.
                reset_ids = torch.tensor(done_indices, device=self.device)
                tracker.reset(reset_ids, task._target_states[reset_ids, 2])
                start_frame[reset_ids] = task.start_times[reset_ids]
                motion[reset_ids] = task.data_id[reset_ids]
                episode[reset_ids] = torch.arange(next_episode, next_episode + len(reset_ids), device=self.device)
                next_episode += len(reset_ids)
                step[reset_ids] = 0
                previous_action[reset_ids] = 0
                noise[reset_ids] = torch.tensor((0., .05, .10, .20), device=self.device)[torch.randint(4, (len(reset_ids),), generator=assignment_rng, device=self.device)]
            before = snapshot(task, tracker)
            ctx = context(task, tracker)
            phase = task.progress_buf.clone()
            active = ~finished
            action = self.get_action(obs, ARGS.mode == "evaluate")
            if ARGS.mode == "collect":
                action = action + noise[:, None] * torch.randn(action.shape, generator=noise_rng, device=self.device)
            action = action.clamp(-1, 1)
            gap_before = gap() if bridge is not None else None
            obs, base_reward, done, info = self.env_step(self.env, action)
            done = done.to(self.device).bool().reshape(-1)
            terminate = info["terminate"].to(self.device).bool().reshape(-1)
            if bridge is not None:
                reward, components = shared_reward(task, tracker, base_reward.to(self.device),
                                                   before[:, 38], gap_before, gap(), gamma, approach)
            else:
                stable, _ = tracker.step(task._target_states[:, 2], contacts(task).bool().all(-1))
                reward = base_reward.to(self.device).reshape(-1) + stable
            after = snapshot(task, tracker)
            next_ctx = context(task, tracker)
            if ARGS.mode == "collect":
                values = dict(state=before, next_state=after, context=ctx, next_context=next_ctx,
                              action=action, previous_action=previous_action.clone(), reward=reward,
                              reward_components=components, done=done, terminate=terminate,
                              timeout=done & ~terminate, episode_id=episode.clone(), env_id=ids,
                              step=step.clone(), progress=phase, start_frame=start_frame.clone(),
                              motion_id=motion.clone(), noise_std=noise.clone())
                for key, value in values.items():
                    parts.setdefault(key, []).append(value[active].detach().cpu())
                count = int(active.sum())
                rows += count
                shard_rows += count
                if shard_rows >= 32768:
                    flush()
            step += 1
            previous_action = action.clone()
            terminal_ids = (done & active).nonzero(as_tuple=False).reshape(-1)
            for env_id in terminal_ids.tolist():
                completed_episodes.append(dict(env_id=env_id, episode_id=int(episode[env_id]),
                    motion_id=int(motion[env_id]), start_frame=int(start_frame[env_id]),
                    steps=int(step[env_id]), stable_success=bool(tracker.stable[env_id]),
                    drop_after_success=bool(tracker.drop_after_success[env_id]),
                    max_hold_seconds=float(tracker.max_run[env_id]),
                    initial_object_height=float(tracker.initial_height[env_id]),
                    control_dt=float(task.dt), terminate=bool(terminate[env_id])))
            if ARGS.mode == "evaluate":
                finished |= done
            else:
                if rows >= ARGS.rows:
                    ending = True
                if ending:
                    finished |= done
            if finished.all():
                break
            done_indices = done.nonzero(as_tuple=False).reshape(-1).tolist()
            if tick % 250 == 0:
                print(json.dumps(dict(tick=tick, rows=rows, complete_episodes=len(completed_episodes))), flush=True)
        else:
            raise RuntimeError("max steps before completing environment phase")
        flush()
        if ARGS.mode == "evaluate" and len(completed_episodes) != num:
            raise ValueError("incomplete first-episode evaluation")
        result = dict(schema=SCHEMA, mode=ARGS.mode, run_status="COMPLETED", rows=rows,
                      complete_episodes=len(completed_episodes), shards=shard_paths,
                      per_episode=completed_episodes,
                      stable_success_count=sum(row["stable_success"] for row in completed_episodes),
                      drop_after_success_count=sum(row["drop_after_success"] for row in completed_episodes),
                      elapsed_seconds=time.monotonic() - started)
        (ARGS.run_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    global ARGS
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--mode", choices=("collect", "evaluate"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--rows", type=int, default=500000)
    parser.add_argument("--assignment-seed", type=int, default=20260930283)
    parser.add_argument("--seed-namespace", type=int, default=283)
    parser.add_argument("--wall-seconds", type=int, default=2700)
    parser.add_argument("--max-steps", type=int, default=30000)
    ARGS, remaining = parser.parse_known_args()
    source = Path(remaining[remaining.index("--checkpoint") + 1])
    if sha(source) != ARGS.checkpoint_sha256:
        raise ValueError("checkpoint hash drift")
    if ARGS.mode == "collect" and ARGS.checkpoint_sha256 != SOURCE_SHA:
        raise ValueError("collection must use pinned self-trained source")
    if ARGS.run_dir.exists():
        raise FileExistsError(ARGS.run_dir)
    ARGS.run_dir.mkdir(parents=True)
    manifest = dict(run_status="STARTED", command=sys.argv, input_sha256=ARGS.checkpoint_sha256,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    created_at=datetime.now(timezone.utc).isoformat(),
                    physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    wall_seconds=ARGS.wall_seconds, mode=ARGS.mode)
    (ARGS.run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sys.argv = [sys.argv[0], *remaining]
    original.EvalPlayer = PhysicalPlayer
    try:
        original.main()
        manifest["run_status"] = "COMPLETED"
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        (ARGS.run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
