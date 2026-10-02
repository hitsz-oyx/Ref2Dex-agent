#!/usr/bin/env python3
"""Collect real native short-consequence data around frozen rotation-Cup control."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from collect_contact_consequences import build_player
from run_paired_evaluator_resolution import sha

ASSETS = ROOT / "third_party/DExplore/dexplore/data/assets/mjcf"
CODEBOOK = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0))


def make_player(original, args, torch, gymtorch):
    from src.task.CmResidual.cm_residual_policy import (
        RESIDUAL_SCALE_M, apply_residual_action,
    )
    from src.task.CmResidual.executable_contact_options import hold_target, obj_vertices, TableClearance
    from src.task.CmResidual.orientation_anchored_options import orientation_anchored_action
    from src.task.CmResidual.paired_evaluation import fingerprint
    from src.task.CmResidual.weight_normalized_contact import weight_normalized_contacts

    parent = build_player(original, args, torch, gymtorch)

    class ResidualSourcePlayer(parent):
        @torch.no_grad()
        def run(self):
            begin = time.monotonic()
            torch.set_num_threads(2)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            task = self.env.task
            task._hybrid_init_prob = 0.0
            task._enable_early_termination = False
            task._adaptive_kappa_enabled = False
            n, windows = task.num_envs, 2 * args.windows_per_stratum
            if set(getattr(task, "object_name", ["airplane"])) != {"airplane"}:
                raise ValueError("airplane-only residual source")
            if n != 96 or self.is_rnn or abs(task.dt - 1 / 30) > 1e-8:
                raise ValueError("native residual source contract")
            if not 1 <= args.windows_per_stratum <= 4:
                raise ValueError("bounded windows per stratum")
            geometry = TableClearance(
                obj_vertices(ASSETS / "objects/airplane/airplane.obj", self.device),
                obj_vertices(ASSETS / "objects/table/table.obj", self.device),
                getattr(task, "ball_size", 1.0),
            )
            ids = torch.arange(n, device=self.device)
            observation = self.env_reset(ids)
            if self.get_batch_size(observation["obs"], 1) != n or observation["obs"].shape[-1] != 1442:
                raise ValueError("native observation shape contract")
            motion, start = task.data_id.clone(), task.start_times.clone()
            rest = task.hoi_refs[task.data_id, task.ref_index, 0, 108].clone()
            mass = torch.tensor([
                task.gym.get_actor_rigid_body_properties(env, handle)[0].mass
                for env, handle in zip(task.envs, task._target_handles)
            ], device=self.device)
            gravity = abs(task.sim_params.gravity.z)
            body_ids = task._contact_body_ids

            def contacts():
                return weight_normalized_contacts(
                    task._contact_forces[:, body_ids], task._tar_contact_forces, mass, gravity)

            expert_before = fingerprint([
                dict(model=model.state_dict(), rms=rms.state_dict())
                for model, rms in self.frozen_experts
            ])
            zeros = lambda *shape, dtype=torch.float32: torch.zeros(
                n, windows, *shape, dtype=dtype, device=self.device)
            history = torch.zeros(n, 10, 69, device=self.device)
            previous = torch.zeros(n, 18, device=self.device)
            count = torch.zeros(n, dtype=torch.long, device=self.device)
            elapsed = torch.full_like(count, -1)
            cooldown = torch.zeros_like(count)
            contact_run = torch.zeros_like(count)
            ended = torch.zeros(n, dtype=torch.bool, device=self.device)
            stratum_count = torch.zeros(n, 2, dtype=torch.long, device=self.device)
            anchor = torch.zeros(n, 18, device=self.device)
            residual = torch.zeros(n, 3, device=self.device)
            trigger = torch.full((n, windows), -1, dtype=torch.long, device=self.device)
            assignment = torch.full_like(trigger, -1)
            steps = torch.zeros_like(trigger)
            pre = zeros(49)
            hist = zeros(10, 69)
            window_residual = zeros(3)
            initial_clearance = zeros()
            future_state = zeros(10, 49)
            future_contact = zeros(10, 2, dtype=torch.bool)
            future_clearance = zeros(10)
            future_done = zeros(10, dtype=torch.bool)
            initial_hand = zeros(len(body_ids), 3)
            initial_object = zeros(3)
            native_obs = zeros(10, observation["obs"].shape[-1])
            baseline_action = zeros(10, 18)
            actual_action = zeros(10, 18)
            baseline_pd = zeros(10, 18)
            actual_pd = zeros(10, 18)
            requested_world = zeros(10, 3)
            applied_world = zeros(10, 3)
            saturated = zeros(10, dtype=torch.bool)
            raw_hand = zeros(10, len(body_ids), 3)
            raw_object = zeros(10, 3)
            force_ratio = zeros(10, 2)
            allocation_generator = torch.Generator(device=self.device).manual_seed(args.assignment_seed)
            reset = ids[:0]
            codebook = torch.tensor(CODEBOOK, device=self.device, dtype=torch.float32)
            residual_scale = torch.tensor(RESIDUAL_SCALE_M, device=self.device)

            def pd_targets(action, rows):
                """Mirror the native target conversion for a subset of environments."""
                normalized = action.clone()
                normalized[..., 6:] = (1 + action[..., 6:]) / 2
                pd = task._pd_action_offset + task._pd_action_scale * normalized
                pd[..., :6] = pd[..., :6] + task._dof_pos[rows, :6]
                pd[..., 7] = pd[..., 6] * 1.05
                pd[..., 9] = pd[..., 8] * 1.05
                pd[..., 11] = pd[..., 10] * 1.05
                pd[..., 13] = pd[..., 12] * 1.05
                pd[..., 16] = pd[..., 15] * 0.6
                pd[..., 17] = pd[..., 15] * 0.8
                return pd

            for tick in range(args.max_steps):
                if time.monotonic() - begin > args.wall_seconds:
                    raise TimeoutError("bounded residual source")
                observation = self.env_reset(reset)
                state = torch.cat((task._dof_pos.clone(), task._dof_vel.clone(), task._target_states.clone()), -1)
                contact, ratio = contacts()
                history = torch.cat((history[:, 1:], torch.cat((state, contact.float(), previous), -1)[:, None]), 1)
                contact_run = torch.where(contact.all(-1), contact_run + 1, torch.zeros_like(contact_run))
                bank = self.candidates(observation)
                clearance = geometry.clearance(task._target_states, task._table_states)
                already_clear = (state[:, 38] - rest >= .03) & (clearance >= .002)
                strata = already_clear.long()
                eligible = ((~ended) & (elapsed < 0) & (cooldown <= 0) & (contact_run >= 3)
                            & (state[:, 38] - rest >= .005) & (tick >= 10)
                            & (tick <= args.max_steps - 12)
                            & (task.max_episode_length[task.data_id] - task.progress_buf > 11))
                eligible &= stratum_count[ids, strata] < args.windows_per_stratum
                rows = eligible.nonzero().flatten()
                if len(rows):
                    draw = torch.randint(0, len(CODEBOOK), (len(rows),), device=self.device,
                                         generator=allocation_generator)
                    anchor[rows] = hold_target(task._dof_pos[rows], task._pd_action_offset, task._pd_action_scale)
                    residual[rows] = codebook[draw] * residual_scale
                    assignment[rows, count[rows]] = draw
                    window_residual[rows, count[rows]] = residual[rows]
                    trigger[rows, count[rows]] = tick
                    pre[rows, count[rows]] = state[rows]
                    hist[rows, count[rows]] = history[rows]
                    initial_clearance[rows, count[rows]] = clearance[rows]
                    initial_hand[rows, count[rows]] = task._contact_forces[rows][:, body_ids]
                    initial_object[rows, count[rows]] = task._tar_contact_forces[rows]
                    stratum_count[rows, strata[rows]] += 1
                    elapsed[rows] = 0
                live = (elapsed >= 0) & ~ended
                rows = live.nonzero().flatten()
                slot, offset = count[rows], elapsed[rows]
                action = bank[:, 4].clone()
                if len(rows):
                    baseline = orientation_anchored_action(
                        bank[rows, 1], anchor[rows], task._dof_pos[rows],
                        task._pd_action_offset, task._pd_action_scale)
                    current_residual = torch.where(offset[:, None] < 2, residual[rows], torch.zeros_like(residual[rows]))
                    modified, detail = apply_residual_action(baseline, state[rows, 39:43], current_residual)
                    action[rows] = modified
                    baseline_action[rows, slot, offset] = baseline
                    actual_action[rows, slot, offset] = modified
                    requested_world[rows, slot, offset] = detail["requested_delta_world"]
                    applied_world[rows, slot, offset] = detail["applied_delta_world"]
                    saturated[rows, slot, offset] = detail["saturated"]
                    baseline_pd[rows, slot, offset] = pd_targets(baseline.clone(), rows)
                    actual_pd[rows, slot, offset] = pd_targets(modified.clone(), rows)
                    native_obs[rows, slot, offset] = observation["obs"][rows]
                if len(rows) and (saturated[rows, slot, offset] & (offset < 2)).any():
                    raise ValueError("source residual saturated native translation")
                if len(rows):
                    if ((actual_pd[rows, slot, offset, 3:] - baseline_pd[rows, slot, offset, 3:]).abs() > 2e-5).any():
                        raise ValueError("residual changed non-translation native target")
                applied = action.clone()
                _, _, done, _ = self.env_step(self.env, action)
                done = done.bool().reshape(-1)
                after = torch.cat((task._dof_pos.clone(), task._dof_vel.clone(), task._target_states.clone()), -1)
                post_contact, post_ratio = contacts()
                if len(rows):
                    future_state[rows, slot, offset] = after[rows]
                    future_contact[rows, slot, offset] = post_contact[rows]
                    future_clearance[rows, slot, offset] = geometry.clearance(task._target_states, task._table_states)[rows]
                    future_done[rows, slot, offset] = done[rows]
                    raw_hand[rows, slot, offset] = task._contact_forces[rows][:, body_ids]
                    raw_object[rows, slot, offset] = task._tar_contact_forces[rows]
                    force_ratio[rows, slot, offset] = post_ratio[rows]
                if len(rows):
                    steps[rows, slot] += 1
                elapsed[live] += 1
                complete = live & (elapsed == 10)
                count[complete] += 1
                elapsed[complete] = -1
                cooldown[complete] = 6
                cooldown = (cooldown - 1).clamp_min(0)
                ended |= done
                reset = done.nonzero().flatten()
                previous = applied
                if tick % 100 == 0:
                    print(json.dumps(dict(tick=tick, windows=int((steps == 10).sum()), active=int(live.sum()))), flush=True)
                if (ended | (count >= windows)).all():
                    break

            valid = steps == 10
            # A terminal episode can leave a partial trigger window.  Those
            # rows are discarded; a completed window must still be reset-free.
            if (valid & future_done.any(-1)).any():
                raise ValueError("reset-contaminated completed residual window")
            if int(valid.sum()) < (1 if args.engineering_smoke else 24):
                raise ValueError("insufficient residual source support")
            if expert_before != fingerprint([
                    dict(model=model.state_dict(), rms=rms.state_dict())
                    for model, rms in self.frozen_experts]):
                raise ValueError("frozen expert drift")
            env = ids[:, None].expand(-1, windows)[valid]
            residual_rows = window_residual[valid]
            payload = dict(
                schema="ref2dex.cm_residual_source.v1", seed=args.seed,
                assignment_seed=args.assignment_seed, codebook=codebook.cpu(),
                residual_scale_m=residual_scale.cpu(),
                episode_id=[f"s{args.seed}/env{int(i)}/first" for i in env.cpu()],
                env_id=env.cpu(), motion_id=motion[env].cpu(), start_frame=start[env].cpu(),
                trigger=trigger[valid].cpu(), assignment=assignment[valid].cpu(),
                propensity=torch.full((int(valid.sum()),), 1 / len(CODEBOOK)),
                state=pre[valid].cpu(), history=hist[valid].cpu(), residual=residual_rows.cpu(),
                residual_id=assignment[valid].cpu(), initial_clearance=initial_clearance[valid].cpu(),
                rest_z=rest[env].cpu(), future_state=future_state[valid].cpu(),
                future_contact=future_contact[valid].cpu(), future_clearance=future_clearance[valid].cpu(),
                future_done=future_done[valid].cpu(), native_observation=native_obs[valid].cpu(),
                baseline_action=baseline_action[valid].cpu(), actual_action=actual_action[valid].cpu(),
                baseline_pd_targets=baseline_pd[valid].cpu(), actual_pd_targets=actual_pd[valid].cpu(),
                requested_delta_world=requested_world[valid].cpu(), applied_delta_world=applied_world[valid].cpu(),
                residual_saturated=saturated[valid].cpu(), initial_hand_force=initial_hand[valid].cpu(),
                initial_object_force=initial_object[valid].cpu(), future_hand_force=raw_hand[valid].cpu(),
                future_object_force=raw_object[valid].cpu(), future_force_ratio=force_ratio[valid].cpu(),
                mass_kg=mass[env].cpu(), gravity_magnitude=gravity, control_dt_seconds=task.dt,
                frozen_experts=True, cm_used=False, optimizer_used=False,
                baseline="hf15_rotation_cup7_trigger_anchor", residual_steps=2,
                execution="baseline rotation-Cup feedback plus object-local translation residual for first two steps; baseline for remaining H10",
                contact_definition="normalized hand/object net-force presence > .1; pair identity unavailable",
                split_group_seed=12651,
                split_group_bucket=torch.tensor([
                    int(hashlib.sha256(f"12651/{int(motion[i])}/{int(start[i])}".encode()).hexdigest()[:8], 16) % 100
                    for i in env.cpu()
                ]),
            )
            torch.save(payload, args.output / "records.pt")
            result = dict(run_status="COMPLETED", rows=int(valid.sum()),
                          episodes=len(set(payload["episode_id"])),
                          residual_arm_counts=torch.bincount(payload["residual_id"].flatten(), minlength=len(CODEBOOK)).tolist(),
                          saturated_rows=int(payload["residual_saturated"].any(-1).sum()),
                          record_sha256=sha(args.output / "records.pt"),
                          elapsed_seconds=time.monotonic() - begin,
                          frozen_experts=True, cm_used=False)
            (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)

    return ResidualSourcePlayer


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--output-dir", dest="output", type=Path, required=True)
    parser.add_argument("--panel-seed", dest="seed", type=int, required=True)
    parser.add_argument("--assignment-seed", type=int, required=True)
    parser.add_argument("--windows-per-stratum", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=650)
    parser.add_argument("--wall-seconds", type=int, default=240)
    parser.add_argument("--engineering-smoke", action="store_true")
    args, remaining = parser.parse_known_args()
    base = (ROOT / "src/task/CmResidual/research/contact_consequence/output").resolve()
    if base not in args.output.resolve().parents or args.output.is_symlink():
        raise ValueError("new owned residual output required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = dict(run_status="STARTED", pid=os.getpid(), command=sys.argv,
                    gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), git_commit=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    experiment_id="P-20261003-cm-residual-policy", cm_training=False,
                    expert_training=False, residual_training=False)
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        sys.path.insert(0, str(ROOT / "third_party/DExplore/dexplore"))
        from isaacgym import gymtorch
        import torch
        import evaluate as original
        original.EvalPlayer = make_player(original, args, torch, gymtorch)
        sys.argv = [sys.argv[0], *remaining]
        original.main()
        manifest["run_status"] = "COMPLETED"
    except BaseException as error:
        manifest.update(run_status="FAILED", error=repr(error))
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
