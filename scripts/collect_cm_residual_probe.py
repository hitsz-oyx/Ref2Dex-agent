#!/usr/bin/env python3
"""Run a native baseline/Cm-residual/shuffled residual comparison."""
from __future__ import annotations

import argparse
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


def make_player(original, args, torch_module, gymtorch):
    from src.task.CmResidual.cm_residual_policy import FrozenResidualPolicy, context_features, apply_residual_action
    from src.task.CmResidual.executable_contact_options import hold_target, obj_vertices, TableClearance
    from src.task.CmResidual.orientation_anchored_options import orientation_anchored_action
    from src.task.CmResidual.weight_normalized_contact import weight_normalized_contacts
    from src.task.CmResidual.paired_evaluation import fingerprint

    parent = build_player(original, args, torch_module, gymtorch)

    class ResidualProbePlayer(parent):
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
            n = task.num_envs
            if n != 96 or self.is_rnn or abs(task.dt - 1 / 30) > 1e-8:
                raise ValueError("native residual probe contract")
            if set(getattr(task, "object_name", ["airplane"])) != {"airplane"}:
                raise ValueError("airplane-only residual probe")
            if args.windows_per_env < 1 or args.windows_per_env > 8:
                raise ValueError("bounded windows per env")
            geometry = TableClearance(
                obj_vertices(ASSETS / "objects/airplane/airplane.obj", self.device),
                obj_vertices(ASSETS / "objects/table/table.obj", self.device),
                getattr(task, "ball_size", 1.0),
            )
            ids = torch.arange(n, device=self.device)
            observation = self.env_reset(ids)
            if self.get_batch_size(observation["obs"], 1) != n or observation["obs"].shape[-1] != 1442:
                raise ValueError("native observation shape contract")
            windows = args.windows_per_env
            motion = task.data_id.clone()
            start = task.start_times.clone()
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
            policy = None
            if args.policy_checkpoint:
                policy = {
                    "residual": FrozenResidualPolicy(args.policy_checkpoint, self.device, "cm"),
                    "shuffled": FrozenResidualPolicy(args.policy_checkpoint, self.device, "shuffled"),
                }
            history = torch.zeros(n, 10, 69, device=self.device)
            previous = torch.zeros(n, 18, device=self.device)
            count = torch.zeros(n, dtype=torch.long, device=self.device)
            elapsed = torch.full_like(count, -1)
            cooldown = torch.zeros_like(count)
            contact_run = torch.zeros_like(count)
            ended = torch.zeros(n, dtype=torch.bool, device=self.device)
            anchor = torch.zeros(n, 18, device=self.device)
            arm = torch.full((n, windows), -1, dtype=torch.long, device=self.device)
            trigger = torch.full_like(arm, -1)
            steps = torch.zeros_like(arm)
            pre = torch.zeros(n, windows, 49, device=self.device)
            hist = torch.zeros(n, windows, 10, 69, device=self.device)
            proposed = torch.zeros(n, windows, 3, device=self.device)
            executed = torch.zeros(n, windows, 3, device=self.device)
            accepted = torch.zeros(n, windows, dtype=torch.bool, device=self.device)
            ood = torch.zeros_like(accepted)
            score_mm = torch.zeros(n, windows, device=self.device)
            zero_score_mm = torch.zeros_like(score_mm)
            future_state = torch.zeros(n, windows, 10, 49, device=self.device)
            future_contact = torch.zeros(n, windows, 10, 2, dtype=torch.bool, device=self.device)
            future_clearance = torch.zeros(n, windows, 10, device=self.device)
            future_done = torch.zeros(n, windows, 10, dtype=torch.bool, device=self.device)
            initial_clearance = torch.zeros(n, windows, device=self.device)
            assignment_generator = torch.Generator(device=self.device).manual_seed(args.assignment_seed)
            reset = ids[:0]

            for tick in range(args.max_steps):
                if time.monotonic() - begin > args.wall_seconds:
                    raise TimeoutError("native residual probe budget")
                observation = self.env_reset(reset)
                state = torch.cat((task._dof_pos.clone(), task._dof_vel.clone(), task._target_states.clone()), -1)
                contact, _ = contacts()
                history = torch.cat((history[:, 1:], torch.cat((state, contact.float(), previous), -1)[:, None]), 1)
                contact_run = torch.where(contact.all(-1), contact_run + 1, torch.zeros_like(contact_run))
                bank = self.candidates(observation)
                clearance = geometry.clearance(task._target_states, task._table_states)
                strata = ((state[:, 38] - rest >= .03) & (clearance >= .002)).long()
                eligible = ((~ended) & (elapsed < 0) & (cooldown <= 0) & (contact_run >= 3)
                            & (state[:, 38] - rest >= (0.0 if args.engineering_smoke else .005)) & (tick >= 10)
                            & (tick <= args.max_steps - 12)
                            & (task.max_episode_length[task.data_id] - task.progress_buf > 11)
                            & (count < windows))
                rows = eligible.nonzero().flatten()
                if len(rows):
                    slot = count[rows]
                    arm[rows, slot] = torch.randint(0, 3, (len(rows),), device=self.device,
                                                      generator=assignment_generator)
                    anchor[rows] = hold_target(task._dof_pos[rows], task._pd_action_offset,
                                               task._pd_action_scale)
                    trigger[rows, slot] = tick
                    pre[rows, slot] = state[rows]
                    hist[rows, slot] = history[rows]
                    initial_clearance[rows, slot] = clearance[rows]
                    if policy is not None:
                        context = context_features(
                            state[rows], task._contact_forces[rows][:, body_ids],
                            task._tar_contact_forces[rows], mass[rows], gravity,
                            clearance[rows], rest[rows])
                        residual_rows = torch.zeros(len(rows), 3, device=self.device)
                        arm_rows = arm[rows, slot]
                        for mode, mode_name in ((1, "residual"), (2, "shuffled")):
                            selected = (arm_rows == mode).nonzero().flatten()
                            if len(selected):
                                residual_rows[selected], detail = policy[mode_name].choose(
                                    history[rows[selected]], context[selected])
                                proposed[rows[selected], slot[selected]] = detail["proposed"]
                                accepted[rows[selected], slot[selected]] = detail["accepted"]
                                ood[rows[selected], slot[selected]] = detail["ood"]
                                score_mm[rows[selected], slot[selected]] = detail["score_mm"]
                                zero_score_mm[rows[selected], slot[selected]] = detail["zero_score_mm"]
                        executed[rows, slot] = residual_rows
                    elapsed[rows] = 0
                live = (elapsed >= 0) & ~ended
                rows = live.nonzero().flatten()
                slot = count[rows]
                offset = elapsed[rows]
                action = bank[:, 4].clone()
                if len(rows):
                    baseline = orientation_anchored_action(
                        bank[rows, 1], anchor[rows], task._dof_pos[rows],
                        task._pd_action_offset, task._pd_action_scale)
                    residual_rows = executed[rows, slot]
                    residual_rows = torch.where(offset[:, None] < 2, residual_rows,
                                                torch.zeros_like(residual_rows))
                    modified, detail = apply_residual_action(
                        baseline, state[rows, 39:43], residual_rows)
                    action[rows] = modified
                    if (detail["saturated"] & (offset < 2)).any():
                        raise ValueError("accepted residual saturated native translation")
                _, _, done, _ = self.env_step(self.env, action)
                done = done.bool().reshape(-1)
                after = torch.cat((task._dof_pos.clone(), task._dof_vel.clone(), task._target_states.clone()), -1)
                post_contact, _ = contacts()
                post_clearance = geometry.clearance(task._target_states, task._table_states)
                if len(rows):
                    future_state[rows, slot, offset] = after[rows]
                    future_contact[rows, slot, offset] = post_contact[rows]
                    future_clearance[rows, slot, offset] = post_clearance[rows]
                    future_done[rows, slot, offset] = done[rows]
                    steps[rows, slot] += 1
                elapsed[live] += 1
                complete = live & (elapsed == 10)
                count[complete] += 1
                elapsed[complete] = -1
                cooldown = (cooldown - 1).clamp_min(0)
                cooldown[complete] = 6
                ended |= done
                reset = done.nonzero().flatten()
                previous = action
                if tick % 100 == 0:
                    print(json.dumps(dict(tick=tick, rows=int((steps == 10).sum()), active=int(live.sum()))), flush=True)
                if (ended | (count >= windows)).all():
                    break

            valid = (steps == 10) & (arm >= 0)
            if (valid & future_done.any(-1)).any():
                raise ValueError("reset-contaminated completed residual probe window")
            if int(valid.sum()) < (3 if args.engineering_smoke else 24):
                raise ValueError("insufficient native residual probe support")
            if expert_before != fingerprint([
                    dict(model=model.state_dict(), rms=rms.state_dict())
                    for model, rms in self.frozen_experts]):
                raise ValueError("frozen expert drift")
            idx = valid.nonzero(as_tuple=False)
            payload = dict(
                schema="ref2dex.cm_residual_probe.v1", experiment_id="P-20261003-cm-residual-policy",
                seed=args.seed, assignment_seed=args.assignment_seed,
                env_id=idx[:, 0].cpu(), slot=idx[:, 1].cpu(), arm=arm[valid].cpu(),
                motion_id=motion[idx[:, 0]].cpu(), start_frame=start[idx[:, 0]].cpu(),
                trigger=trigger[valid].cpu(), state=pre[valid].cpu(), history=hist[valid].cpu(),
                proposed_residual=proposed[valid].cpu(), executed_residual=executed[valid].cpu(),
                accepted=accepted[valid].cpu(), ood=ood[valid].cpu(),
                score_mm=score_mm[valid].cpu(), zero_score_mm=zero_score_mm[valid].cpu(),
                initial_clearance=initial_clearance[valid].cpu(), rest_z=rest[idx[:, 0]].cpu(),
                future_state=future_state[valid].cpu(), future_contact=future_contact[valid].cpu(),
                future_clearance=future_clearance[valid].cpu(), future_done=future_done[valid].cpu(),
                residual_steps=2, execution="native baseline rotation-Cup feedback; accepted residual for first 2 steps; baseline H10",
                baseline="hf15_rotation_cup7_trigger_anchor", frozen_experts=True,
                contact_definition="normalized hand/object net-force presence > .1; pair identity unavailable",
                held_gate="split_group_bucket>=70 is calculated by audit from motion/start hash",
            )
            torch.save(payload, args.output / "records.pt")
            result = dict(run_status="COMPLETED", rows=int(valid.sum()),
                          arm_counts=torch.bincount(payload["arm"], minlength=3).tolist(),
                          accepted_by_arm=[int(payload["accepted"][payload["arm"] == arm_id].sum()) for arm_id in range(3)],
                          ood_by_arm=[int(payload["ood"][payload["arm"] == arm_id].sum()) for arm_id in range(3)],
                          residual_changed_by_arm=[int((payload["executed_residual"][payload["arm"] == arm_id].abs().sum(-1) > 0).sum()) for arm_id in range(3)],
                          record_sha256=sha(args.output / "records.pt"), elapsed_seconds=time.monotonic() - begin,
                          frozen_experts=True, policy_frozen=True)
            (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)

    return ResidualProbePlayer


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--output-dir", dest="output", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--assignment-seed", type=int, required=True)
    parser.add_argument("--windows-per-env", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=650)
    parser.add_argument("--wall-seconds", type=int, default=240)
    parser.add_argument("--engineering-smoke", action="store_true")
    args, remaining = parser.parse_known_args()
    args.seed = int(os.environ.get("PYTHONHASHSEED", "0"))
    base = (ROOT / "src/task/CmResidual/research/contact_consequence/output").resolve()
    if base not in args.output.resolve().parents or args.output.is_symlink():
        raise ValueError("new owned residual probe output required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = dict(run_status="STARTED", pid=os.getpid(), command=sys.argv,
                    experiment_id="P-20261003-cm-residual-policy", policy_checkpoint_sha256=sha(args.policy_checkpoint),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        sys.path.insert(0, str(ROOT / "third_party/DExplore/dexplore"))
        from isaacgym import gymtorch
        import evaluate as original
        import torch as torch_module
        globals()["torch"] = torch_module
        original.EvalPlayer = make_player(original, args, torch_module, gymtorch)
        sys.argv = [sys.argv[0], *remaining]
        original.main()
        manifest["run_status"] = "COMPLETED"
    except BaseException as error:
        manifest.update(run_status="FAILED", error=repr(error))
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
