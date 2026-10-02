#!/usr/bin/env python3
"""Native A/B validation of the frozen Cm uncertainty fallback."""
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


def candidate_player(original, args, torch, gymtorch):
    from src.task.CmResidual.executable_contact_options import hold_target, obj_vertices, TableClearance
    from src.task.CmResidual.native_pd_selector import FrozenNativePDSelector, native_pd_targets
    from src.task.CmResidual.orientation_anchored_options import orientation_anchored_action
    from src.task.CmResidual.paired_evaluation import fingerprint
    from src.task.CmResidual.weight_normalized_contact import weight_normalized_contacts

    parent = build_player(original, args, torch, gymtorch)

    class CandidatePlayer(parent):
        @torch.no_grad()
        def run(self):
            started = time.monotonic()
            torch.set_num_threads(2)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            task = self.env.task
            n, w = task.num_envs, args.windows_per_episode
            task._hybrid_init_prob = 0.0
            task._enable_early_termination = False
            task._adaptive_kappa_enabled = False
            if set(getattr(task, "object_name", ["airplane"])) != {"airplane"}:
                raise ValueError("airplane-only candidate contract")
            if n != 96 or self.is_rnn or abs(task.dt - 1 / 30) > 1e-8:
                raise ValueError("native execution contract")
            selector = FrozenNativePDSelector(args.checkpoint, self.device)
            if selector.fixed != 7:
                raise ValueError("fixed Cup candidate must be index 7")
            model_hash = sha(args.checkpoint)
            model_before = fingerprint({
                mode: [m.state_dict() for m in models]
                for mode, models in selector.models.items()
            })
            expert_before = fingerprint([
                dict(model=m.state_dict(), rms=r.state_dict())
                for m, r in self.frozen_experts
            ])
            geometry = TableClearance(
                obj_vertices(ASSETS / "objects/airplane/airplane.obj", self.device),
                obj_vertices(ASSETS / "objects/table/table.obj", self.device),
                getattr(task, "ball_size", 1.),
            )
            ids = torch.arange(n, device=self.device)
            obs = self.env_reset(ids)
            if self.get_batch_size(obs["obs"], 1) != n:
                raise ValueError("native batch")
            motion = task.data_id.clone()
            start = task.start_times.clone()
            rest = task.hoi_refs[task.data_id, task.ref_index, 0, 108].clone()
            held = torch.tensor([
                int(hashlib.sha256(f"9851/{int(i)}/{int(j)}".encode()).hexdigest()[:8], 16) % 100 >= 70
                for i, j in zip(motion, start)
            ], device=self.device)
            mass = torch.tensor([
                task.gym.get_actor_rigid_body_properties(e, h)[0].mass
                for e, h in zip(task.envs, task._target_handles)
            ], device=self.device)
            gravity = abs(task.sim_params.gravity.z)
            body_ids = task._contact_body_ids

            def contact_state():
                return weight_normalized_contacts(
                    task._contact_forces[:, body_ids], task._tar_contact_forces, mass, gravity
                )

            history = torch.zeros(n, 10, 69, device=self.device)
            previous = torch.zeros(n, 18, device=self.device)
            count = torch.zeros(n, dtype=torch.long, device=self.device)
            elapsed = torch.full_like(count, -1)
            cooldown = torch.zeros_like(count)
            contact_run = torch.zeros_like(count)
            ended = torch.zeros(n, dtype=torch.bool, device=self.device)
            anchor = torch.zeros(n, 18, device=self.device)
            cached = torch.zeros(n, 18, device=self.device)
            generator = torch.Generator(device=self.device).manual_seed(args.assignment_seed)
            reset = ids[:0]

            def zeros(*shape, dtype=torch.float32):
                return torch.zeros(n, w, *shape, dtype=dtype, device=self.device)

            trigger = torch.full((n, w), -1, dtype=torch.long, device=self.device)
            policy_assignment = torch.full_like(trigger, -1)
            chosen_candidate = torch.full_like(trigger, -1)
            steps = torch.zeros_like(trigger)
            pre = zeros(49)
            hist = zeros(10, 69)
            initial_clearance = zeros()
            future = zeros(10, 49)
            future_contact = zeros(10, 2, dtype=torch.bool)
            future_clearance = zeros(10)
            future_done = zeros(10, dtype=torch.bool)
            actual = zeros(10, 18)
            actual_pd = zeros(10, 18)
            candidate_actions = zeros(8, 18)
            candidate_pd = zeros(8, 18)
            cm_choice = torch.zeros(n, w, dtype=torch.long, device=self.device)
            cm_ood = torch.zeros(n, w, dtype=torch.bool, device=self.device)
            cm_candidate_ood = torch.zeros(n, w, 8, dtype=torch.bool, device=self.device)
            cm_diagnostics = {
                key: torch.zeros(n, w, 8, device=self.device)
                for key in ("score_mm", "relative_std_mm", "retention", "release")
            }

            for tick in range(args.max_steps):
                if time.monotonic() - started > args.wall_seconds:
                    raise TimeoutError("bounded candidate-advantage execution")
                obs = self.env_reset(reset)
                state = torch.cat((task._dof_pos.clone(), task._dof_vel.clone(), task._target_states.clone()), -1)
                contact = contact_state()[0].float()
                history = torch.cat((history[:, 1:], torch.cat((state, contact, previous), -1)[:, None]), 1)
                contact_run = torch.where(contact.bool().all(-1), contact_run + 1, 0)
                bank = self.candidates(obs)
                if bank.shape != (n, 6, 18):
                    raise ValueError("six expert candidate bank required")
                base = bank[:, 4]
                clearance = geometry.clearance(task._target_states, task._table_states)
                lifted = (state[:, 38] - rest >= .03) & (clearance >= .002)
                eligible = (
                    held & ~ended & (elapsed < 0) & (cooldown <= 0) & (contact_run >= 3) & lifted
                    & (tick >= 10) & (tick <= args.max_steps - 10)
                    & (task.max_episode_length[task.data_id] - task.progress_buf > 11)
                    & (count < w)
                )
                new = eligible.nonzero().flatten()
                slot = count[new]
                if len(new):
                    anchor[new] = hold_target(task._dof_pos[new], task._pd_action_offset, task._pd_action_scale)
                    base_hold = orientation_anchored_action(
                        bank[new, 4], anchor[new], task._dof_pos[new], task._pd_action_offset, task._pd_action_scale
                    )
                    cup_hold = orientation_anchored_action(
                        bank[new, 1], anchor[new], task._dof_pos[new], task._pd_action_offset, task._pd_action_scale
                    )
                    raw = torch.cat((bank[new], base_hold[:, None], cup_hold[:, None]), 1)
                    motor = native_pd_targets(
                        raw, task._dof_pos[new, None].expand(-1, 8, -1),
                        task._pd_action_offset, task._pd_action_scale,
                    )
                    _, diagnostic = selector.choose(
                        "cm", history[new], obs["obs"][new], rest[new], motor,
                        task._contact_forces[new][:, body_ids], task._tar_contact_forces[new], mass[new],
                        gravity, clearance[new],
                    )
                    score_mm = diagnostic["score_mm"]
                    std_mm = diagnostic["relative_std_mm"]
                    top = score_mm.argmax(-1)
                    top_score = score_mm.gather(1, top[:, None]).squeeze(1)
                    fixed_score = score_mm[:, selector.fixed]
                    top_std = std_mm.gather(1, top[:, None]).squeeze(1)
                    top_ood = diagnostic["candidate_ood"].gather(1, top[:, None]).squeeze(1)
                    top_retention = diagnostic["retention"].gather(1, top[:, None]).squeeze(1)
                    fixed_retention = diagnostic["retention"][:, selector.fixed]
                    top_release = diagnostic["release"].gather(1, top[:, None]).squeeze(1)
                    fixed_release = diagnostic["release"][:, selector.fixed]
                    cm_fallback = torch.where(
                        (top_score - fixed_score > 2 * top_std)
                        & ~top_ood
                        & (top_retention >= fixed_retention - .05)
                        & (top_release <= fixed_release + .05),
                        top, torch.full_like(top, selector.fixed),
                    )
                    policy = torch.randint(2, (len(new),), device=self.device, generator=generator)
                    draw = torch.where(policy == 0, cm_fallback, torch.full_like(cm_fallback, selector.fixed))
                    policy_assignment[new, slot] = policy
                    chosen_candidate[new, slot] = draw
                    trigger[new, slot] = tick
                    pre[new, slot] = state[new]
                    hist[new, slot] = history[new]
                    initial_clearance[new, slot] = clearance[new]
                    candidate_actions[new, slot] = raw
                    candidate_pd[new, slot] = motor
                    cm_choice[new, slot] = cm_fallback
                    cm_ood[new, slot] = diagnostic["ood"]
                    cm_candidate_ood[new, slot] = diagnostic["candidate_ood"]
                    for key in cm_diagnostics:
                        cm_diagnostics[key][new, slot] = diagnostic[key]
                    cached[new] = raw[torch.arange(len(new), device=self.device), draw]
                    elapsed[new] = 0

                live = (elapsed >= 0) & ~ended
                rows = live.nonzero().flatten()
                slot = count[rows]
                offset = elapsed[rows]
                action = base.clone()
                if len(rows):
                    first = offset == 0
                    if first.any():
                        action[rows[first]] = cached[rows[first]]
                    later = ~first
                    if later.any():
                        r = rows[later]
                        action[r] = orientation_anchored_action(
                            bank[r, 1], anchor[r], task._dof_pos[r],
                            task._pd_action_offset, task._pd_action_scale,
                        )
                applied = action.clone()
                targets = task._action_to_pd_targets(applied.clone()).clone()
                if len(rows):
                    expected = candidate_pd[rows, slot, chosen_candidate[rows, slot]]
                    if (offset == 0).any() and not torch.allclose(
                        targets[rows[offset == 0]], expected[offset == 0], atol=2e-5, rtol=1e-5
                    ):
                        raise ValueError("random candidate native target mismatch")
                    actual[rows, slot, offset] = applied[rows]
                    actual_pd[rows, slot, offset] = targets[rows]
                _, _, done, _ = self.env_step(self.env, action)
                done = done.bool().reshape(-1)
                after = torch.cat((task._dof_pos.clone(), task._dof_vel.clone(), task._target_states.clone()), -1)
                bits, _ = contact_state()
                if len(rows):
                    future[rows, slot, offset] = after[rows]
                    future_contact[rows, slot, offset] = bits[rows]
                    future_clearance[rows, slot, offset] = geometry.clearance(
                        task._target_states, task._table_states
                    )[rows]
                    future_done[rows, slot, offset] = done[rows]
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
                    print(json.dumps(dict(tick=tick, windows=int((steps == 10).sum()), held=int(held.sum()))), flush=True)
                if bool((ended | ~held | (count >= w)).all()):
                    break

            valid = steps == 10
            if ((steps > 0) & (~valid | future_done.any(-1))).any():
                raise ValueError("incomplete or terminal-contaminated window")
            if int(valid.sum()) < (1 if args.engineering_smoke else args.min_windows):
                raise ValueError("insufficient candidate windows")
            if model_before != fingerprint({
                mode: [m.state_dict() for m in models]
                for mode, models in selector.models.items()
            }) or expert_before != fingerprint([
                dict(model=m.state_dict(), rms=r.state_dict())
                for m, r in self.frozen_experts
            ]):
                raise ValueError("frozen model drift")

            env = ids[:, None].expand(-1, w)[valid]
            state_rows = pre[valid]
            f = future[valid]
            contact_rows = future_contact[valid].all(-1)
            clear_rows = future_clearance[valid]
            rest_rows = rest[:, None].expand(-1, w)[valid]
            retention = (clear_rows[:, -3:] >= .002).all(-1) & contact_rows[:, -3:].all(-1)
            score = (
                (f[:, -3:, 38].amin(-1) - rest_rows).clamp_min(0) * retention
                - (state_rows[:, 38] - rest_rows).clamp_min(0)
            ) * 1000
            payload = dict(
                schema="ref2dex.cm_uncertainty_native.v1",
                seed=args.seed, assignment_seed=args.assignment_seed,
                episode_id=[f"s{args.seed}/env{int(i)}/first" for i in env.cpu()],
                env_id=env.cpu(), motion_id=motion[:, None].expand(-1, w)[valid].cpu(),
                start_frame=start[:, None].expand(-1, w)[valid].cpu(), trigger=trigger[valid].cpu(),
                policy_assignment=policy_assignment[valid].cpu(),
                chosen_candidate=chosen_candidate[valid].cpu(),
                propensity=torch.full((int(valid.sum()),), .5),
                state=state_rows.cpu(), history=hist[valid].cpu(),
                initial_clearance=initial_clearance[valid].cpu(), rest_z=rest_rows.cpu(),
                candidate_actions=candidate_actions[valid].cpu(), candidate_pd_targets=candidate_pd[valid].cpu(),
                actual_action=actual[valid].cpu(), actual_pd_targets=actual_pd[valid].cpu(),
                future_state=f.cpu(), future_contact=future_contact[valid].cpu(),
                future_clearance=clear_rows.cpu(), future_done=future_done[valid].cpu(),
                cm_choice=cm_choice[valid].cpu(),
                cm_diagnostics={key: value[valid].cpu() for key, value in cm_diagnostics.items()},
                cm_ood=cm_ood[valid].cpu(), cm_candidate_ood=cm_candidate_ood[valid].cpu(),
                outcome=dict(
                    score_mm=score.cpu(), retained=retention.cpu(),
                    contact_last3=contact_rows[:, -3:].all(-1).cpu(),
                    clearance_last3=(clear_rows[:, -3:] >= .002).all(-1).cpu(),
                    first_delta_mm=((f[:, 0, 38] - state_rows[:, 38]) * 1000).cpu(),
                ),
                candidate_names=["expert0", "expert1", "expert2", "expert3", "expert4", "expert5", "base_hold", "fixed_cup"],
                intervention="policy A/B one-tick native candidate after observed contact; fixed Cup continuation nine ticks",
                assignment_after_observation=True,
                policy_propensity_definition="policy A=uncertainty fallback and policy B=fixed Cup, independent p=.5 after observation",
                policy_definition="policy_assignment 0=cm_uncertainty_fallback_2std, 1=fixed_cup; cm_choice stores fallback candidate; chosen_candidate stores executed candidate",
                fixed_cup_index=7, frozen_experts=True, frozen_cm=True, cm_used=True, optimizer_used=False,
                checkpoint_sha256=model_hash, pd_offset=task._pd_action_offset.cpu(), pd_scale=task._pd_action_scale.cpu(),
                contact_definition="weight-normalized netforce presence proxy",
                geometry_definition="full source mesh clearance over table plane",
            )
            torch.save(payload, args.output / "records.pt")
            result = dict(run_status="COMPLETED", rows=len(score), episodes=len(set(payload["episode_id"])),
                          policy_counts=torch.bincount(payload["policy_assignment"], minlength=2).tolist(),
                          elapsed_seconds=time.monotonic() - started,
                          record_sha256=sha(args.output / "records.pt"), frozen_cm=True, frozen_experts=True,
                          complete_labels=True)
            (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)

    return CandidatePlayer


def main():
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--output-dir", dest="output", type=Path, required=True)
    p.add_argument("--native-pd-checkpoint", dest="checkpoint", type=Path, required=True)
    p.add_argument("--panel-seed", dest="seed", type=int, required=True)
    p.add_argument("--assignment-seed", type=int, required=True)
    p.add_argument("--engineering-smoke", action="store_true")
    p.add_argument("--min-windows", type=int, default=12)
    p.add_argument("--windows-per-episode", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=650)
    p.add_argument("--wall-seconds", type=int, default=240)
    args, remaining = p.parse_known_args()
    base = (ROOT / "src/task/CmResidual/research/contact_consequence/output").resolve()
    if base not in args.output.resolve().parents or args.output.is_symlink():
        raise ValueError("new owned output required")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(run_status="STARTED", pid=os.getpid(), command=sys.argv,
                    gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    expert_training=False, cm_training=False, started=time.monotonic())
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        sys.path.insert(0, str(ROOT / "third_party/DExplore/dexplore"))
        from isaacgym import gymtorch
        import torch
        import evaluate as original
        original.EvalPlayer = candidate_player(original, args, torch, gymtorch)
        sys.argv = [sys.argv[0], *remaining]
        original.main()
        manifest["run_status"] = "COMPLETED"
    except BaseException as error:
        manifest.update(run_status="FAILED", error=repr(error))
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - manifest["started"]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
