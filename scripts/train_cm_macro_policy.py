#!/usr/bin/env python3
"""Train/evaluate an unbiased binary policy over a short Cm macro.

The frozen Cm selector proposes the raw-top candidate only.  A trainable binary
actor decides whether to execute that candidate for three ticks or keep the
fixed Cup action for the whole ten-tick window.
"""
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


def macro_player(original, args, torch, gymtorch):
    from src.task.CmResidual.cm_macro_policy import CmMacroPolicy, macro_features, ppo_loss
    from src.task.CmResidual.executable_contact_options import hold_target, obj_vertices, TableClearance
    from src.task.CmResidual.native_pd_selector import FrozenNativePDSelector, native_pd_targets
    from src.task.CmResidual.orientation_anchored_options import orientation_anchored_action
    from src.task.CmResidual.paired_evaluation import fingerprint
    from src.task.CmResidual.physical_value_contract import HoldTracker, private_initialization
    from src.task.CmResidual.physical_value_live import contacts
    from src.task.CmResidual.weight_normalized_contact import weight_normalized_contacts

    parent = build_player(original, args, torch, gymtorch)

    class MacroPlayer(parent):
        @torch.no_grad()
        def run(self):
            started = time.monotonic()
            torch.set_num_threads(2)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            task = self.env.task
            n = task.num_envs
            task._hybrid_init_prob = 0.0
            task._enable_early_termination = False
            task._adaptive_kappa_enabled = False
            if n != 96 or self.is_rnn or abs(task.dt - 1 / 30) > 1e-8:
                raise ValueError("native96env/stateless/30Hz required")
            if set(getattr(task, "object_name", ["airplane"])) != {"airplane"}:
                raise ValueError("airplane-only macro contract")

            selector = FrozenNativePDSelector(args.native_pd_checkpoint, self.device)
            if selector.fixed != 7:
                raise ValueError("fixed Cup candidate must be index 7")
            selector_hash = sha(args.native_pd_checkpoint)
            selector_before = fingerprint({
                mode: [m.state_dict() for m in models]
                for mode, models in selector.models.items()
            })
            experts_before = fingerprint([
                dict(model=m.state_dict(), rms=r.state_dict())
                for m, r in self.frozen_experts
            ])
            geometry = TableClearance(
                obj_vertices(ASSETS / "objects/airplane/airplane.obj", self.device),
                obj_vertices(ASSETS / "objects/table/table.obj", self.device),
                getattr(task, "ball_size", 1.),
            )

            with private_initialization(9381):
                policy = CmMacroPolicy().to(self.device)
            initial_fingerprint = fingerprint(policy.state_dict())
            if args.policy:
                saved = torch.load(args.policy, map_location=self.device, weights_only=False)
                if saved.get("cm_on") != args.cm_on or saved.get("selector_sha256") != selector_hash:
                    raise ValueError("policy mode/input drift")
                policy.load_state_dict(saved["policy"], strict=True)
            initial_policy = CmMacroPolicy().to(self.device)
            initial_policy.load_state_dict(policy.state_dict())
            if args.policy:
                # For a loaded evaluation checkpoint the saved initial fingerprint
                # remains the reference used by the training result.
                initial_fingerprint = saved.get("initial_fingerprint", initial_fingerprint)

            optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
            generator = torch.Generator(device=self.device).manual_seed(args.assignment_seed)
            reports = []
            all_buffers = []
            traces = []
            updates = 0
            latencies = []
            ids = torch.arange(n, device=self.device)
            empty = ids[:0]
            fixed = selector.fixed

            for rollout in range(args.rollouts):
                observation = self.env_reset(ids)
                if self.get_batch_size(observation["obs"], 1) != n:
                    raise ValueError("native batch")
                motion = task.data_id.clone()
                start = task.start_times.clone()
                rest = task.hoi_refs[task.data_id, task.ref_index, 0, 108].clone()
                # Keep the raw macro's held subset and deterministic assignment.
                held = torch.tensor([
                    int(hashlib.sha256((f"9851/{int(i)}/{int(j)}").encode()).hexdigest()[:8], 16) % 100 >= 70
                    for i, j in zip(motion, start)
                ], device=self.device, dtype=torch.bool)
                mass = torch.tensor([
                    task.gym.get_actor_rigid_body_properties(e, h)[0].mass
                    for e, h in zip(task.envs, task._target_handles)
                ], device=self.device)
                gravity = abs(task.sim_params.gravity.z)
                body_ids = task._contact_body_ids

                def contact_state():
                    return weight_normalized_contacts(
                        task._contact_forces[:, body_ids], task._tar_contact_forces,
                        mass, gravity
                    )

                history = torch.zeros(n, 10, 69, device=self.device)
                previous = torch.zeros(n, 18, device=self.device)
                contact_run = torch.zeros(n, dtype=torch.long, device=self.device)
                ended = torch.zeros(n, dtype=torch.bool, device=self.device)
                elapsed = torch.full_like(contact_run, -1)
                cooldown = torch.zeros_like(contact_run)
                cached = torch.zeros(n, 18, device=self.device)
                anchor = torch.zeros(n, 18, device=self.device)
                chosen_option = torch.full((n,), -1, dtype=torch.long, device=self.device)
                chosen_candidate = torch.full((n,), -1, dtype=torch.long, device=self.device)
                trigger_tick = torch.full((n,), -1, dtype=torch.long, device=self.device)
                acquired = torch.zeros(n, dtype=torch.bool, device=self.device)
                contact_lost = torch.zeros(n, dtype=torch.long, device=self.device)
                tracker = HoldTracker(n, self.device)
                tracker.reset(ids, rest)
                horizon = int(task.max_episode_length.max()) + 5
                rewards = torch.zeros(horizon, n, device=self.device)
                native_rewards = torch.zeros_like(rewards)
                episode_steps = torch.zeros(n, dtype=torch.long, device=self.device)
                object_trace = torch.zeros(horizon, n, 13, device=self.device)
                contact_trace = torch.zeros(horizon, n, 2, dtype=torch.bool, device=self.device)
                action_trace = torch.zeros(horizon, n, 18, device=self.device)
                done_trace = torch.zeros(horizon, n, dtype=torch.bool, device=self.device)
                buffer = []
                reset_ids = empty
                previous_z = rest.clone()
                metrics = {k: torch.zeros(n, dtype=torch.bool, device=self.device) for k in (
                    "stable_success", "ever_stable", "drop_after_stable", "five_step_hold",
                    "acquired_lift", "post_lift_release")}

                for tick in range(horizon):
                    if time.monotonic() - started > args.wall_seconds:
                        raise TimeoutError("bounded binary macro learning")
                    observation = self.env_reset(reset_ids)
                    state = torch.cat((task._dof_pos.clone(), task._dof_vel.clone(), task._target_states.clone()), -1)
                    contact = contacts(task)
                    history = torch.cat((history[:, 1:], torch.cat((state, contact, previous), -1)[:, None]), 1)
                    contact_run = torch.where(contact.bool().all(-1), contact_run + 1, 0)
                    bank = self.candidates(observation)
                    if bank.shape != (n, 6, 18):
                        raise ValueError("six expert candidate bank required")
                    base = bank[:, 4]
                    clearance = geometry.clearance(task._target_states, task._table_states)
                    lifted = (state[:, 38] - rest >= .03) & (clearance >= .002)
                    remaining = task.max_episode_length[task.data_id] - task.progress_buf
                    eligible = (
                        held & ~ended & (elapsed < 0) & (cooldown <= 0) & (contact_run >= 3)
                        & lifted & (tick >= 10) & (tick <= horizon - 10)
                        & (remaining > 11)
                    )
                    rows = eligible.nonzero().flatten()
                    if len(rows):
                        anchor[rows] = hold_target(
                            task._dof_pos[rows], task._pd_action_offset, task._pd_action_scale
                        )
                        base_hold = orientation_anchored_action(
                            bank[rows, 4], anchor[rows], task._dof_pos[rows],
                            task._pd_action_offset, task._pd_action_scale
                        )
                        cup_hold = orientation_anchored_action(
                            bank[rows, 1], anchor[rows], task._dof_pos[rows],
                            task._pd_action_offset, task._pd_action_scale
                        )
                        raw = torch.cat((bank[rows], base_hold[:, None], cup_hold[:, None]), 1)
                        motor = native_pd_targets(
                            raw, task._dof_pos[rows, None].expand(-1, 8, -1),
                            task._pd_action_offset, task._pd_action_scale,
                        )
                        _, diagnostic = selector.choose(
                            "cm", history[rows], observation["obs"][rows], rest[rows], motor,
                            task._contact_forces[rows][:, body_ids], task._tar_contact_forces[rows],
                            mass[rows], gravity, clearance[rows],
                        )
                        top = diagnostic["score_mm"].argmax(-1)
                        diff = raw[torch.arange(len(rows), device=self.device), top] - raw[:, fixed]
                        physical = torch.stack((
                            diagnostic["score_mm"].gather(1, top[:, None]).squeeze(1),
                            diagnostic["relative_std_mm"].gather(1, top[:, None]).squeeze(1),
                            diagnostic["retention"].gather(1, top[:, None]).squeeze(1),
                            diagnostic["release"].gather(1, top[:, None]).squeeze(1),
                        ), -1)
                        features = macro_features(history[rows], diff, physical, args.cm_on)
                        torch.cuda.synchronize()
                        infer_start = time.monotonic()
                        distribution, value = policy(features)
                        if args.evaluate:
                            selected = distribution.probs.argmax(-1)
                        else:
                            selected = torch.multinomial(distribution.probs, 1, generator=generator).squeeze(-1)
                        torch.cuda.synchronize()
                        latencies.append(1000 * (time.monotonic() - infer_start))
                        option_action = torch.where(
                            selected[:, None] == 0,
                            raw[torch.arange(len(rows), device=self.device), top],
                            raw[:, fixed],
                        )
                        cached[rows] = option_action
                        chosen_option[rows] = selected
                        chosen_candidate[rows] = torch.where(selected == 0, top, torch.full_like(top, fixed))
                        trigger_tick[rows] = tick
                        elapsed[rows] = 0
                        buffer.append(dict(
                            features=features.clone(), selected=selected.clone(),
                            old_logprob=distribution.log_prob(selected).clone(), old_value=value.clone(),
                            env=rows.clone(), tick=tick, candidate=option_action.clone(),
                            cm_choice=top.clone(), physical=physical.clone(),
                            action_difference=diff.clone(), option0_action=raw[torch.arange(len(rows), device=self.device), top].clone(),
                            fixed_action=raw[:, fixed].clone(),
                        ))

                    live = (elapsed >= 0) & ~ended
                    rows_live = live.nonzero().flatten()
                    action = base.clone()
                    if len(rows_live):
                        offset = elapsed[rows_live]
                        opt = chosen_option[rows_live]
                        macro = (opt == 0) & (offset < args.macro_steps)
                        if macro.any():
                            action[rows_live[macro]] = cached[rows_live[macro]]
                        cup = (~macro)
                        if cup.any():
                            r = rows_live[cup]
                            action[r] = orientation_anchored_action(
                                bank[r, 1], anchor[r], task._dof_pos[r],
                                task._pd_action_offset, task._pd_action_scale,
                            )
                    applied = action.clone()
                    if len(rows_live) and (elapsed[rows_live] == 0).any():
                        first = rows_live[elapsed[rows_live] == 0]
                        # The raw candidate-to-motor conversion is checked for
                        # every newly triggered row directly from the cached
                        # command, before the simulator advances.
                        targets = task._action_to_pd_targets(applied.clone()).clone()
                        if not torch.allclose(
                            targets[first], native_pd_targets(
                                cached[first], task._dof_pos[first],
                                task._pd_action_offset, task._pd_action_scale,
                            ),
                            atol=2e-5, rtol=1e-5,
                        ):
                            raise ValueError("macro candidate native target mismatch")

                    before_z = task._target_states[:, 2].clone()
                    _, native_reward, done, _ = self.env_step(self.env, action)
                    done = done.bool().reshape(-1)
                    if not torch.isfinite(native_reward).all() or not torch.isfinite(task._target_states).all():
                        raise ValueError("nonfinite native trajectory")
                    pair = contacts(task).bool().all(-1)
                    z = task._target_states[:, 2]
                    clear_after = geometry.clearance(task._target_states, task._table_states)
                    # Common physical reward: supported height progress, a small
                    # supported-height term, and explicit contact/clearance loss.
                    support = ((z - rest) / .03).clamp(0, 1) * pair.float()
                    progress = ((z - before_z) / .03).clamp(-1, 1) * pair.float()
                    acquired |= (z - rest >= .03) & pair
                    contact_lost = torch.where(acquired & ~pair, contact_lost + 1, torch.zeros_like(contact_lost))
                    loss = acquired & ((contact_lost >= 1) | (clear_after < .002))
                    rewards[tick] = (progress + .1 * support - .5 * loss.float()) * (~ended)
                    native_rewards[tick] = native_reward.reshape(-1) * (~ended)
                    episode_steps += (~ended).long()
                    object_trace[tick] = task._target_states.clone()
                    contact_trace[tick] = contacts(task).bool()
                    action_trace[tick] = applied
                    tracker.step(z, pair)
                    new = done & ~ended
                    metrics["ever_stable"][new] = tracker.stable[new] | (tracker.max_run[new] >= 1 - 1e-6)
                    metrics["drop_after_stable"][new] = tracker.drop_after_success[new]
                    metrics["stable_success"][new] = tracker.stable[new] & ~tracker.drop_after_success[new]
                    metrics["five_step_hold"][new] = tracker.max_run[new] >= 5 / 30 - 1e-6
                    metrics["acquired_lift"][new] = acquired[new]
                    metrics["post_lift_release"][new] = acquired[new] & loss[new]
                    done_trace[tick] = done
                    elapsed[live] += 1
                    complete = live & (elapsed == 10)
                    elapsed[complete] = -1
                    cooldown[complete] = 6
                    cooldown = (cooldown - 1).clamp_min(0)
                    ended |= done
                    reset_ids = done.nonzero().flatten()
                    previous = applied
                    if tick % 200 == 0:
                        print(json.dumps(dict(rollout=rollout, tick=tick, episodes_complete=int(ended.sum()),
                                              decisions=sum(len(b["selected"]) for b in buffer))), flush=True)
                    if ended.all():
                        break

                if not ended.all():
                    raise ValueError("incomplete episodes; no trimming")
                if not buffer:
                    raise ValueError("no binary macro decisions")
                running = torch.zeros(n, device=self.device)
                returns = torch.zeros_like(rewards)
                for tick in reversed(range(horizon)):
                    running = rewards[tick] + .99 * running
                    returns[tick] = running
                features = torch.cat([b["features"] for b in buffer])
                selected = torch.cat([b["selected"] for b in buffer])
                old = torch.cat([b["old_logprob"] for b in buffer])
                old_value = torch.cat([b["old_value"] for b in buffer])
                target = torch.cat([returns[b["tick"], b["env"]] for b in buffer])
                advantage = target - old_value
                advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
                before_update = fingerprint(policy.state_dict())
                losses = []
                first_ratio_error = 0.0
                if not args.evaluate:
                    with torch.enable_grad():
                        for epoch in range(args.epochs):
                            permutation = torch.randperm(len(selected), device=self.device, generator=generator)
                            for offset in range(0, len(selected), 128):
                                r = permutation[offset:offset + 128]
                                distribution, value = policy(features[r])
                                loss, stats = ppo_loss(distribution, value, selected[r], old[r], advantage[r], target[r])
                                if updates == 0:
                                    first_ratio_error = float((stats["ratio"] - 1).abs().max())
                                if not torch.isfinite(loss):
                                    raise ValueError("nonfinite PPO loss")
                                optimizer.zero_grad()
                                loss.backward()
                                if not all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None):
                                    raise ValueError("nonfinite PPO gradient")
                                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1)
                                if not torch.isfinite(norm):
                                    raise ValueError("nonfinite PPO gradient norm")
                                optimizer.step()
                                updates += 1
                                losses.append(float(loss))
                with torch.no_grad():
                    after_distribution, _ = policy(features)
                changed = (after_distribution.probs.argmax(-1) != 0).float().mean()
                training_return = returns[0].mean()
                report = dict(
                    rollout=rollout, episodes=n, decisions=len(selected),
                    episode_steps=episode_steps.cpu().tolist(),
                    supported_return=float(training_return),
                    native_mean_discounted_return=float((native_rewards[0] + 0).mean()),
                    metrics={k: int(v.sum()) for k, v in metrics.items()},
                    selected_counts=torch.bincount(selected, minlength=2).tolist(),
                    argmax_option1_fraction=float(changed),
                    first_actual_option_ratio_error=first_ratio_error,
                    parameter_update=before_update != fingerprint(policy.state_dict()),
                    optimizer_updates=updates,
                    loss_mean=(sum(losses) / len(losses)) if losses else None,
                )
                reports.append(report)
                traces.append(dict(object_state=object_trace[:int(episode_steps.max())].cpu(),
                                   contact=contact_trace[:int(episode_steps.max())].cpu(),
                                   action=action_trace[:int(episode_steps.max())].cpu(),
                                   done=done_trace[:int(episode_steps.max())].cpu(),
                                   episode_steps=episode_steps.cpu(),
                                   reward=rewards[:int(episode_steps.max())].cpu(),
                                   native_reward=native_rewards[:int(episode_steps.max())].cpu()))
                all_buffers.append(dict(features=features.cpu(), selected=selected.cpu(),
                                        old_logprob=old.cpu(), returns=target.cpu(),
                                        env=torch.cat([b["env"] for b in buffer]).cpu(),
                                        tick=torch.cat([torch.full_like(b["env"], b["tick"]) for b in buffer]).cpu(),
                                        cm_choice=torch.cat([b["cm_choice"] for b in buffer]).cpu(),
                                        physical=torch.cat([b["physical"] for b in buffer]).cpu(),
                                        action_difference=torch.cat([b["action_difference"] for b in buffer]).cpu()))
                print(json.dumps(report), flush=True)

            if selector_before != fingerprint({mode: [m.state_dict() for m in models] for mode, models in selector.models.items()}):
                raise ValueError("frozen Cm selector drift")
            if experts_before != fingerprint([dict(model=m.state_dict(), rms=r.state_dict()) for m, r in self.frozen_experts]):
                raise ValueError("frozen expert drift")
            torch.save(dict(schema="ref2dex.cm_macro_policy.v1", policy=policy.state_dict(), cm_on=args.cm_on,
                            selector_sha256=selector_hash, initial_fingerprint=initial_fingerprint,
                            optimizer_updates=updates, macro_steps=args.macro_steps), args.output / "policy.pt")
            torch.save(all_buffers, args.output / "decisions.pt")
            torch.save(traces, args.output / "episode_traces.pt")
            result = dict(run_status="COMPLETED", cm_on=args.cm_on, evaluate=args.evaluate, smoke=args.smoke,
                          rollouts=reports, optimizer_updates=updates,
                          policy_parameters_changed=initial_fingerprint != fingerprint(policy.state_dict()),
                          actual_argmax_option1_fraction=float(torch.cat([torch.as_tensor([r["argmax_option1_fraction"]]) for r in reports]).mean()),
                          native_env_steps=sum(sum(r["episode_steps"]) for r in reports),
                          batched_inference_ms=latencies,
                          selector_sha256=selector_hash, frozen_cm=True, frozen_experts=True,
                          policy_sha256=sha(args.output / "policy.pt"), decisions_sha256=sha(args.output / "decisions.pt"),
                          episode_traces_sha256=sha(args.output / "episode_traces.pt"), elapsed_seconds=time.monotonic() - started,
                          scientific_label="ENGINEERING_SMOKE" if args.smoke else "AWAITING_MATCHED_ANALYSIS")
            (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k not in ("rollouts", "batched_inference_ms")}), flush=True)

    return MacroPlayer


def main():
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--output-dir", dest="output", type=Path, required=True)
    p.add_argument("--native-pd-checkpoint", type=Path, required=True)
    p.add_argument("--policy", type=Path)
    p.add_argument("--cm-on", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--rollouts", type=int, default=4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--assignment-seed", type=int, default=8103)
    p.add_argument("--macro-steps", type=int, default=3)
    p.add_argument("--wall-seconds", type=int, default=1800)
    args, remaining = p.parse_known_args()
    if not 1 <= args.macro_steps < 10:
        raise ValueError("macro-steps must be in [1,9]")
    base = (ROOT / "src/task/CmResidual/research/contact_consequence/output").resolve()
    if base not in args.output.resolve().parents or args.output.is_symlink():
        raise ValueError("new owned output required")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(run_status="STARTED", pid=os.getpid(), command=sys.argv,
                    gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    cm_training=False, expert_training=False, option_policy_training=not args.evaluate,
                    started=time.monotonic())
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        sys.path.insert(0, str(ROOT / "third_party/DExplore/dexplore"))
        from isaacgym import gymtorch
        import torch
        import evaluate as original
        original.EvalPlayer = macro_player(original, args, torch, gymtorch)
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
