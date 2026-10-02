#!/usr/bin/env python3
"""Collect a small native candidate-panel Probe for the Cm decision interface.

The physical-value bundle is frozen.  At eligible contact states we build the
same 37 local action candidates used by the value teacher, score the full panel
with direct-Q and short model rollout, then randomly execute one of five arms.
No actor, value, or dynamics parameter is updated in this script.
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


ARMS = ("cup", "direct_q", "cm_value", "shuffled", "random")


def _load_teacher(checkpoint, device):
    import torch
    from torch import nn
    from src.task.CmResidual.physical_value_models import Features, OutcomeNetwork, Teacher

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.physical_value_models.v1":
        raise ValueError("invalid physical-value checkpoint schema")
    if abs(float(payload["gamma"]) - .99) > 1e-8:
        raise ValueError("decision-interface Probe requires gamma=.99")
    stats = {key: value.to(device) for key, value in payload["stats"].items()}
    features = Features(**stats, device=device).to(device).eval()
    value = OutcomeNetwork(payload["context_dim"], 0, 1, 9283).to(device)
    direct_q = OutcomeNetwork(payload["context_dim"], 18, 1, 9284).to(device)
    dynamics = nn.ModuleList([
        OutcomeNetwork(payload["context_dim"], 18, 53, 9300 + i).to(device)
        for i in range(3)
    ])
    value.load_state_dict(payload["value"], strict=True)
    direct_q.load_state_dict(payload["direct_q"], strict=True)
    for model, state in zip(dynamics, payload["dynamics"]):
        model.load_state_dict(state, strict=True)
    for model in (value, direct_q, dynamics):
        model.eval().requires_grad_(False)
    teacher = Teacher(features, dynamics, value, direct_q, float(payload["gamma"]),
                      2026100201, payload["reward_scale"].to(device))
    return teacher, features, value, direct_q, dynamics


def make_player(original, args, torch, gymtorch):
    from src.task.CmResidual.executable_contact_options import hold_target, obj_vertices, TableClearance
    from src.task.CmResidual.orientation_anchored_options import orientation_anchored_action
    from src.task.CmResidual.paired_evaluation import fingerprint
    from src.task.CmResidual.physical_value_contract import HistoryBuffer, make_candidates
    from src.task.CmResidual.physical_value_live import contacts, context, snapshot
    from src.task.CmResidual.weight_normalized_contact import weight_normalized_contacts

    parent = build_player(original, args, torch, gymtorch)

    class DecisionInterfacePlayer(parent):
        @torch.no_grad()
        def run(self):
            started = time.monotonic()
            torch.set_num_threads(2)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            task = self.env.task
            task._hybrid_init_prob = 0.0
            task._enable_early_termination = False
            task._adaptive_kappa_enabled = False
            if task.num_envs != 96 or self.is_rnn or abs(task.dt - 1 / 30) > 1e-8:
                raise ValueError("native decision-interface contract")
            if set(getattr(task, "object_name", ["airplane"])) != {"airplane"}:
                raise ValueError("airplane-only decision-interface Probe")
            if not 1 <= args.windows_per_stratum <= 4:
                raise ValueError("bounded windows per stratum")

            # rl_games exposes the generic ``cuda`` device while Isaac Gym
            # tensors are explicitly ``cuda:0``; the FK contract requires the
            # latter exact device identity.
            model_device = task._dof_pos.device
            teacher, features, value, direct_q, dynamics = _load_teacher(args.value_checkpoint, model_device)
            teacher_fingerprint = fingerprint([
                {"features": features.state_dict(), "value": value.state_dict(),
                 "direct_q": direct_q.state_dict(),
                 "dynamics": dynamics.state_dict()}
            ])
            assets = ROOT / "third_party/DExplore/dexplore/data/assets/mjcf"
            geometry = TableClearance(
                obj_vertices(assets / "objects/airplane/airplane.obj", self.device),
                obj_vertices(assets / "objects/table/table.obj", self.device),
                getattr(task, "ball_size", 1.0),
            )
            ids = torch.arange(task.num_envs, device=self.device)
            observation = self.env_reset(ids)
            if self.get_batch_size(observation["obs"], 1) != task.num_envs:
                raise ValueError("player batch dimension mismatch")
            if observation["obs"].shape[-1] != 1442:
                raise ValueError("native observation dimension drift")
            tracker = __import__("src.task.CmResidual.physical_value_contract", fromlist=["HoldTracker"]).HoldTracker(task.num_envs, self.device)
            tracker.reset(ids, task._target_states[:, 2])
            history = HistoryBuffer(task.num_envs, self.device)
            previous_action = torch.zeros(task.num_envs, 18, device=self.device)
            motion, start_frame = task.data_id.clone(), task.start_times.clone()
            rest = task.hoi_refs[task.data_id, task.ref_index, 0, 108].clone()
            body_ids = task._contact_body_ids
            mass = torch.tensor([
                task.gym.get_actor_rigid_body_properties(env, handle)[0].mass
                for env, handle in zip(task.envs, task._target_handles)
            ], device=self.device)
            gravity = abs(task.sim_params.gravity.z)

            def normalized_contacts():
                return weight_normalized_contacts(
                    task._contact_forces[:, body_ids], task._tar_contact_forces,
                    mass, gravity)

            n = task.num_envs
            windows = 2 * args.windows_per_stratum
            zeros = lambda *shape, dtype=torch.float32: torch.zeros(
                n, windows, *shape, dtype=dtype, device=self.device)
            trigger = torch.full((n, windows), -1, dtype=torch.long, device=self.device)
            assignment = torch.full_like(trigger, -1)
            selected_index = torch.full_like(trigger, -1)
            window_stratum = torch.full_like(trigger, -1)
            steps = torch.zeros_like(trigger)
            count = torch.zeros(n, dtype=torch.long, device=self.device)
            elapsed = torch.full_like(count, -1)
            cooldown = torch.zeros_like(count)
            contact_run = torch.zeros_like(count)
            stratum_count = torch.zeros(n, 2, dtype=torch.long, device=self.device)
            ended = torch.zeros(n, dtype=torch.bool, device=self.device)
            selected_action = torch.zeros(n, 18, device=self.device)
            pre_state = zeros(55)
            pre_history_state = torch.zeros(n, windows, 16, 55, device=self.device)
            pre_history_action = torch.zeros(n, windows, 16, 18, device=self.device)
            pre_history_mask = torch.zeros(n, windows, 16, 1, device=self.device)
            pre_context = torch.zeros(n, windows, 435, device=self.device)
            next_context = torch.zeros_like(pre_context)
            candidates = torch.zeros(n, windows, 37, 18, device=self.device)
            direct_scores = torch.zeros(n, windows, 37, device=self.device)
            cm_scores = torch.zeros_like(direct_scores)
            direct_active = torch.zeros(n, windows, dtype=torch.bool, device=self.device)
            cm_active = torch.zeros_like(direct_active)
            future_state = zeros(10, 55)
            future_contact = torch.zeros(n, windows, 10, 2, dtype=torch.bool, device=self.device)
            future_clearance = zeros(10)
            future_done = torch.zeros(n, windows, 10, dtype=torch.bool, device=self.device)
            future_reward = zeros(10)
            actual_action = zeros(10, 18)
            initial_clearance = zeros()
            allocation_generator = torch.Generator(device=self.device).manual_seed(args.assignment_seed)
            candidate_generator = torch.Generator(device=self.device).manual_seed(args.assignment_seed + 1)
            reset = ids[:0]

            for tick in range(args.max_steps):
                if time.monotonic() - started > args.wall_seconds:
                    raise TimeoutError("decision-interface native budget")
                observation = self.env_reset(reset)
                if len(reset):
                    tracker.reset(reset, task._target_states[reset, 2])
                    history.reset(reset)
                    previous_action[reset] = 0
                    ended[reset] = False
                state = snapshot(task, tracker)
                history.append(state, previous_action)
                ctx = context(task, tracker)
                nctx = context(task, tracker, delta=1)
                contact, _ = normalized_contacts()
                contact_run = torch.where(contact.bool().all(-1), contact_run + 1, torch.zeros_like(contact_run))
                bank = self.candidates(observation)
                anchor = hold_target(task._dof_pos, task._pd_action_offset, task._pd_action_scale)
                cup = orientation_anchored_action(
                    bank[:, 1], anchor, task._dof_pos,
                    task._pd_action_offset, task._pd_action_scale)
                clearance = geometry.clearance(task._target_states, task._table_states)
                clear = (state[:, 38] - rest >= .03) & (clearance >= .002)
                eligible = ((~ended) & (elapsed < 0) & (cooldown <= 0) & (contact_run >= 3)
                            & (state[:, 38] - rest >= .005) & (tick >= 10)
                            & (tick <= args.max_steps - 12)
                            & (task.max_episode_length[task.data_id] - task.progress_buf > 11))
                eligible &= stratum_count[ids, clear.long()] < args.windows_per_stratum
                rows = eligible.nonzero().flatten()
                if len(rows):
                    hs, ha, hm = history.tensors()
                    if not torch.isfinite(hs[rows]).all():
                        bad = (~torch.isfinite(hs[rows])).nonzero(as_tuple=False)[0].tolist()
                        raise FloatingPointError(f"nonfinite decision history at {bad}; state range={hs[rows].amin().item()}..{hs[rows].amax().item()}")
                    if hs.device != torch.device(features.fk.device):
                        raise ValueError(f"decision history/FK device mismatch: {hs.device} vs {features.fk.device}")
                    panel, valid = make_candidates(cup[rows])
                    _, d_active, d_scores = teacher.labels(
                        "direct_q", cup[rows], hs[rows], ha[rows], hm[rows], ctx[rows], nctx[rows])
                    _, c_active, c_scores = teacher.labels(
                        "cm_value", cup[rows], hs[rows], ha[rows], hm[rows], ctx[rows], nctx[rows])
                    d_index = d_scores.masked_fill(~valid, -torch.inf).argmax(-1)
                    c_index = c_scores.masked_fill(~valid, -torch.inf).argmax(-1)
                    # The short-rollout score already subtracts ensemble spread;
                    # an inactive score falls back to the Cup baseline.
                    c_index = torch.where(c_active, c_index, torch.zeros_like(c_index))
                    random_index = torch.randint(0, 37, (len(rows),), device=self.device,
                                                 generator=candidate_generator)
                    random_index = torch.where(valid.gather(1, random_index[:, None]).squeeze(1),
                                               random_index, torch.zeros_like(random_index))
                    # Shuffle is a score-permutation control with the same panel.
                    permutation = torch.stack([
                        torch.randperm(37, device=self.device, generator=candidate_generator)
                        for _ in range(len(rows))
                    ])
                    shuffled_index = permutation.gather(
                        1, d_scores.gather(1, permutation).masked_fill(~valid.gather(1, permutation), -torch.inf).argmax(1)[:, None]
                    ).squeeze(1)
                    arm = torch.randint(len(ARMS), (len(rows),), device=self.device,
                                        generator=allocation_generator)
                    index_panel = torch.stack((torch.zeros_like(d_index), d_index, c_index,
                                               shuffled_index, random_index), 1)
                    chosen_index = index_panel[torch.arange(len(rows), device=self.device), arm]
                    chosen = panel[torch.arange(len(rows), device=self.device), chosen_index]
                    slot = count[rows]
                    trigger[rows, slot], assignment[rows, slot] = tick, arm
                    window_stratum[rows, slot] = clear[rows].long()
                    selected_index[rows, slot] = chosen_index
                    pre_state[rows, slot] = state[rows]
                    pre_history_state[rows, slot] = hs[rows]
                    pre_history_action[rows, slot] = ha[rows]
                    pre_history_mask[rows, slot] = hm[rows]
                    pre_context[rows, slot], next_context[rows, slot] = ctx[rows], nctx[rows]
                    candidates[rows, slot] = panel
                    direct_scores[rows, slot], cm_scores[rows, slot] = d_scores, c_scores
                    direct_active[rows, slot], cm_active[rows, slot] = d_active, c_active
                    initial_clearance[rows, slot] = clearance[rows]
                    selected_action[rows] = chosen
                    stratum_count[rows, clear[rows].long()] += 1
                    elapsed[rows] = 0
                live = (elapsed >= 0) & ~ended
                rows = live.nonzero().flatten()
                slot, offset = count[rows], elapsed[rows]
                action = bank[:, 4].clone()
                if len(rows):
                    action[rows] = selected_action[rows]
                    actual_action[rows, slot, offset] = action[rows]
                before_height = state[:, 38].clone()
                _, _, done, _ = self.env_step(self.env, action)
                done = done.bool().reshape(-1)
                after = snapshot(task, tracker)
                observed, _ = normalized_contacts()
                # Keep the tracker in lockstep with the training-time adapter.
                stable_reward, _ = tracker.step(task._target_states[:, 2], observed.bool().all(-1))
                after = snapshot(task, tracker)
                rows = live.nonzero().flatten()
                slot, offset = count[rows], elapsed[rows]
                future_state[rows, slot, offset] = after[rows]
                future_contact[rows, slot, offset] = observed[rows].bool()
                future_clearance[rows, slot, offset] = geometry.clearance(
                    task._target_states, task._table_states)[rows]
                future_done[rows, slot, offset] = done[rows]
                future_reward[rows, slot, offset] = (after[rows, 38] - before_height[rows]) + stable_reward[rows]
                steps[rows, slot] += 1
                elapsed[live] += 1
                complete = live & (elapsed == 10)
                count[complete] += 1
                elapsed[complete], cooldown[complete] = -1, 6
                cooldown = (cooldown - 1).clamp_min(0)
                ended |= done
                previous_action = action.clone()
                reset = done.nonzero().flatten()
                if tick % 100 == 0:
                    print(json.dumps(dict(tick=tick, windows=int((steps == 10).sum()),
                                          eligible=int(len(rows)))), flush=True)
                if (ended | (count >= windows)).all():
                    break

            valid_windows = steps == 10
            if ((steps > 0) & (~valid_windows | future_done.any(-1))).any():
                raise ValueError("incomplete or terminal decision window")
            if not valid_windows.any():
                raise ValueError("no complete decision windows")
            valid = valid_windows
            payload = dict(
                schema="ref2dex.cm_decision_interface.v1", experiment_id=args.experiment_id,
                panel_seed=args.seed, assignment_seed=args.assignment_seed,
                arm_names=list(ARMS), motion_id=motion[:, None].expand(-1, windows)[valid].cpu(),
                start_frame=start_frame[:, None].expand(-1, windows)[valid].cpu(),
                env_id=ids[:, None].expand(-1, windows)[valid].cpu(),
                stratum=window_stratum[valid].cpu(),
                trigger=trigger[valid].cpu(), assignment=assignment[valid].cpu(),
                selected_index=selected_index[valid].cpu(), propensity=torch.full((int(valid.sum()),), 0.2),
                state=pre_state[valid].cpu(), history_state=pre_history_state[valid].cpu(),
                history_action=pre_history_action[valid].cpu(), history_mask=pre_history_mask[valid].cpu(),
                context=pre_context[valid].cpu(), next_context=next_context[valid].cpu(),
                candidate_actions=candidates[valid].cpu(), direct_q_scores=direct_scores[valid].cpu(),
                cm_value_scores=cm_scores[valid].cpu(), direct_active=direct_active[valid].cpu(),
                cm_active=cm_active[valid].cpu(), initial_clearance=initial_clearance[valid].cpu(),
                actual_action=actual_action[valid].cpu(), future_state=future_state[valid].cpu(),
                future_contact=future_contact[valid].cpu(), future_clearance=future_clearance[valid].cpu(),
                future_done=future_done[valid].cpu(), future_reward=future_reward[valid].cpu(),
                rest_z=rest[:, None].expand(-1, windows)[valid].cpu(), control_dt=task.dt,
                gamma=.99, checkpoint=str(args.value_checkpoint.resolve()),
                checkpoint_sha256=sha(args.value_checkpoint), teacher_fingerprint=teacher_fingerprint,
                model_training=False, actor_training=False,
                local_utility_contract="10-step height delta + tracker stable reward; full policy success is not evaluated",
                assignment_contract="uniform random arm after full current-state candidate panel; no future filtering",
            )
            torch.save(payload, args.output / "records.pt")
            result = dict(run_status="COMPLETED", rows=int(valid.sum()),
                          arms=torch.bincount(assignment[valid], minlength=len(ARMS)).tolist(),
                          changed_from_cup=int((selected_index[valid] != 0).sum()),
                          cm_active=int(cm_active[valid].sum()), elapsed_seconds=time.monotonic() - started,
                          record_sha256=sha(args.output / "records.pt"),
                          model_training=False, actor_training=False)
            (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)

    return DecisionInterfacePlayer


def main():
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--output-dir", dest="output", type=Path, required=True)
    p.add_argument("--value-checkpoint", type=Path, required=True)
    p.add_argument("--experiment-id", default="P-20261002-cm-decision-interface")
    p.add_argument("--panel-seed", dest="seed", type=int, required=True)
    p.add_argument("--assignment-seed", type=int, required=True)
    p.add_argument("--windows-per-stratum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=650)
    p.add_argument("--wall-seconds", type=int, default=240)
    args, remaining = p.parse_known_args()
    base = (ROOT / "src/task/CmResidual/research/decision_interface/output").resolve()
    if base not in args.output.resolve().parents or args.output.is_symlink():
        raise ValueError("new owned decision-interface output required")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(run_status="STARTED", pid=os.getpid(), command=sys.argv,
                    gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    value_checkpoint=str(args.value_checkpoint.resolve()),
                    value_checkpoint_sha256=sha(args.value_checkpoint), model_training=False, actor_training=False)
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
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
        (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
