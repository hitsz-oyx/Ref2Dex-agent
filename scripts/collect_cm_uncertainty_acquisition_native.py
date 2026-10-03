#!/usr/bin/env python3
"""Collect guarded native transitions selected by frozen Cm disagreement.

The Cm ensemble only scores the six frozen expert actions.  At eligible contact
states a high-disagreement proposal is randomized against the baseline, applied
for two steps, and then the baseline resumes.  This is data acquisition only:
there is no actor, reward, success-predictor, or Cm update in this collector.
"""
from __future__ import annotations

import argparse
import copy
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


def load_physical(checkpoint, device):
    import torch
    from torch import nn
    from src.task.CmResidual.physical_value_models import Features, OutcomeNetwork

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.physical_value_models.v1":
        raise ValueError("invalid physical-value checkpoint")
    if abs(float(payload["gamma"]) - .99) > 1e-8:
        raise ValueError("HF30 requires gamma=.99")
    stats = {key: value.to(device) for key, value in payload["stats"].items()}
    features = Features(**stats, device=device).to(device).eval()
    dynamics = nn.ModuleList([
        OutcomeNetwork(payload["context_dim"], 18, 53, 9300 + i).to(device)
        for i in range(3)
    ])
    for model, state in zip(dynamics, payload["dynamics"]):
        model.load_state_dict(state, strict=True)
    dynamics.eval().requires_grad_(False)
    return payload, features, dynamics


def score_uncertainty(features, dynamics, states, actions, mask, context, candidates):
    """Return disagreement [batch, candidates] over the predeclared 15-D target."""
    import torch
    from src.task.CmResidual.physical_value_models import decode_dynamics

    batch, count = candidates.shape[:2]
    history = features.history(states, actions, mask)
    hidden = [model.encode(history) for model in dynamics]
    current = states[:, -1]
    c = features.context(context)
    physical = []
    flat_candidates = candidates.reshape(batch * count, -1)
    flat_current = current[:, None].expand(-1, count, -1).reshape(batch * count, -1)
    flat_context = c[:, None].expand(-1, count, -1).reshape(batch * count, -1)
    for model, prefix in zip(dynamics, hidden):
        flat_hidden = prefix[:, None].expand(-1, count, -1).reshape(batch * count, -1)
        raw = model.from_hidden(flat_hidden, flat_context, flat_candidates)
        predicted, contact, reward, terminal = decode_dynamics(raw, flat_current, features)
        physical.append(torch.cat((
            predicted[:, 36:39] - flat_current[:, 36:39],
            predicted[:, 43:49], contact.sigmoid(), reward[:, None],
            terminal.sigmoid()[:, None]), dim=-1).reshape(batch, count, -1))
    return torch.stack(physical, dim=2).std(dim=2, unbiased=False).square().mean(-1).sqrt()


def make_player(original, args, torch, gymtorch):
    from src.task.CmResidual.contact_consequence import BASE_INDEX, EXECUTION_STEPS, HORIZON, local_outcomes
    from src.task.CmResidual.paired_evaluation import fingerprint
    from src.task.CmResidual.physical_value_contract import HistoryBuffer
    from src.task.CmResidual.physical_value_live import contacts, context, snapshot

    parent = build_player(original, args, torch, gymtorch)

    class AcquisitionPlayer(parent):
        @torch.no_grad()
        def run(self):
            started = time.monotonic()
            torch.set_num_threads(2)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            task = self.env.task
            if task.num_envs != 96 or self.is_rnn or abs(task.dt - 1 / 30) > 1e-8:
                raise ValueError("HF30 requires stateless 96-env 30Hz execution")
            task._hybrid_init_prob = 0.0
            task._enable_early_termination = False
            task._adaptive_kappa_enabled = False
            ids = torch.arange(task.num_envs, device=self.device)
            observation = self.env_reset(ids)
            if self.get_batch_size(observation["obs"], 1) != task.num_envs:
                raise ValueError("native batch dimension drift")
            payload, features, dynamics = load_physical(args.value_checkpoint, task._dof_pos.device)
            model_fingerprint = fingerprint([features.state_dict(), dynamics.state_dict()])
            history = HistoryBuffer(task.num_envs, self.device)
            tracker = __import__("src.task.CmResidual.physical_value_contract", fromlist=["HoldTracker"]).HoldTracker(task.num_envs, self.device)
            tracker.reset(ids, task._target_states[:, 2])
            previous_action = torch.zeros(task.num_envs, 18, device=self.device)
            motion, start_frame = task.data_id.clone(), task.start_times.clone()
            rest = task.hoi_refs[task.data_id, task.ref_index, 0, 108].clone()
            count = torch.zeros(task.num_envs, dtype=torch.long, device=self.device)
            elapsed = torch.full((task.num_envs,), -1, dtype=torch.long, device=self.device)
            cooldown = torch.zeros(task.num_envs, dtype=torch.long, device=self.device)
            contact_run = torch.zeros(task.num_envs, dtype=torch.long, device=self.device)
            ended = torch.zeros(task.num_envs, dtype=torch.bool, device=self.device)
            chosen_action = torch.zeros(task.num_envs, 18, device=self.device)
            proposed_index = torch.full((task.num_envs,), -1, dtype=torch.long, device=self.device)
            treated = torch.zeros(task.num_envs, dtype=torch.bool, device=self.device)
            trigger = torch.full((task.num_envs,), -1, dtype=torch.long, device=self.device)
            trigger_state = torch.zeros(task.num_envs, 55, device=self.device)
            trigger_history_state = torch.zeros(task.num_envs, 16, 55, device=self.device)
            trigger_history_action = torch.zeros(task.num_envs, 16, 18, device=self.device)
            trigger_history_mask = torch.zeros(task.num_envs, 16, 1, device=self.device)
            trigger_context = torch.zeros(task.num_envs, 435, device=self.device)
            candidate_panel = torch.zeros(task.num_envs, 6, 18, device=self.device)
            uncertainty_panel = torch.zeros(task.num_envs, 6, device=self.device)
            future_state = torch.zeros(task.num_envs, HORIZON, 55, device=self.device)
            future_contact = torch.zeros(task.num_envs, HORIZON, 2, dtype=torch.bool, device=self.device)
            future_done = torch.zeros(task.num_envs, HORIZON, dtype=torch.bool, device=self.device)
            reset = ids[:0]
            generator = torch.Generator(device=self.device).manual_seed(args.assignment_seed)

            for tick in range(args.max_steps):
                if time.monotonic() - started > args.wall_seconds:
                    raise TimeoutError("HF30 native acquisition budget")
                observation = self.env_reset(reset)
                if len(reset):
                    tracker.reset(reset, task._target_states[reset, 2])
                    history.reset(reset)
                    previous_action[reset] = 0
                    ended[reset] = False
                state = snapshot(task, tracker)
                history.append(state, previous_action)
                hs, ha, hm = history.tensors()
                ctx = context(task, tracker)
                contact = contacts(task)
                contact_run = torch.where(contact.bool().all(-1), contact_run + 1, torch.zeros_like(contact_run))
                bank = self.candidates(observation)
                baseline = bank[:, BASE_INDEX]
                eligible = ((~ended) & (elapsed < 0) & (cooldown <= 0) & (count < 1) &
                            (contact_run >= 3) & (state[:, 38] - rest >= .005) &
                            (tick >= 10) & (tick <= args.max_steps - HORIZON - 2) &
                            (task.max_episode_length[task.data_id] - task.progress_buf > HORIZON + 1))
                rows = eligible.nonzero().flatten()
                if len(rows):
                    candidates = bank[rows]
                    scores = score_uncertainty(features, dynamics, hs[rows], ha[rows], hm[rows], ctx[rows], candidates)
                    top = scores.argmax(-1)
                    active = top != BASE_INDEX
                    random_treatment = torch.rand(len(rows), device=self.device, generator=generator) < .5
                    do_treat = active & random_treatment
                    chosen = torch.where(do_treat[:, None], candidates[torch.arange(len(rows), device=self.device), top], baseline[rows])
                    chosen_action[rows], proposed_index[rows] = chosen, top
                    treated[rows], trigger[rows] = do_treat, tick
                    trigger_state[rows] = state[rows]
                    trigger_history_state[rows], trigger_history_action[rows], trigger_history_mask[rows] = hs[rows], ha[rows], hm[rows]
                    trigger_context[rows] = ctx[rows]
                    candidate_panel[rows], uncertainty_panel[rows] = candidates, scores
                    elapsed[rows] = 0
                live = (elapsed >= 0) & ~ended
                rows = live.nonzero().flatten()
                offset = elapsed[rows]
                action = baseline.clone()
                if len(rows):
                    use = offset < EXECUTION_STEPS
                    action[rows[use]] = chosen_action[rows[use]]
                    future_state[rows, offset] = state[rows]
                _, _, done, _ = self.env_step(self.env, action)
                done = done.bool().reshape(-1)
                after = snapshot(task, tracker)
                observed = contacts(task)
                tracker.step(task._target_states[:, 2], observed.bool().all(-1))
                after = snapshot(task, tracker)
                if len(rows):
                    future_state[rows, offset] = after[rows]
                    future_contact[rows, offset] = observed[rows].bool()
                    future_done[rows, offset] = done[rows]
                steps = live.nonzero().flatten()
                elapsed[live] += 1
                complete = live & (elapsed == HORIZON)
                count[complete] += 1
                elapsed[complete], cooldown[complete] = -1, 6
                cooldown = (cooldown - 1).clamp_min(0)
                ended |= done
                previous_action = action.clone()
                reset = done.nonzero().flatten()
                if tick % 100 == 0:
                    print(json.dumps(dict(tick=tick, windows=int((count > 0).sum()), eligible=int(len(rows)))), flush=True)
                if bool((ended | (count >= 1)).all()):
                    break

            valid = count >= 1
            if int(valid.sum()) < args.min_windows:
                raise ValueError(f"only {int(valid.sum())} complete windows; need {args.min_windows}")
            if future_done[valid].any():
                raise ValueError("terminal-contaminated acquisition window")
            if fingerprint([features.state_dict(), dynamics.state_dict()]) != model_fingerprint:
                raise ValueError("Cm bundle changed during acquisition")
            outcomes = local_outcomes(future_state[valid, :, :49], future_contact[valid].all(-1),
                                      trigger_state[valid, 38], rest[valid])
            targeted = treated[valid]
            if int(targeted.sum()) < args.min_targeted:
                raise ValueError(f"only {int(targeted.sum())} targeted windows; need {args.min_targeted}")
            groups = {f"{int(m)}/{int(s)}" for m, s in zip(motion[valid][targeted].cpu(), start_frame[valid][targeted].cpu())}
            if len(groups) < args.min_groups:
                raise ValueError(f"only {len(groups)} targeted motion/start groups; need {args.min_groups}")
            target_u = uncertainty_panel[valid][targeted].gather(1, proposed_index[valid][targeted, None]).squeeze(1)
            base_u = uncertainty_panel[valid][targeted, BASE_INDEX]
            target_contact_loss = float(outcomes["contact_loss"][targeted].mean())
            base_contact_loss = float(outcomes["contact_loss"][~targeted].mean()) if (~targeted).any() else float("nan")
            target_drop = float(outcomes["drop"][targeted].float().mean())
            base_drop = float(outcomes["drop"][~targeted].float().mean()) if (~targeted).any() else float("nan")
            result = dict(run_status="COMPLETED", experiment_id=args.experiment_id,
                          rows=int(valid.sum()), targeted_windows=int(targeted.sum()),
                          baseline_windows=int((~targeted).sum()), targeted_groups=len(groups),
                          target_uncertainty=float(target_u.mean()), baseline_uncertainty=float(base_u.mean()),
                          uncertainty_gain=float((target_u - base_u).mean()),
                          targeted_contact_loss=target_contact_loss, baseline_contact_loss=base_contact_loss,
                          targeted_drop_rate=target_drop, baseline_drop_rate=base_drop,
                          contact_loss_delta=target_contact_loss - base_contact_loss,
                          drop_rate_delta=target_drop - base_drop,
                          gates=dict(targeted_windows=int(targeted.sum()) >= args.min_targeted,
                                     targeted_groups=len(groups) >= args.min_groups,
                                     uncertainty_higher=float((target_u - base_u).mean()) > 0,
                                     contact_loss_not_worse=(target_contact_loss - base_contact_loss) <= .05,
                                     drop_not_worse=(target_drop - base_drop) <= .05),
                          elapsed_seconds=time.monotonic() - started,
                          record_sha256="pending")
            result["probe_label"] = "PROMISING" if all(result["gates"].values()) else "UNPROMISING"
            saved = dict(schema="ref2dex.cm_uncertainty_acquisition_native.v1",
                         experiment_id=args.experiment_id, panel_seed=args.seed,
                         assignment_seed=args.assignment_seed, checkpoint=str(args.value_checkpoint.resolve()),
                         checkpoint_sha256=sha(args.value_checkpoint), motion_id=motion[valid].cpu(),
                         start_frame=start_frame[valid].cpu(), trigger=trigger[valid].cpu(),
                         treated=targeted.cpu(), proposed_index=proposed_index[valid].cpu(),
                         state=trigger_state[valid].cpu(), history_state=trigger_history_state[valid].cpu(),
                         history_action=trigger_history_action[valid].cpu(), history_mask=trigger_history_mask[valid].cpu(),
                         context=trigger_context[valid].cpu(), candidate_actions=candidate_panel[valid].cpu(),
                         uncertainty=uncertainty_panel[valid].cpu(), actual_action=chosen_action[valid].cpu(),
                         future_state=future_state[valid].cpu(), future_contact=future_contact[valid].cpu(),
                         future_done=future_done[valid].cpu(), outcome={key: value.cpu() for key, value in outcomes.items()},
                         propensity=torch.full((int(valid.sum()),), .5), actor_training=False,
                         model_training=False, success_predictor=False,
                         acquisition_contract="top frozen-Cm disagreement candidate versus BASE_INDEX baseline; candidate for two steps then baseline")
            torch.save(saved, args.output / "records.pt")
            result["record_sha256"] = sha(args.output / "records.pt")
            (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)

    return AcquisitionPlayer


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--output-dir", dest="output", type=Path, required=True)
    parser.add_argument("--value-checkpoint", type=Path, required=True)
    parser.add_argument("--experiment-id", default="P-20261003-cm-uncertainty-acquisition-native")
    parser.add_argument("--panel-seed", dest="seed", type=int, required=True)
    parser.add_argument("--assignment-seed", type=int, required=True)
    parser.add_argument("--min-windows", type=int, default=32)
    parser.add_argument("--min-targeted", type=int, default=32)
    parser.add_argument("--min-groups", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=650)
    parser.add_argument("--wall-seconds", type=int, default=600)
    args, remaining = parser.parse_known_args()
    base = (ROOT / "src/task/CmResidual/research/uncertainty_acquisition/output").resolve()
    if base not in args.output.resolve().parents or args.output.is_symlink():
        raise ValueError("new owned uncertainty-acquisition output required")
    args.output.mkdir(parents=True, exist_ok=False)
    args.reference = None
    manifest = dict(run_status="STARTED", pid=os.getpid(), command=sys.argv,
                    gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    value_checkpoint=str(args.value_checkpoint.resolve()),
                    value_checkpoint_sha256=sha(args.value_checkpoint), actor_training=False, model_training=False)
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
