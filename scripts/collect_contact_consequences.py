#!/usr/bin/env python3
"""Cold-start, prefix-replayed physical interventions by frozen experts."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha


def build_player(original, args, torch, gymtorch):
    from src.task.CmResidual.paired_evaluation import (
        capture_initial, capture_rng, cpu_copy, fingerprint, physical_property_value,
        restore_initial, restore_rng,
    )
    from src.task.CmResidual.physical_value_live import contacts
    from src.task.CmResidual.contact_consequence import (
        SCHEMA, HISTORY, HORIZON, EXECUTION_STEPS, BASE_INDEX, local_outcomes, prefix_errors,
    )
    route = json.loads((ROOT/'src/task/CmResidual/configs/hf02_temporal_canonical_route.json').read_text())

    def properties(task):
        return [[dict(name=task.gym.get_actor_name(env, actor),
                      dof=cpu_copy(task.gym.get_actor_dof_properties(env, actor)),
                      rigid_body=physical_property_value(task.gym.get_actor_rigid_body_properties(env, actor)),
                      rigid_shape=physical_property_value(task.gym.get_actor_rigid_shape_properties(env, actor)))
                 for actor in range(task.gym.get_actor_count(env))] for env in task.envs]

    class ConsequencePlayer(original.EvalPlayer):
        def restore(self, filename):
            super().restore(filename)
            self.frozen_experts = []
            target_keys = set(self.model.state_dict())
            for name in route['candidate_experts']:
                spec = route['experts'][name]
                path = (ROOT/spec['checkpoint']).resolve()
                if sha(path) != spec['sha256']:
                    raise ValueError('expert hash changed: '+name)
                payload = torch.load(path, map_location=self.device, weights_only=False)
                model = copy.deepcopy(self.model)
                state = {}
                for key, value in payload['model'].items():
                    bare = key.removeprefix('_orig_mod.')
                    state[bare if bare in target_keys else '_orig_mod.'+bare] = value
                model.load_state_dict(state, strict=True)
                model.eval().requires_grad_(False)
                rms = copy.deepcopy(self.running_mean_std)
                rms.load_state_dict(payload['running_mean_std'], strict=True)
                rms.eval()
                self.frozen_experts.append((model, rms))

        def candidates(self, observation):
            model, rms = self.model, self.running_mean_std
            try:
                result = []
                for self.model, self.running_mean_std in self.frozen_experts:
                    result.append(self.get_action(observation, True).clamp(-1, 1).clone())
                return torch.stack(result, dim=1)
            finally:
                self.model, self.running_mean_std = model, rms

        @torch.no_grad()
        def run(self):
            started = time.monotonic()
            torch.set_num_threads(2)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            task = self.env.task
            task._enable_early_termination = False
            task._adaptive_kappa_enabled = False
            # Seeded reference-frame initialization makes distinct contact
            # states; it does not restore a hot solver or replay old HF08.
            task._hybrid_init_prob = 0.0
            if abs(task.dt-1/30) > 1e-8 or task.num_envs != 96:
                raise ValueError('time/environment contract changed')
            if self.is_rnn:
                raise ValueError('this expert pool must be stateless; no shared RNN across experts')
            ids = torch.arange(task.num_envs, device=self.device)
            observation = self.env_reset(ids)
            props = properties(task)
            ref = torch.load(args.reference, map_location='cpu', weights_only=False) if args.reference else None
            initial = ref['initial'] if ref else capture_initial(task, self, observation, props)
            restore_initial(task, self, initial, gymtorch.unwrap_tensor, props)
            initial_hash = fingerprint(initial)
            before_hash = fingerprint([dict(model=m.state_dict(), rms=r.state_dict()) for m, r in self.frozen_experts])
            n = task.num_envs
            trigger = (ref['trigger'].to(self.device) if ref else torch.full((n,), -1, dtype=torch.long, device=self.device))
            window_steps = torch.zeros(n, dtype=torch.long, device=self.device)
            future = torch.zeros(n, HORIZON, 49, device=self.device)
            future_contact = torch.zeros(n, HORIZON, 2, dtype=torch.bool, device=self.device)
            future_action = torch.zeros(n, HORIZON, 18, device=self.device)
            future_done = torch.zeros(n, HORIZON, dtype=torch.bool, device=self.device)
            trigger_state = torch.zeros(n, 49, device=self.device)
            trigger_obs = torch.zeros(n, observation['obs'].shape[-1], device=self.device)
            trigger_contact = torch.zeros(n, 2, device=self.device)
            candidate_actions = torch.zeros(n, 6, 18, device=self.device)
            history = torch.zeros(n, HISTORY, 69, device=self.device)
            trigger_history = torch.zeros_like(history)
            previous_action = torch.zeros(n, 18, device=self.device)
            first_ended = torch.zeros(n, dtype=torch.bool, device=self.device)
            reset_ids = torch.empty(0, dtype=torch.long, device=self.device)
            motion = task.data_id.clone(); start_frame = task.start_times.clone()
            rest_z = task.hoi_refs[task.data_id, task.ref_index, 0, 108].clone()
            prefix_actions = []; prefix_states = []; rng = []
            horizon = len(ref['action']) if ref else args.max_steps
            quotas = [args.max_states//3 + int(i < args.max_states%3) for i in range(3)]
            for tick in range(horizon):
                if time.monotonic()-started > args.wall_seconds:
                    raise TimeoutError('native consequence collection budget')
                if ref: restore_rng(ref['rng'][tick]['reset'])
                before_reset = capture_rng()
                observation = self.env_reset(reset_ids)
                state = torch.cat((task._dof_pos.clone(), task._dof_vel.clone(), task._target_states.clone()), -1)
                contact = contacts(task)
                history = torch.cat((history[:, 1:], torch.cat((state, contact, previous_action), -1)[:, None]), 1)
                if ref: restore_rng(ref['rng'][tick]['action'])
                before_action = capture_rng()
                if ref:
                    current_trigger = trigger == tick
                    base = self.get_action(observation, True).clamp(-1, 1).clone()
                    actions = ref['action'][tick].to(self.device).clone()
                    elapsed = tick-trigger
                    active = (trigger >= 0) & (elapsed >= 0) & (elapsed < HORIZON)
                    # Base replay uses the exact same cached two-step proposal
                    # as the base arm; after two steps each branch is closed-loop.
                    candidate = (elapsed < EXECUTION_STEPS) & active
                    if current_trigger.any():
                        candidate_actions[current_trigger] = ref['candidate_actions'][current_trigger.cpu()].to(self.device)
                    actions[active] = base[active]
                    actions[candidate] = candidate_actions[candidate, args.arm]
                else:
                    candidates = self.candidates(observation)
                    base = candidates[:, BASE_INDEX]
                    eligible = (contact.bool().all(-1) & ~first_ended & (trigger < 0) & (tick >= HISTORY) &
                                (task.max_episode_length[task.data_id]-task.progress_buf > HORIZON+1))
                    current_trigger = torch.zeros(n, dtype=torch.bool, device=self.device)
                    for motion_id in range(3):
                        remaining = quotas[motion_id]-int(((trigger >= 0) & (motion == motion_id)).sum())
                        chosen = (eligible & (motion == motion_id)).nonzero().flatten()[:max(remaining, 0)]
                        trigger[chosen] = tick; current_trigger[chosen] = True
                    candidate_actions[current_trigger] = candidates[current_trigger]
                    elapsed = tick-trigger
                    active = (trigger >= 0) & (elapsed >= 0) & (elapsed < HORIZON)
                    actions = base.clone()
                    candidate = (elapsed < EXECUTION_STEPS) & active
                    actions[candidate] = candidate_actions[candidate, BASE_INDEX]
                trigger_state[current_trigger] = state[current_trigger]
                trigger_obs[current_trigger] = observation['obs'][current_trigger]
                trigger_contact[current_trigger] = contact[current_trigger]
                trigger_history[current_trigger] = history[current_trigger]
                if not torch.isfinite(actions).all(): raise ValueError('nonfinite actual action')
                if ref: restore_rng(ref['rng'][tick]['physics'])
                before_physics = capture_rng()
                applied = actions.clone()
                _, _, done, _ = self.env_step(self.env, actions)
                after = torch.cat((task._dof_pos.clone(), task._dof_vel.clone(), task._target_states.clone()), -1)
                after_contact = contacts(task).bool()
                done = done.bool().reshape(-1)
                rows = active.nonzero().flatten()
                slots = elapsed[rows].long()
                future[rows, slots] = after[rows]
                future_contact[rows, slots] = after_contact[rows]
                future_action[rows, slots] = applied[rows]
                future_done[rows, slots] = done[rows]
                window_steps[rows] += 1
                first_ended |= done
                reset_ids = done.nonzero().flatten()
                previous_action = applied.clone()
                prefix_actions.append(applied.cpu()); prefix_states.append(state.cpu())
                rng.append(dict(reset=before_reset, action=before_action, physics=before_physics))
                if tick % 100 == 0:
                    print(json.dumps(dict(tick=tick, triggers=int((trigger>=0).sum()), windows=int((window_steps==HORIZON).sum()))), flush=True)
                if not ref and int((trigger>=0).sum()) == args.max_states and bool((window_steps[trigger>=0] == HORIZON).all()):
                    break
                if not ref and bool(first_ended.all()): break
            selected = trigger >= 0
            if int(selected.sum()) < 24 or (window_steps[selected] != HORIZON).any() or future_done[selected].any():
                raise ValueError('incomplete contact panel or first-episode window')
            if fingerprint([dict(model=m.state_dict(), rms=r.state_dict()) for m, r in self.frozen_experts]) != before_hash:
                raise ValueError('frozen expert or normalization updated')
            valid, errors = prefix_errors(trigger_state[selected], ref['trigger_state'][selected.cpu()].to(self.device)) if ref else (torch.ones(int(selected.sum()), dtype=torch.bool, device=self.device), {})
            if ref:
                history_valid, history_errors = prefix_errors(
                    trigger_history[selected, :, :49], ref['history'][selected.cpu(), :, :49].to(self.device))
                valid &= history_valid.all(-1) & trigger_contact[selected].bool().all(-1)
                errors['history'] = history_errors
            outcome = local_outcomes(future[selected], future_contact[selected].all(-1), trigger_state[selected, 38], rest_z[selected])
            payload = dict(schema=SCHEMA, initial=initial, initial_fingerprint=initial_hash,
                           arm=args.arm if ref else BASE_INDEX, repeat=args.repeat,
                           initial_seed=args.seed, trigger=trigger.cpu(), selected=selected.cpu(),
                           motion_id=motion.cpu(), start_frame=start_frame.cpu(), rest_z=rest_z.cpu(),
                           trigger_state=trigger_state.cpu(), trigger_observation=trigger_obs.cpu(),
                           trigger_contact=trigger_contact.cpu(), history=trigger_history.cpu(),
                           candidate_actions=candidate_actions.cpu(), future_state=future.cpu(),
                           future_contact=future_contact.cpu(), future_action=future_action.cpu(),
                           future_done=future_done.cpu(), window_steps=window_steps.cpu(),
                           action=torch.stack(prefix_actions), state=torch.stack(prefix_states), rng=rng,
                           outcome=cpu_copy(outcome), paired_valid=valid.cpu(),
                           prefix_errors=cpu_copy(errors), frozen_experts_unchanged=True,
                           contact_definition='native hand force>0.1 AND object force>0.1; pair identity unavailable',
                           execution='cached normalized candidate2steps; closed-loop base8steps',
                           pairing='same cold state and cached pretrigger actions; tolerance-checked trigger, no hot restore')
            if ref:
                if fingerprint(rng) != fingerprint(ref['rng']): raise ValueError('common RNG schedule drift')
                if initial_hash != ref['initial_fingerprint']: raise ValueError('initial state mismatch')
                for tick in range(len(prefix_actions)):
                    pre = (trigger < 0) | (trigger > tick)
                    if not torch.equal(payload['action'][tick, pre.cpu()], ref['action'][tick, pre.cpu()]):
                        raise ValueError('pretrigger action changed')
                expected = candidate_actions[selected, args.arm][:, None].expand(-1, EXECUTION_STEPS, -1)
                if not torch.equal(future_action[selected, :EXECUTION_STEPS], expected):
                    raise ValueError('candidate not actually executed')
            torch.save(payload, args.output/'records.pt')
            result = dict(run_status='COMPLETED', states=int(selected.sum()), arm=payload['arm'], repeat=args.repeat,
                          paired_valid_count=int(valid.sum()), full_initial_fingerprint=initial_hash,
                          mean_supported_lift_mm=float(outcome['supported_lift_mm'].mean()),
                          mean_contact_fraction=float(outcome['contact_fraction'].mean()),
                          drop_eligible=int(outcome['drop_eligible'].sum()), drops=int(outcome['drop'].sum()),
                          elapsed_seconds=time.monotonic()-started, record_sha256=sha(args.output/'records.pt'))
            (args.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result), flush=True)
    return ConsequencePlayer


def main():
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument('--output-dir', dest='output', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--arm', type=int, choices=range(6), default=4)
    p.add_argument('--repeat', type=int, default=0)
    p.add_argument('--panel-seed', dest='seed', type=int, required=True)
    p.add_argument('--max-states', type=int, default=32)
    p.add_argument('--max-steps', type=int, default=650)
    p.add_argument('--wall-seconds', type=int, default=240)
    args, remaining = p.parse_known_args()
    base = (ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    if base not in args.output.resolve().parents: raise ValueError('output outside owned task root')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(run_status='STARTED', pid=os.getpid(), command=sys.argv,
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                    created_at=datetime.now(timezone.utc).isoformat(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    actor_training=False, cm_training=False)
    (args.output/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    started = time.monotonic()
    try:
        sys.path.insert(0, str(ROOT/'third_party/DExplore/dexplore'))
        from isaacgym import gymtorch
        import torch
        import evaluate as original
        original.EvalPlayer = build_player(original, args, torch, gymtorch)
        sys.argv = [sys.argv[0], *remaining]
        original.main()
        manifest['run_status'] = 'COMPLETED'
    except BaseException as error:
        manifest.update(run_status='FAILED', error=repr(error)); raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic()-started
        (args.output/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__': main()
