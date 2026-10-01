#!/usr/bin/env python3
"""Compact cold-start physics replay with a fixed single action intervention."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_paired_physical_value_environment import sha


def make_player(original, args, torch, gymtorch):
    from src.task.CmResidual.contact_response import pulse_action, trigger_steps
    from src.task.CmResidual.paired_evaluation import (
        cpu_copy, fingerprint, physical_property_value, restore_initial, restore_rng,
    )
    from src.task.CmResidual.physical_value_contract import HoldTracker
    from src.task.CmResidual.physical_value_live import contacts, snapshot

    class ResponsePlayer(original.EvalPlayer):
        @torch.no_grad()
        def run(self):
            begin = time.monotonic()
            torch.set_num_threads(2)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            task = self.env.task
            task._enable_early_termination = False
            task._adaptive_kappa_enabled = False
            task._hybrid_init_prob = 1.
            if task.num_envs != 96 or abs(task.dt - 1 / 30) > 1e-8:
                raise ValueError('native time/environment mismatch')
            ids = torch.arange(task.num_envs, device=self.device)
            obs = self.env_reset(ids)
            self.get_batch_size(obs['obs'], 1)
            if self.is_rnn:
                self.init_rnn()
            properties = []
            for env in task.envs:
                actors = []
                for actor in range(task.gym.get_actor_count(env)):
                    actors.append(dict(
                        name=task.gym.get_actor_name(env, actor),
                        dof=cpu_copy(task.gym.get_actor_dof_properties(env, actor)),
                        rigid_body=physical_property_value(task.gym.get_actor_rigid_body_properties(env, actor)),
                        rigid_shape=physical_property_value(task.gym.get_actor_rigid_shape_properties(env, actor)),
                    ))
                properties.append(actors)
            saved = torch.load(args.initial, map_location='cpu', weights_only=False)
            reference = torch.load(args.trace, map_location='cpu', weights_only=False)
            initial_hash = fingerprint(saved)
            if reference['initial_state_fingerprint'] != initial_hash:
                raise ValueError('trace/initial mismatch')
            obs = restore_initial(task, self, saved, gymtorch.unwrap_tensor, properties)
            triggers = trigger_steps(reference['state_after'], reference['active'])
            length = int(triggers.max()) + 30
            if reference['done'][:length].any():
                raise ValueError('reference terminates inside intervention window')
            if abs(float(task._pd_action_scale[2]) - 1.) > 1e-8:
                raise ValueError('wrist-z native unit scale mismatch')
            tracker = HoldTracker(task.num_envs, self.device)
            tracker.reset(ids, task._target_states[:, 2])
            model_before = fingerprint(self.model.state_dict())
            rows = {k: [] for k in ('state_before', 'state_after', 'action')}
            for tick in range(length):
                if time.monotonic() - begin > args.wall_seconds:
                    raise TimeoutError('native physics replay wall limit')
                restore_rng(reference['rng'][tick]['before_reset'])
                self.env_reset(torch.empty(0, dtype=torch.long, device=self.device))
                restore_rng(reference['rng'][tick]['before_action'])
                action = pulse_action(reference['action'][tick].to(self.device), triggers, tick, args.amplitude)
                rows['state_before'].append(snapshot(task, tracker).cpu())
                rows['action'].append(action.cpu().clone())
                restore_rng(reference['rng'][tick]['before_physics'])
                _, _, done, _ = self.env_step(self.env, action)
                if done.any():
                    raise ValueError('intervention caused termination; window invalid')
                tracker.step(task._target_states[:, 2], contacts(task).bool().all(-1))
                after = snapshot(task, tracker).cpu()
                if not torch.isfinite(after).all():
                    raise FloatingPointError('nonfinite physical response')
                rows['state_after'].append(after)
            if fingerprint(self.model.state_dict()) != model_before:
                raise ValueError('actor update')
            data = {k: torch.stack(v) for k, v in rows.items()}
            data.update(schema='ref2dex.contact_response.v1', triggers=triggers,
                        amplitude=args.amplitude, initial_state_fingerprint=initial_hash,
                        motion_id=saved['tensors']['data_id'].long(), control_dt=float(task.dt))
            path = args.run_dir / 'response.pt'
            torch.save(data, path)
            result = dict(run_status='COMPLETED', amplitude=args.amplitude,
                          complete_response_windows=96, ticks=length,
                          initial_state_fingerprint=initial_hash, output_sha256=sha(path),
                          contact_proxy_boundary='native independent hand/object force proxies, not pairwise contacts',
                          replay='reference actions/RNG; one fixed wrist-z pulse; actor not used for control',
                          wall_seconds=time.monotonic() - begin)
            (args.run_dir / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result), flush=True)
    return ResponsePlayer


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--amplitude', type=float, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--wall-seconds', type=int, default=150)
    args, remaining = parser.parse_known_args()
    if args.amplitude not in (-.01, 0., .01):
        raise ValueError('pulse amplitude is frozen')
    if ROOT not in args.run_dir.resolve().parents or args.run_dir.exists():
        raise ValueError('new output must be unique and inside this worktree')
    checkpoint = Path(remaining[remaining.index('--checkpoint') + 1])
    if sha(checkpoint) != args.checkpoint_sha256:
        raise ValueError('checkpoint drift')
    args.run_dir.mkdir(parents=True)
    inputs = {str(p): sha(p) for p in (args.initial, args.trace, checkpoint)}
    manifest = dict(run_status='RUNNING', pid=os.getpid(), command=sys.argv,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    inputs=inputs, created_at=datetime.now(timezone.utc).isoformat(),
                    gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), no_training=True)
    begin = time.monotonic()
    try:
        sys.path.insert(0, str(ROOT / 'third_party/DExplore/dexplore'))
        from isaacgym import gymtorch
        import torch
        import evaluate as original
        original.EvalPlayer = make_player(original, args, torch, gymtorch)
        sys.argv = [sys.argv[0], *remaining]
        original.main()
        if any(sha(Path(p)) != h for p, h in inputs.items()):
            raise ValueError('input modified during replay')
        manifest.update(run_status='COMPLETED', inputs_unchanged=True)
    except BaseException as error:
        manifest.update(run_status='FAILED', error=repr(error))
        raise
    finally:
        manifest['wall_seconds'] = time.monotonic() - begin
        (args.run_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
