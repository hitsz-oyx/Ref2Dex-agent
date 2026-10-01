#!/usr/bin/env python3
"""Frozen actor-only first episodes on explicitly synthetic hold references."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_contact_response_probe import sha


def make_player(original, args, torch):
    from src.task.CmResidual.paired_evaluation import capture_rng, restore_rng, fingerprint
    from src.task.CmResidual.physical_value_contract import HoldTracker
    from src.task.CmResidual.physical_value_live import snapshot, contacts
    from src.task.CmResidual.dexplore_cm_geometry import DExploreCmv2GeometryBridge

    class HoldPlayer(original.EvalPlayer):
        @torch.no_grad()
        def run(self):
            begin = time.monotonic()
            torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            task = self.env.task
            task._enable_early_termination = False
            task._adaptive_kappa_enabled = False
            task._hybrid_init_prob = 1.
            if task.num_envs != 192 or abs(task.dt - 1 / 30) > 1e-8:
                raise ValueError('frozen batch/dt contract')
            device = task._dof_state.device
            ids = torch.arange(192, device=device)
            obs = self.env_reset(ids)
            self.get_batch_size(obs['obs'], 1)
            if self.is_rnn:
                self.init_rnn()
            if (task.start_times != 0).any() or task.progress_buf.any():
                raise ValueError('initial native frame must be zero')
            if [int((task.data_id == i).sum()) for i in range(3)] != [64] * 3:
                raise ValueError('motion coverage contract')
            generation = json.loads(args.references_manifest.read_text())
            by_name = {r['name']: r for r in generation['references']}
            phase_start = torch.empty(3, dtype=torch.long, device=device)
            phase_stop = phase_start.clone()
            loader_checks = []
            for motion, filename in enumerate(task.motion_file):
                record = by_name[Path(filename).name]
                if Path(filename).resolve() != Path(record['generated']).parent.resolve():
                    raise ValueError('actual loaded reference path drift')
                start, stop = record['plateau_reference_frames_inclusive']
                if stop - start + 1 != 90:
                    raise ValueError('plateau length drift')
                phase_start[motion], phase_stop[motion] = start, stop
                loaded = task.hoi_data_dict[motion]
                if int(task.max_episode_length[motion]) != record['generated_frames']:
                    raise ValueError('native reference length drift')
                maxima = {}
                for key in ('obj_pos_vel', 'obj_rot_vel', 'right_hand_pos_vel',
                            'right_dof_pos_vel', 'robot_dof_pos_vel',
                            'key_body_pos_vel', 'human_rot_vel'):
                    values = loaded[key][start:stop + 1]
                    maxima[key] = float(values.abs().max())
                    if not torch.isfinite(values).all() or maxima[key] != 0.:
                        raise ValueError(f'nonstationary ACTUAL native {key}')
                for key in ('obj_pos', 'obj_rot', 'robot_dof_pos'):
                    values = loaded[key][start:stop + 1]
                    if not torch.equal(values, values[:1].expand_as(values)):
                        raise ValueError('actual loaded plateau pose drift')
                loader_checks.append(dict(motion=motion, name=record['name'],
                    reference_frames=int(task.max_episode_length[motion]),
                    plateau_progress_inclusive=[start, stop], maximum_abs_velocity=maxima))
            actor_hash = fingerprint(self.model.state_dict())
            rms_hash = fingerprint(self.running_mean_std.state_dict()) if self.normalize_input else None
            rng = capture_rng()
            tracker = HoldTracker(192, device)
            tracker.reset(ids, task._target_states[:, 2])
            state = snapshot(task, tracker)
            asset = ROOT / 'third_party/DExplore/dexplore/data/assets'
            bridge = DExploreCmv2GeometryBridge(
                hand_urdf=asset / 'inspire_hand_new/inspire_hand_right.urdf',
                object_urdf=asset / 'mjcf/airplane.urdf', device=device, seed=42)
            gaps = []
            for group in ids.split(32):
                geometry = bridge.current(state[group, :18], state[group, 36:49])
                gaps.append(torch.cdist(geometry.hand_points, geometry.object_points).amin((1, 2)))
            gap = torch.cat(gaps)
            if not torch.isfinite(state).all() or not torch.isfinite(gap).all():
                raise ValueError('nonfinite initial state/geometry')
            motion_ids = task.data_id.clone()
            initial_height = task._target_states[:, 2].clone()
            torch.save(dict(root=task._root_states.cpu().clone(), dof=task._dof_state.cpu().clone(),
                rng=rng, state=state.cpu(), gap=gap.cpu(), motion=motion_ids.cpu(),
                initial_height=initial_height.cpu()), args.run_dir / 'initial.pt')
            restore_rng(rng)
            finished = torch.zeros(192, dtype=torch.bool, device=device)
            hold_run = torch.zeros(192, dtype=torch.long, device=device)
            max_run = hold_run.clone()
            phase_steps = hold_run.clone()
            steps = hold_run.clone()
            done_ids = torch.empty(0, dtype=torch.long, device=device)
            trace = {key: [] for key in ('height', 'contact', 'active', 'done', 'progress')}
            episodes = []
            for tick in range(900):
                if time.monotonic() - begin > args.wall_seconds:
                    raise TimeoutError('native hold feasibility budget')
                obs = self.env_reset(done_ids)
                if len(done_ids) and self.is_rnn:
                    for rnn in self.states:
                        rnn[:, done_ids, :] = 0
                active = ~finished
                before_progress = task.progress_buf.clone()
                if not torch.equal(before_progress[active], steps[active]):
                    raise ValueError('first episode progress/tick correspondence')
                action = self.get_action(obs, True).clamp(-1, 1)
                if not torch.isfinite(action).all():
                    raise ValueError('nonfinite actor action')
                obs, _, done, info = self.env_step(self.env, action)
                done = done.bool().reshape(-1)
                progress = task.progress_buf.clone()
                if not torch.equal(progress[active], before_progress[active] + 1):
                    raise ValueError('after-physics progress increment drift')
                height = task._target_states[:, 2].clone()
                pair = contacts(task)
                if not torch.isfinite(height[active]).all() or not torch.isfinite(pair[active]).all():
                    raise ValueError('nonfinite physical trajectory')
                in_phase = active & (progress >= phase_start[motion_ids]) & (progress <= phase_stop[motion_ids])
                held = in_phase & (height - initial_height >= .03) & pair.bool().all(-1)
                hold_run = torch.where(held, hold_run + 1, torch.zeros_like(hold_run))
                max_run = torch.maximum(max_run, hold_run)
                phase_steps += in_phase.long()
                steps += active.long()
                for key, value in dict(height=height, contact=pair, active=active,
                                       done=done, progress=progress).items():
                    trace[key].append(value.cpu().clone())
                for env_id in (done & active).nonzero().flatten().tolist():
                    if int(phase_steps[env_id]) != 90:
                        raise ValueError('first episode omitted plateau samples')
                    episodes.append(dict(environment=env_id, motion=int(motion_ids[env_id]),
                        steps=int(steps[env_id]), phase_steps=int(phase_steps[env_id]),
                        max_phase_hold_steps=int(max_run[env_id]), stable45=bool(max_run[env_id] >= 45),
                        retained75=bool(max_run[env_id] >= 75),
                        terminate=bool(info['terminate'].reshape(-1)[env_id])))
                finished |= done
                done_ids = done.nonzero().flatten()
                if finished.all():
                    break
                if tick % 100 == 0:
                    print(json.dumps(dict(tick=tick, complete=int(finished.sum()),
                        phase75=int((max_run >= 75).sum()))), flush=True)
            else:
                raise ValueError('incomplete first native episodes')
            if fingerprint(self.model.state_dict()) != actor_hash or (self.normalize_input and
                    fingerprint(self.running_mean_std.state_dict()) != rms_hash):
                raise ValueError('actor/RMS changed')
            torch.save(dict(**{k: torch.stack(v) for k, v in trace.items()},
                initial_height=initial_height.cpu(), motion=motion_ids.cpu(), episodes=episodes,
                control_dt=float(task.dt), phase_start=phase_start.cpu(), phase_stop=phase_stop.cpu(),
                loader_checks=loader_checks), args.run_dir / 'episodes.pt')
            result = dict(run_status='COMPLETED', complete_episodes=len(episodes), required_episodes=192,
                all_episodes_complete=bool(finished.all()), actor_and_rms_unchanged=True,
                actor_fingerprint=actor_hash, rms_fingerprint=rms_hash,
                native_loader_checks=loader_checks, actual_progress_increment_verified=True,
                wall_seconds=time.monotonic() - begin,
                initial_sha256=sha(args.run_dir / 'initial.pt'),
                episodes_sha256=sha(args.run_dir / 'episodes.pt'))
            (args.run_dir / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result), flush=True)
    return HoldPlayer


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--training-seed', type=int, required=True)
    parser.add_argument('--eval-seed', type=int, required=True)
    parser.add_argument('--references-manifest', type=Path, required=True)
    parser.add_argument('--wall-seconds', type=int, default=300)
    args, remaining = parser.parse_known_args()
    if args.run_dir.exists() or ROOT not in args.run_dir.resolve().parents:
        raise ValueError('unique isolated output required')
    checkpoint = Path(remaining[remaining.index('--checkpoint') + 1])
    if sha(checkpoint) != args.checkpoint_sha256:
        raise ValueError('actor drift')
    args.run_dir.mkdir(parents=True)
    begin = time.monotonic()
    manifest = dict(run_status='RUNNING', pid=os.getpid(), command=sys.argv)
    try:
        sys.path.insert(0, str(ROOT / 'third_party/DExplore/dexplore'))
        from isaacgym import gymtorch  # IsaacGym must initialize before torch.
        import torch
        import evaluate as original
        original.EvalPlayer = make_player(original, args, torch)
        sys.argv = [sys.argv[0], *remaining]
        original.main()
        if sha(checkpoint) != args.checkpoint_sha256:
            raise ValueError('actor modified')
        manifest['run_status'] = 'COMPLETED'
    except BaseException as error:
        manifest.update(run_status='FAILED', error=repr(error))
        raise
    finally:
        manifest['wall_seconds'] = time.monotonic() - begin
        (args.run_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
