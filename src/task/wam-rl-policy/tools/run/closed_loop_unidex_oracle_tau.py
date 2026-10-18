#!/usr/bin/env python3
"""Closed-loop UniDex student with online teacher-forked oracle tau.

One persistent single-environment student simulator is advanced only by the
student's own actions.  When the student is tau-conditioned, every tick asks a
fresh ``oracle_tau_query.py`` process to reset with the same seed, replay the
student's applied action prefix, and roll the frozen teacher forward.  The
student never executes teacher actions; the oracle process is only a query.

The per-tick replayed-observation error is recorded.  A query above tolerance
invalidates the fork and stops the run by default.
"""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'run'))
from probe_unidex_oracle_tau import DEFAULT_OBJECT, load_replay_helpers, write_json

P2_SLICES = ((0, 44), (66, 131), (161, 191))


def p2_from_obs(obs):
    return np.concatenate([obs[lo:hi] for lo, hi in P2_SLICES])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subset', type=Path, required=True)
    p.add_argument('--vendor', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--variant', choices=('p2', 'p2_tau'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--oracle-script', type=Path,
                   default=Path(__file__).resolve().parent / 'oracle_tau_query.py')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--ticks', type=int, default=120)
    p.add_argument('--horizon', type=int, default=8)
    p.add_argument('--max-replay-error', type=float, default=5e-2)
    p.add_argument('--object-code', default=DEFAULT_OBJECT)
    p.add_argument('--scale', type=float, default=.08)
    p.add_argument('--gpu-seconds', type=int, default=1800)
    args = p.parse_args()
    for key in ('subset', 'vendor', 'assets', 'policy', 'output', 'oracle_script'):
        setattr(args, key, getattr(args, key).resolve())
    if args.output == Path('/tmp') or Path('/tmp') in args.output.parents:
        raise ValueError('outputs must remain in the project workspace')
    if 'float' not in np.__dict__:
        np.float = float
    from isaacgym import gymapi, gymutil
    import torch
    import yaml
    from isaacgym.torch_utils import quat_apply

    torch.set_num_threads(2)
    replay = load_replay_helpers()
    vendor = args.vendor.resolve()
    assets = args.assets.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'tmp').mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(vendor / 'dexgrasp'))
    ShadowHandGrasp = replay.task_class(vendor, args.output)

    cfg = yaml.safe_load((vendor / 'dexgrasp/cfg/shadow_hand_grasp.yaml').read_text())
    cfg['env'].update(numEnvs=1, random_time=False, printNumSuccesses=False,
                      object_code_dict={args.object_code: [args.scale]})
    cfg['env']['asset']['assetRoot'] = str(assets)
    cfg['graphics_device_id'] = -1
    sim = gymapi.SimParams()
    sim.dt = 1 / 60
    sim.num_client_threads = 0
    sim.physx.solver_type = 1
    sim.physx.num_position_iterations = 4
    sim.physx.num_velocity_iterations = 0
    sim.physx.num_threads = 4
    sim.physx.use_gpu = True
    sim.physx.num_subscenes = 0
    sim.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
    sim.use_gpu_pipeline = True
    gymutil.parse_sim_config(cfg['sim'], sim)
    work = assets.parent / 'dexgrasp'
    work.mkdir(exist_ok=True)
    os.chdir(work)
    task = ShadowHandGrasp(cfg, sim, gymapi.SIM_PHYSX, 'cuda', 0, True)

    names = ['robot0:palm', 'robot0:thhub', 'robot0:ffknuckle', 'robot0:mfknuckle',
             'robot0:rfknuckle', 'robot0:lfknuckle', 'robot0:thdistal', 'robot0:ffdistal',
             'robot0:mfdistal', 'robot0:rfdistal', 'robot0:lfdistal']
    indices = [task.gym.find_actor_rigid_body_index(task.envs[0], task.shadow_hands[0],
                                                     n, gymapi.DOMAIN_ACTOR) for n in names]
    if any(i < 0 for i in indices):
        raise ValueError('missing hand landmark indices')
    tip_offsets = torch.tensor([[0, 0, .0275]] + [[0, 0, .026]] * 4,
                               dtype=torch.float32, device='cuda:0')

    def hand_points():
        rigid = task.rigid_body_states[:, indices]
        points = rigid[:, :, :3].clone()
        points[:, 6:] += quat_apply(
            rigid[:, 6:, 3:7].reshape(-1, 4),
            tip_offsets[None].expand(1, -1, -1).reshape(-1, 3)).reshape(1, 5, 3)
        return points

    def sim_step(action):
        task.pre_physics_step(action)
        for _ in range(task.control_freq_inv):
            task.gym.simulate(task.sim)
            task.gym.fetch_results(task.sim, True)
        task.post_physics_step()

    def seed_runtime(seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def reset_envs():
        seed_runtime(args.seed)
        task.reset_buf[:] = 1
        task.step(torch.zeros((1, 24), device='cuda:0'), -1)
        if task.reset_buf.any():
            raise ValueError('reset left reset flags set')
        if not torch.isfinite(task.obs_buf).all():
            raise ValueError('nonfinite reset observation')

    checkpoint = torch.load(args.policy, map_location='cuda:0', weights_only=False)
    if checkpoint['variant'] != args.variant:
        raise ValueError(f"checkpoint variant {checkpoint['variant']} != {args.variant}")
    model = torch.nn.Sequential(
        torch.nn.Linear(checkpoint['input_dim'], checkpoint['hidden']), torch.nn.ELU(),
        torch.nn.Linear(checkpoint['hidden'], checkpoint['hidden']), torch.nn.ELU(),
        torch.nn.Linear(checkpoint['hidden'], 24)).to('cuda:0').eval()
    model.load_state_dict(checkpoint['state_dict'])
    mean = torch.tensor(checkpoint['mean'], dtype=torch.float32, device='cuda:0')
    std = torch.tensor(checkpoint['std'], dtype=torch.float32, device='cuda:0')

    oracle_root = args.output / 'oracle'
    oracle_root.mkdir(parents=True, exist_ok=True)
    actions_history = []
    records = []
    max_replay_error = 0.0
    fork_failure = None
    started = time.monotonic()
    with torch.inference_mode():
        reset_envs()
        reset_envs()
        for tick in range(args.ticks):
            if time.monotonic() - started > args.gpu_seconds:
                raise TimeoutError('closed-loop deadline')
            obs_np = task.obs_buf[0].cpu().numpy().astype(np.float32)
            features = p2_from_obs(obs_np)
            tau_np = None
            replay_error = 0.0
            if args.variant == 'p2_tau':
                query_dir = oracle_root / f'tick-{tick:03d}'
                query_dir.mkdir(parents=True, exist_ok=True)
                prefix = (np.stack(actions_history, axis=0) if actions_history
                          else np.zeros((0, 24), np.float32))
                np.save(query_dir / 'actions.npy', prefix.astype(np.float32))
                np.save(query_dir / 'student_obs.npy', obs_np)
                command = [
                    sys.executable, str(args.oracle_script),
                    '--subset', str(args.subset), '--vendor', str(args.vendor),
                    '--assets', str(args.assets),
                    '--actions', str(query_dir / 'actions.npy'),
                    '--student-obs', str(query_dir / 'student_obs.npy'),
                    '--output', str(query_dir), '--gpu', str(args.gpu),
                    '--seed', str(args.seed), '--horizon', str(args.horizon),
                    '--object-code', args.object_code, '--scale', str(args.scale)]
                environment = os.environ.copy()
                environment.setdefault('TORCH_EXTENSIONS_DIR',
                                       str(args.output / 'tmp/extensions'))
                completed = subprocess.run(command, env=environment, capture_output=True,
                                           text=True, timeout=args.gpu_seconds)
                (query_dir / 'stdout.log').write_text(completed.stdout)
                (query_dir / 'stderr.log').write_text(completed.stderr)
                if completed.returncode != 0:
                    raise RuntimeError(f'oracle query failed at tick {tick}: '
                                       f'{completed.stderr[-2000:]}')
                result = json.loads((query_dir / 'result.json').read_text())
                replay_error = float(result['replay_error'])
                max_replay_error = max(max_replay_error, replay_error)
                if replay_error > args.max_replay_error:
                    fork_failure = dict(tick=tick, replay_error=replay_error)
                    break
                with np.load(query_dir / 'tau.npz') as query:
                    tau_np = query['tau'].reshape(-1).astype(np.float32)
                features = np.concatenate([features, tau_np])
            x = torch.tensor(features, dtype=torch.float32, device='cuda:0')
            action = model(((x - mean) / std).reshape(1, -1)).clamp(-1, 1)
            records.append(dict(
                tick=np.int32(tick), obs=obs_np, action=action[0].cpu().numpy(),
                hand=hand_points()[0].cpu().numpy(),
                object_pose=task.root_state_tensor[task.object_indices, 0:13][0].cpu().numpy(),
                success=np.float32(task.successes[0].item()),
                replay_error=np.float32(replay_error),
                tau=(tau_np.reshape(args.horizon, 11, 3) if tau_np is not None
                     else np.zeros((0,), np.float32))))
            sim_step(action)
            actions_history.append(action[0].cpu().numpy())
            if tick % 10 == 0:
                print(json.dumps(dict(tick=tick, success=float(task.successes[0]),
                                      lift=float(task.root_state_tensor[task.object_indices[0], 2]
                                                 - records[0]['object_pose'][2]),
                                      replay_error=replay_error)), flush=True)
            if task.reset_buf.any():
                break
    trajectory = {key: np.stack([r[key] for r in records], axis=0)
                  for key in records[0] if key != 'tau'}
    trajectory['action_prefix'] = np.stack(actions_history, axis=0)
    np.savez_compressed(args.output / 'trajectory.npz', **trajectory)
    success = np.asarray([r['success'] for r in records])
    lift = np.asarray([r['object_pose'][2] for r in records]) - records[0]['object_pose'][2]
    manifest = dict(
        status='COMPLETE' if fork_failure is None else 'FORK_INVALID',
        variant=args.variant, seed=args.seed, ticks=int(len(records)),
        horizon=args.horizon, policy=str(args.policy),
        oracle_script=str(args.oracle_script),
        fork_failure=fork_failure, max_replay_error=max_replay_error,
        max_replay_tolerance=args.max_replay_error,
        final_success=float(success[-1]), max_success=float(success.max()),
        max_lift_m=float(lift.max()),
        success_frames=int(success.sum()),
        elapsed_s=time.monotonic() - started,
        note=('Student actions only. The oracle process is queried for tau and '
              'never supplies an executed action.'))
    write_json(args.output / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
