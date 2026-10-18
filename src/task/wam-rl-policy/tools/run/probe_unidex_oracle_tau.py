#!/usr/bin/env python3
"""Collect a bounded UniDexGrasp++ live-state oracle-tau identification probe."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np

DEFAULT_OBJECT = 'core/bottle-109d55a137c042f5760315ac3bf2c13e'
REPLAY = Path(__file__).resolve().parents[3] / 'video-data-preparation/tools/run/replay_unidexgrasp.py'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(value, indent=2))
    partial.replace(path)


def load_replay_helpers():
    spec = importlib.util.spec_from_file_location('replay_unidexgrasp', REPLAY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect(args):
    # Isaac Gym must load before torch.
    if 'float' not in np.__dict__:
        np.float = float
    from isaacgym import gymapi, gymtorch, gymutil
    import torch
    import yaml
    from isaacgym.torch_utils import quat_apply

    torch.set_num_threads(2)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    replay = load_replay_helpers()
    vendor = args.vendor.resolve()
    # TORCH_EXTENSIONS_DIR may create the output directory while Isaac Gym's
    # native extension is imported, before the collector reaches this point.
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'tmp').mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(vendor / 'dexgrasp'))
    ShadowHandGrasp = replay.task_class(vendor, args.output)
    from utils.data_info import plane2euler

    assets = args.assets.resolve()
    if not assets.exists():
        raise FileNotFoundError(assets)
    object_code = args.object_code
    object_dict = {object_code: [args.scale]}
    pose_file = assets / 'datasetv4.1_posedata.npy'
    pose_data = np.load(pose_file, allow_pickle=True).item()
    if object_code not in pose_data or args.scale not in pose_data[object_code]:
        raise ValueError(f'prepared pose data has no {object_code}/{args.scale}')

    work = assets.parent / 'dexgrasp'
    work.mkdir(exist_ok=True)
    os.chdir(work)
    cfg = yaml.safe_load((vendor / 'dexgrasp/cfg/shadow_hand_grasp.yaml').read_text())
    cfg['env'].update(numEnvs=args.envs, random_time=False, printNumSuccesses=False,
                      object_code_dict=object_dict)
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
    write_json(args.output / 'simulation-config.json', dict(dt=sim.dt, substeps=sim.substeps,
        use_gpu_pipeline=sim.use_gpu_pipeline, envs=args.envs, scale=args.scale, object_code=object_code,
        horizon=args.horizon, ticks=args.ticks, noise_std=args.noise_std))
    task = ShadowHandGrasp(cfg, sim, gymapi.SIM_PHYSX, 'cuda', 0, True)

    layers = []
    widths = [300, 1024, 1024, 512, 512, 24]
    for i, (lo, hi) in enumerate(zip(widths, widths[1:])):
        layers.append(torch.nn.Linear(lo, hi))
        if i < 4:
            layers.append(torch.nn.ELU())
    teacher = torch.nn.Sequential(*layers).to('cuda:0').eval()
    weights = torch.load(args.subset / 'plus/dexgrasp/example_model/state_based_model.pt',
                         map_location='cuda:0', weights_only=True)
    teacher.load_state_dict({k[len('actor.'):]: v for k, v in weights.items() if k.startswith('actor.')}, strict=True)
    for p in teacher.parameters():
        p.requires_grad_(False)

    names = ['robot0:palm', 'robot0:thhub', 'robot0:ffknuckle', 'robot0:mfknuckle',
             'robot0:rfknuckle', 'robot0:lfknuckle', 'robot0:thdistal', 'robot0:ffdistal',
             'robot0:mfdistal', 'robot0:rfdistal', 'robot0:lfdistal']
    indices = [task.gym.find_actor_rigid_body_index(task.envs[0], task.shadow_hands[0], n,
                                                     gymapi.DOMAIN_ACTOR) for n in names]
    if any(i < 0 for i in indices):
        raise ValueError('missing hand landmark indices: ' + str(dict(zip(names, indices))))
    tip_offsets = torch.tensor([[0, 0, .0275]] + [[0, 0, .026]] * 4,
                               dtype=torch.float32, device='cuda:0')

    def hand_points():
        rigid = task.rigid_body_states[:, indices]
        points = rigid[:, :, :3].clone()
        points[:, 6:] += quat_apply(rigid[:, 6:, 3:7].reshape(-1, 4),
                                    tip_offsets[None].expand(args.envs, -1, -1).reshape(-1, 3)).reshape(args.envs, 5, 3)
        return points

    def sim_step(action):
        task.pre_physics_step(action)
        for _ in range(task.control_freq_inv):
            task.gym.simulate(task.sim)
            task.gym.fetch_results(task.sim, True)
        task.post_physics_step()

    # These are the mutable tensors that can affect a continuation. Physics tensors are
    # restored through indexed setters; task buffers are restored in-place.
    tensor_names = ['prev_targets', 'cur_targets', 'actions', 'apply_forces', 'apply_torque',
                    'progress_buf', 'reset_buf', 'reset_goal_buf', 'successes', 'current_successes',
                    'consecutive_successes', 'z_theta', 'target_qpos', 'target_hand_pos',
                    'target_hand_rot', 'object_init_euler_xy', 'object_init_z', 'goal_states']
    actor_ids = torch.cat([task.hand_indices, task.object_indices, task.goal_object_indices, task.table_indices]).to(torch.int32)
    dof_ids = task.hand_indices.to(torch.int32)

    def snapshot_state():
        return dict(root=task.root_state_tensor.clone(), dof=task.dof_state.clone(),
                    tensors={n: getattr(task, n).clone() for n in tensor_names},
                    hand=hand_points().clone(), obs=task.obs_buf.clone())

    def restore_state(snap):
        task.root_state_tensor.copy_(snap['root'])
        task.dof_state.copy_(snap['dof'])
        for n, value in snap['tensors'].items():
            getattr(task, n).copy_(value)
        task.gym.set_actor_root_state_tensor(task.sim, gymtorch.unwrap_tensor(task.root_state_tensor))
        task.gym.set_dof_state_tensor(task.sim, gymtorch.unwrap_tensor(task.dof_state))
        task.gym.set_dof_position_target_tensor(task.sim, gymtorch.unwrap_tensor(task.prev_targets))
        # Do not refresh force sensors here: PhysX reports the previous
        # branch's contact forces until the next simulation step.  The fixed
        # initial observation is exact at replay tick zero; after one replay
        # step post_physics_step refreshes the sensors from the new branch.
        task.obs_buf.copy_(snap['obs'])

    records = []
    replay_errors = []
    start = time.monotonic()
    with torch.inference_mode():
        task.reset_buf[:] = 1
        task.step(torch.zeros((args.envs, 24), device='cuda:0'), -1)
        # Do not fork by writing an intermediate PhysX state back into the live
        # simulation.  Keep the post-reset state fixed and reconstruct each
        # sampled state by replaying the learner action prefix from this state.
        initial = snapshot_state()
        history = []
        for tick in range(args.ticks):
            if time.monotonic() - start > args.gpu_seconds:
                raise TimeoutError('bounded oracle probe deadline')
            restore_state(initial)
            for prior_action in history:
                sim_step(prior_action)
            if task.reset_buf.any():
                raise ValueError('replayed learner prefix reset before sampled tick')
            if not torch.isfinite(task.obs_buf).all():
                raise ValueError('nonfinite replayed observation')
            current = snapshot_state()
            state = task.obs_buf.clone()
            current_hand = current['hand']
            teacher_action = teacher(state.clamp(-5, 5)).clamp(-1, 1)
            future = []
            fork_reset = False
            for _ in range(args.horizon):
                if task.reset_buf.any():
                    fork_reset = True
                    break
                action = teacher(task.obs_buf.clamp(-5, 5)).clamp(-1, 1)
                sim_step(action)
                future.append(hand_points().clone())
            if fork_reset or len(future) != args.horizon:
                raise ValueError('teacher fork reset before horizon; no valid live-state row')
            tau = torch.stack(future, dim=1) - current_hand[:, None]
            noise = torch.randn(teacher_action.shape, device='cuda:0') * args.noise_std
            learner_action = (teacher_action + noise).clamp(-1, 1)
            # Reconstruct the live state once more from the same initial state,
            # then apply exactly the learner action that advances the trajectory.
            restore_state(initial)
            for prior_action in history:
                sim_step(prior_action)
            replay_error = float((task.obs_buf - state).abs().max().item())
            replay_errors.append(replay_error)
            if replay_error > args.max_replay_error:
                diff = (task.obs_buf - state).abs()
                flat = int(diff.argmax().item())
                print(json.dumps(dict(replay_error=replay_error, diff_index=flat,
                                      replay_value=float(task.obs_buf.flatten()[flat].item()),
                                      reference_value=float(state.flatten()[flat].item()))), flush=True)
                raise ValueError(f'non-deterministic replay error {replay_error}')
            records.append(dict(state=state.cpu().numpy(), tau=tau.cpu().numpy(),
                                teacher_action=teacher_action.cpu().numpy(), learner_action=learner_action.cpu().numpy(),
                                tick=np.full(args.envs, tick, np.int32),
                                replay_error=np.full(args.envs, replay_error, np.float32)))
            sim_step(learner_action)
            history.append(learner_action.clone())
            if tick % 8 == 0:
                print(json.dumps(dict(tick=tick, replay_error=replay_error,
                                      max_tau=float(tau.abs().max().item()),
                                      success=task.successes.cpu().tolist())), flush=True)
    data = {key: np.concatenate([r[key] for r in records], axis=0) for key in records[0]}
    if not np.isfinite(data['state']).all() or not np.isfinite(data['tau']).all():
        raise ValueError('nonfinite oracle probe data')
    np.savez_compressed(args.output / 'oracle_queries.npz', **data)
    result = dict(status='COMPLETE', rows=int(len(data['state'])), envs=args.envs, ticks=args.ticks,
                  horizon=args.horizon, replay_error_max=max(replay_errors),
                  replay_error_mean=float(np.mean(replay_errors)), hand_landmarks=names,
                  tau_definition='teacher_future_hand_minus_live_hand', training_allowed=False,
                  elapsed_s=time.monotonic() - start)
    write_json(args.output / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subset', type=Path, required=True)
    p.add_argument('--vendor', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--envs', type=int, default=8)
    p.add_argument('--ticks', type=int, default=32)
    p.add_argument('--horizon', type=int, default=8)
    p.add_argument('--object-code', default=DEFAULT_OBJECT)
    p.add_argument('--scale', type=float, default=.08)
    p.add_argument('--seed', type=int, default=418)
    p.add_argument('--noise-std', type=float, default=.05)
    p.add_argument('--max-replay-error', type=float, default=5e-2)
    p.add_argument('--gpu-seconds', type=int, default=300)
    args = p.parse_args()
    if not (1 <= args.envs <= 8 and 1 <= args.ticks <= 32 and 1 <= args.horizon <= 8):
        raise ValueError('bounded probe arguments required')
    if args.output == Path('/tmp') or Path('/tmp') in args.output.parents:
        raise ValueError('outputs must remain in the project workspace')
    for key in ('subset', 'vendor', 'assets', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    collect(args)


if __name__ == '__main__':
    main()
