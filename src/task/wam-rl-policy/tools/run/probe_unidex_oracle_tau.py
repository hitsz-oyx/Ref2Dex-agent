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
    from isaacgym import gymapi, gymutil
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

    def seed_runtime(seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def reset_envs():
        """Return to the fixed post-reset state with a real task reset.

        The earlier implementation wrote an intermediate PhysX snapshot back
        into the same simulation.  That is not a fork: contact/warm-start state
        lives outside the root/dof tensors and diverged after a contact-rich
        teacher rollout.  A fresh reset plus the recorded action prefix lets
        the physics engine rebuild the trajectory instead.
        """
        seed_runtime(args.seed)
        task.reset_buf[:] = 1
        task.step(torch.zeros((args.envs, 24), device='cuda:0'), -1)
        if task.reset_buf.any():
            raise ValueError('reset left reset flags set')
        if not torch.isfinite(task.obs_buf).all():
            raise ValueError('nonfinite reset observation')

    def replay_prefix(actions):
        reset_envs()
        for prior_action in actions:
            sim_step(prior_action)
        if task.reset_buf.any():
            raise ValueError('replayed learner prefix reset before sampled tick')
        if not torch.isfinite(task.obs_buf).all():
            raise ValueError('nonfinite replayed observation')
        return task.obs_buf.clone()

    records = []
    replay_errors = []
    truncation = None
    start = time.monotonic()
    history = []
    # Reset randomization and action noise use independent streams.  Every
    # tick re-seeds the reset stream so the replayed prefix starts from the
    # same initial state, while the noise bank is drawn once up front.
    noise_generator = torch.Generator(device='cuda:0')
    noise_generator.manual_seed(args.seed + 104729)
    noise_bank = torch.randn((args.ticks, args.envs, 24), generator=noise_generator,
                             device='cuda:0')
    with torch.inference_mode():
        # The first reset of a freshly created task can expose the construction
        # pose instead of the randomized reset pose in the mirrored observation.
        # Warm up once so every measured tick starts from a settled reset.
        reset_envs()
        for tick in range(args.ticks):
            if time.monotonic() - start > args.gpu_seconds:
                truncation = f'bounded probe deadline at tick {tick}'
                break
            state = replay_prefix(history)
            current_hand = hand_points().clone()
            first_meta = dict(z=task.z_theta.clone(), euler=task.object_init_euler_xy.clone(),
                              target=task.target_hand_pos.clone())
            replay_error = 0.0
            if args.replay_checks:
                second = replay_prefix(history)
                replay_error = float((second - state).abs().max().item())
                replay_errors.append(replay_error)
                if replay_error > args.max_replay_error:
                    diff = (second - state).abs()
                    top = torch.topk(diff.flatten(), 8)
                    meta_diff = dict(
                        z=float((task.z_theta - first_meta['z']).abs().max().item()),
                        euler=float((task.object_init_euler_xy - first_meta['euler']).abs().max().item()),
                        target=float((task.target_hand_pos - first_meta['target']).abs().max().item()))
                    print(json.dumps(dict(tick=tick, replay_error=replay_error,
                                          top_indices=top.indices.cpu().tolist(),
                                          top_values=[float(v) for v in top.values],
                                          second_values=[float(second.flatten()[i]) for i in top.indices],
                                          first_values=[float(state.flatten()[i]) for i in top.indices],
                                          n_over_threshold=int((diff > 1e-3).sum().item()),
                                          meta_diff=meta_diff)), flush=True)
                    # Each row is still self-consistent (s, tau and the teacher
                    # action come from the same simulator instance); the error
                    # audits whether the prefix replay is a reproducible
                    # trajectory, which matters for closed-loop use, not for
                    # offline identification.
                # Continue from the state the fork will actually mutate.
                state = second
                current_hand = hand_points().clone()
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
                truncation = f'teacher fork reset before horizon at tick {tick}'
                break
            tau = torch.stack(future, dim=1) - current_hand[:, None]
            learner_action = (teacher_action + noise_bank[tick] * args.noise_std).clamp(-1, 1)
            records.append(dict(state=state.cpu().numpy(), tau=tau.cpu().numpy(),
                                teacher_action=teacher_action.cpu().numpy(),
                                learner_action=learner_action.cpu().numpy(),
                                tick=np.full(args.envs, tick, np.int32),
                                replay_error=np.full(args.envs, replay_error, np.float32)))
            history.append(learner_action.clone())
            if tick % 8 == 0:
                print(json.dumps(dict(tick=tick, replay_error=replay_error,
                                      max_tau=float(tau.abs().max().item()))), flush=True)
    if not records:
        raise ValueError(f'no valid live-state rows: {truncation}')
    data = {key: np.concatenate([r[key] for r in records], axis=0) for key in records[0]}
    data['learner_actions'] = torch.stack(history, dim=0).cpu().numpy()
    if not np.isfinite(data['state']).all() or not np.isfinite(data['tau']).all():
        raise ValueError('nonfinite oracle probe data')
    np.savez_compressed(args.output / 'oracle_queries.npz', **data)
    replay_max = max(replay_errors) if replay_errors else 0.0
    replay_mean = float(np.mean(replay_errors)) if replay_errors else 0.0
    replay_valid_fraction = (float(np.mean(np.asarray(replay_errors) <= args.max_replay_error))
                             if replay_errors else 1.0)
    valid = truncation is None and len(records) > 0
    result = dict(status='COMPLETE' if truncation is None else 'TRUNCATED',
                  truncation=truncation, rows=int(len(data['state'])),
                  envs=args.envs, ticks=int(len(records)), horizon=args.horizon,
                  seed=args.seed, object_code=object_code, scale=args.scale,
                  noise_std=args.noise_std,
                  replay_mode='reset_then_replay_prefix',
                  replay_checks=bool(args.replay_checks),
                  replay_error_max=replay_max, replay_error_mean=replay_mean,
                  trajectory_replay_valid_fraction=replay_valid_fraction,
                  hand_landmarks=names,
                  tau_definition='teacher_future_hand_minus_live_hand',
                  identification_allowed=valid, training_allowed=False,
                  closed_loop_allowed=False, elapsed_s=time.monotonic() - start)
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
    p.add_argument('--replay-checks', action='store_true', default=True)
    p.add_argument('--no-replay-checks', dest='replay_checks', action='store_false')
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
