#!/usr/bin/env python3
"""Serial single-environment UniDexGrasp++ replay determinism check.

This reproduces the narrow contract that was previously validated in the
consequence-evaluator task (P-20261010-serial-finger-pulse-prefix): one
environment, one fresh process, fixed seed, and a deterministic closed-loop
teacher.  No PhysX state is restored or forked.  The trajectory is compared
bitwise against other fresh processes running the same protocol.

The check answers only one engineering question: is a single-environment
reset-from-scratch UniDex rollout reproducible enough to serve as the prefix
half of a later matched-state fork?  It makes no grasp or tau claim.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_unidex_oracle_tau import DEFAULT_OBJECT, load_replay_helpers, write_json


def collect(args):
    # Isaac Gym must load before torch.
    if 'float' not in np.__dict__:
        np.float = float
    from isaacgym import gymapi, gymutil
    import torch
    import yaml
    from isaacgym.torch_utils import quat_apply

    torch.set_num_threads(2)
    replay = load_replay_helpers()
    vendor = args.vendor.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'tmp').mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(vendor / 'dexgrasp'))
    ShadowHandGrasp = replay.task_class(vendor, args.output)

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
    write_json(args.output / 'simulation-config.json', dict(
        dt=sim.dt, substeps=sim.substeps, use_gpu_pipeline=sim.use_gpu_pipeline,
        envs=args.envs, scale=args.scale, object_code=object_code, ticks=args.ticks,
        seed=args.seed))
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
    teacher.load_state_dict({k[len('actor.'):]: v for k, v in weights.items()
                             if k.startswith('actor.')}, strict=True)
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
        points[:, 6:] += quat_apply(
            rigid[:, 6:, 3:7].reshape(-1, 4),
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
        seed_runtime(args.seed)
        task.reset_buf[:] = 1
        task.step(torch.zeros((args.envs, 24), device='cuda:0'), -1)
        if task.reset_buf.any():
            raise ValueError('reset left reset flags set')
        if not torch.isfinite(task.obs_buf).all():
            raise ValueError('nonfinite reset observation')

    rows = []
    start = time.monotonic()
    reset_tick = None
    with torch.inference_mode():
        # Warm up once so the measured reset is never the construction pose.
        reset_envs()
        reset_envs()
        for tick in range(args.ticks):
            if time.monotonic() - start > args.gpu_seconds:
                raise TimeoutError('bounded serial replay deadline')
            obs = task.obs_buf.clone()
            q = task.shadow_hand_dof_pos.clone()
            dq = task.shadow_hand_dof_vel.clone()
            hand = hand_points().clone()
            object_pose = task.root_state_tensor[task.object_indices, 0:13].clone()
            forces = task.vec_sensor_tensor.clone()
            dof_force = task.dof_force_tensor.clone()
            action = teacher(obs.clamp(-5, 5)).clamp(-1, 1)
            rows.append(dict(tick=np.int32(tick), obs=obs.cpu().numpy(),
                             q=q.cpu().numpy(), dq=dq.cpu().numpy(),
                             hand=hand.cpu().numpy(),
                             object_pose=object_pose.cpu().numpy(),
                             forces=forces.cpu().numpy(),
                             dof_force=dof_force.cpu().numpy(),
                             action=action.cpu().numpy(),
                             success=task.successes.cpu().numpy()))
            sim_step(action)
            if task.reset_buf.any():
                reset_tick = tick
                break
    data = {key: np.stack([r[key] for r in rows], axis=0) for key in rows[0]}
    for key, value in data.items():
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():
            raise ValueError(f'nonfinite serial replay field {key}')
    np.savez_compressed(args.output / 'trajectory.npz', **data)
    manifest = dict(status='COMPLETE', seed=args.seed, envs=args.envs, ticks=len(rows),
                    horizon=None, object_code=object_code, scale=args.scale,
                    reset_tick=reset_tick,
                    protocol=('single_env_fresh_process_closed_loop_teacher'
                              if args.envs == 1 else 'multi_env_closed_loop_teacher'),
                    note=('No PhysX state restore or fork. Compare trajectory.npz '
                          'bitwise across fresh processes with the same seed.'),
                    elapsed_s=time.monotonic() - start)
    write_json(args.output / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subset', type=Path, required=True)
    p.add_argument('--vendor', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--ticks', type=int, default=100)
    p.add_argument('--envs', type=int, default=1)
    p.add_argument('--object-code', default=DEFAULT_OBJECT)
    p.add_argument('--scale', type=float, default=.08)
    p.add_argument('--seed', type=int, default=421)
    p.add_argument('--gpu-seconds', type=int, default=300)
    args = p.parse_args()
    if not (1 <= args.envs <= 8 and 1 <= args.ticks <= 400):
        raise ValueError('bounded rollout arguments required')
    if args.output == Path('/tmp') or Path('/tmp') in args.output.parents:
        raise ValueError('outputs must remain in the project workspace')
    for key in ('subset', 'vendor', 'assets', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    collect(args)


if __name__ == '__main__':
    main()
