#!/usr/bin/env python3
"""Answer one online oracle-tau query in a fresh process.

The student process writes its applied action prefix and current 300-D
observation.  This process resets one environment with the same seed, replays
the prefix, verifies that the replayed observation matches the student state,
then rolls the frozen teacher forward for ``horizon`` steps and writes

    tau_t = H_teacher[t+1:t+horizon] - H_t.

It is deliberately a fresh process: Isaac Gym permits only one simulator
instance per process, and the validated fork contract is a fresh reset plus
prefix replay, never an intermediate PhysX restore.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'run'))
from probe_unidex_oracle_tau import DEFAULT_OBJECT, load_replay_helpers, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subset', type=Path, required=True)
    p.add_argument('--vendor', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--actions', type=Path, required=True)
    p.add_argument('--student-obs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--horizon', type=int, default=8)
    p.add_argument('--object-code', default=DEFAULT_OBJECT)
    p.add_argument('--scale', type=float, default=.08)
    args = p.parse_args()
    for key in ('subset', 'vendor', 'assets', 'actions', 'student_obs', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    if args.output == Path('/tmp') or Path('/tmp') in args.output.parents:
        raise ValueError('oracle output must remain in the project workspace')
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
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

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

    prefix = np.load(args.actions)
    student_obs = np.load(args.student_obs).astype(np.float32).reshape(-1)
    if prefix.ndim != 2 or prefix.shape[1] != 24:
        raise ValueError(f'action prefix must be (T,24), got {prefix.shape}')
    if student_obs.shape != (300,):
        raise ValueError(f'student observation must be (300,), got {student_obs.shape}')
    started = time.monotonic()
    with torch.inference_mode():
        reset_envs()
        reset_envs()
        for step in range(prefix.shape[0]):
            action = torch.tensor(prefix[step:step + 1], dtype=torch.float32, device='cuda:0')
            sim_step(action)
            if task.reset_buf.any():
                raise ValueError(f'prefix replay reset at step {step}')
        replay_obs = task.obs_buf.clone()
        replay_error = float((replay_obs - torch.tensor(student_obs, device='cuda:0').reshape(1, -1)).abs().max())
        current_hand = hand_points()
        future = []
        teacher_actions = []
        fork_reset = False
        for _ in range(args.horizon):
            if task.reset_buf.any():
                fork_reset = True
                break
            action = teacher(task.obs_buf.clamp(-5, 5)).clamp(-1, 1)
            teacher_actions.append(action.clone())
            sim_step(action)
            future.append(hand_points().clone())
        if fork_reset or len(future) != args.horizon:
            raise ValueError('teacher fork reset before horizon')
        tau = torch.stack(future, dim=1) - current_hand[:, None]
        np.savez_compressed(
            args.output / 'tau.npz',
            tau=tau[0].cpu().numpy(),
            replay_obs=replay_obs[0].cpu().numpy(),
            current_hand=current_hand[0].cpu().numpy(),
            object_pose=task.root_state_tensor[task.object_indices, 0:13][0].cpu().numpy(),
            teacher_actions=torch.cat(teacher_actions).cpu().numpy())
    manifest = dict(status='COMPLETE', seed=args.seed, horizon=args.horizon,
                    prefix_steps=int(prefix.shape[0]),
                    replay_error=replay_error,
                    replay_tolerance=5e-2,
                    fork_valid=bool(replay_error <= 5e-2),
                    tau_max_abs=float(np.abs(tau.cpu().numpy()).max()),
                    elapsed_s=time.monotonic() - started)
    write_json(args.output / 'result.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
