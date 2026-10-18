#!/usr/bin/env python3
"""Check online oracle-tau fork feasibility for one UniDex environment.

The validated contract resets a single environment and replays the action
prefix; it never restores intermediate PhysX state.  A closed-loop student
needs such a query at every tick.  This audit checks whether a *fresh task in
the same process* reaches the student state bitwise, which would make the
online query cheap, or whether it must fall back to a fresh process per query.
"""
import argparse
import copy
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
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=421)
    p.add_argument('--ticks', type=int, default=30)
    p.add_argument('--query-ticks', type=int, nargs='+', default=[0, 1, 5, 10, 20, 29])
    p.add_argument('--horizon', type=int, default=8)
    p.add_argument('--object-code', default=DEFAULT_OBJECT)
    p.add_argument('--scale', type=float, default=.08)
    args = p.parse_args()
    for key in ('subset', 'vendor', 'assets', 'output'):
        setattr(args, key, getattr(args, key).resolve())
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
    sys.path.insert(0, str(vendor / 'dexgrasp'))
    cfg = yaml.safe_load((vendor / 'dexgrasp/cfg/shadow_hand_grasp.yaml').read_text())
    cfg['env'].update(numEnvs=1, random_time=False, printNumSuccesses=False,
                      object_code_dict={args.object_code: [args.scale]})
    cfg['env']['asset']['assetRoot'] = str(assets)
    cfg['graphics_device_id'] = -1
    work = assets.parent / 'dexgrasp'
    work.mkdir(exist_ok=True)
    os.chdir(work)

    def make_sim():
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
        return sim

    def make_task(label):
        out = args.output / f'fork-{label}'
        out.mkdir(parents=True, exist_ok=True)
        (out / 'tmp').mkdir(parents=True, exist_ok=True)
        Shadow = replay.task_class(vendor, out)
        return Shadow(copy.deepcopy(cfg), make_sim(), gymapi.SIM_PHYSX, 'cuda', 0, True)

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
    tip_offsets = torch.tensor([[0, 0, .0275]] + [[0, 0, .026]] * 4,
                               dtype=torch.float32, device='cuda:0')

    def hand_points(task):
        rigid = task.rigid_body_states[:, task._audit_indices]
        points = rigid[:, :, :3].clone()
        points[:, 6:] += quat_apply(
            rigid[:, 6:, 3:7].reshape(-1, 4),
            tip_offsets[None].expand(1, -1, -1).reshape(-1, 3)).reshape(1, 5, 3)
        return points

    def sim_step(task, action):
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

    def reset(task):
        seed_runtime(args.seed)
        task.reset_buf[:] = 1
        task.step(torch.zeros((1, 24), device='cuda:0'), -1)
        if task.reset_buf.any():
            raise ValueError('reset left reset flags set')

    student = make_task('student')
    indices = [student.gym.find_actor_rigid_body_index(student.envs[0], student.shadow_hands[0],
                                                       n, gymapi.DOMAIN_ACTOR) for n in names]
    if any(i < 0 for i in indices):
        raise ValueError('missing hand landmark indices')
    student._audit_indices = indices
    reset(student)
    reset(student)

    actions = []
    states = []
    hands = []
    with torch.inference_mode():
        for tick in range(args.ticks):
            states.append(student.obs_buf.clone())
            hands.append(hand_points(student).clone())
            action = teacher(student.obs_buf.clamp(-5, 5)).clamp(-1, 1)
            actions.append(action.clone())
            sim_step(student, action)

        reports = []
        for tick in args.query_ticks:
            if tick >= args.ticks:
                continue
            fresh = make_task(f'query-{tick}')
            fresh._audit_indices = indices
            reset(fresh)
            reset(fresh)
            for action in actions[:tick]:
                sim_step(fresh, action)
            obs_diff = float((fresh.obs_buf - states[tick]).abs().max().item())
            hand_diff = float((hand_points(fresh) - hands[tick]).abs().max().item())
            future = []
            fork_reset = False
            for _ in range(args.horizon):
                if fresh.reset_buf.any():
                    fork_reset = True
                    break
                action = teacher(fresh.obs_buf.clamp(-5, 5)).clamp(-1, 1)
                sim_step(fresh, action)
                future.append(hand_points(fresh).clone())
            tau_max = None
            tau_ref_max = None
            if not fork_reset:
                tau = torch.stack(future, 1) - hands[tick][:, None]
                tau_max = float(tau.abs().max().item())
                if tick + args.horizon < args.ticks:
                    reference = torch.stack(hands[tick + 1:tick + 1 + args.horizon], 1) - hands[tick][:, None]
                    tau_ref_max = float((tau - reference).abs().max().item())
            # Compare the teacher future against the student trajectory at the
            # same ticks when it exists; this is the semantic oracle check.
            reports.append(dict(tick=tick, obs_max_abs_diff=obs_diff,
                                hand_max_abs_diff=hand_diff,
                                fork_reset=fork_reset,
                                tau_max_abs=tau_max,
                                tau_vs_student_future_max_abs=tau_ref_max))
            print(json.dumps(reports[-1]), flush=True)
            fresh.gym.destroy_sim(fresh.sim)
            del fresh
            torch.cuda.empty_cache()
    write_json(args.output / 'fork-determinism.json', dict(
        schema='ref2dex.unidex-fork-determinism.v1', seed=args.seed,
        ticks=args.ticks, horizon=args.horizon, query_ticks=args.query_ticks,
        reports=reports,
        bitwise_exact=all(r['obs_max_abs_diff'] == 0 for r in reports)))
    print(json.dumps(dict(bitwise_exact=all(r['obs_max_abs_diff'] == 0 for r in reports),
                          max_obs_diff=max(r['obs_max_abs_diff'] for r in reports)), indent=2))


if __name__ == '__main__':
    main()
