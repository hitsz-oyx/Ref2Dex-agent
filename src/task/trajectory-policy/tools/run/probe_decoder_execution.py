"""One bounded oracle-label decoder coverage wave; no actor training or WM."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
OLD = ROOT / 'src/task/consequence-evaluator'
sys.path[:0] = [str(TASK / 'src'), str(OLD / 'src'), str(OLD / 'tools/run'),
               str(ROOT), str(ROOT / 'third_party/DExplore/dexplore'),
               str(ROOT / 'src/task/cm-interaction-oracle/src')]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def gpu_state(gpu):
    row = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=utilization.gpu,memory.used,memory.total',
        '--format=csv,noheader,nounits'], text=True)
    util, used, total = map(int, row.strip().split(','))
    processes = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).strip()
    return dict(utilization=util, used_mib=used, total_mib=total, processes=processes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('inputs', 'reference', 'geometry', 'checkpoint', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or ROOT / 'outputs/trajectory-policy' not in args.output.parents:
        raise ValueError('fresh trajectory-policy output required')
    before = gpu_state(args.gpu)
    if before['utilization'] > 10 or before['used_mib'] > 512 or before['total_mib'] - before['used_mib'] < 20000:
        raise ValueError('GPU is not idle: ' + repr(before))
    initial_pids = {int(row.split(',')[0]) for row in before['processes'].splitlines() if row}
    scratch = ROOT / 'tmp/trajectory-policy'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(scratch),
        TORCH_EXTENSIONS_DIR=str(scratch / 'torch-extensions'), TRITON_CACHE_DIR=str(scratch / 'triton'),
        OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    from isaacgym import gymtorch  # noqa: must precede torch
    import torch
    from consequence_evaluator.data import sha
    from consequence_evaluator.contracts import HAND_LINKS
    from consequence_evaluator.reference_tracking import reference_velocity, future_reference_velocity, wrist_feedforward, native_action
    from consequence_evaluator.tau_tracking import TauTracker, SCHEMA, tau_features
    from consequence_evaluator.native_reset import install_reset_patch
    from consequence_evaluator.physical_geometry import PhysicalGeometry, poses
    from consequence_evaluator.value_geometry import TableSupport
    from consequence_evaluator.gate1 import episode_outcome
    from trajectory_policy.decoder import TrajectoryDecoder
    from trajectory_policy.inputs import load_geometry
    from oracle_y_utility import align_native_reference_tables
    import evaluate as native
    from env.tasks.base_dexplore_task import DexploreTask

    if torch.__version__ != '2.4.1+cu121':
        raise ValueError('pinned graspenv runtime required')
    torch.set_num_threads(2)
    torch.manual_seed(297)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0')
    urdf = ROOT / 'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    geometry_data, initial, hand, hashes = load_geometry(args.geometry, args.reference, urdf)
    cfg = json.loads(args.inputs.read_text())
    essential = [cfg['actor'], cfg['cfg_env'], cfg['cfg_train']]
    essential += [str(ROOT / 'third_party/DExplore/dexplore' / file) for file in (
        'evaluate.py', 'env/tasks/base_dexplore_task.py', 'env/tasks/dexplore_inspire.py')]
    essential += [path for path in cfg['input_sha256'] if path.endswith('interaction_hand_inspire.pt')]
    for path in essential:
        if sha(path) != cfg['input_sha256'][path]:
            raise ValueError('native input drift: ' + path)
    sources = list((TASK / 'src/trajectory_policy').glob('*.py'))
    sources += [OLD / 'src/consequence_evaluator' / (name + '.py') for name in (
        'tau_tracking', 'reference_tracking', 'reset_kinematics', 'retargeter', 'native_reset',
        'physical_geometry', 'value_geometry', 'gate1', 'contracts', 'reference_motion')]
    sources += [ROOT / 'src/task/cm-interaction-oracle/src/oracle_y_utility.py', Path(__file__), args.inputs, args.checkpoint]
    sources += [Path(path) for path in essential] + list(Path(cfg['motions']).rglob('*.pt'))
    for path in sources:
        hashes[str(path.resolve())] = sha(path)
    if sha(args.checkpoint) != '79ee282e6b2287375c6480eb8f793864c858311b968ffa97b95c65842502fa45':
        raise ValueError('fixed retained executor identity required')
    reference_q = torch.as_tensor(geometry_data['q'], device=device)
    reference_hand = torch.as_tensor(hand, device=device)
    decoder = TrajectoryDecoder(urdf, device)
    policy = TauTracker().to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if payload['schema'] != SCHEMA:
        raise ValueError('tau controller schema mismatch')
    policy.load_state_dict(payload['state_dict'], strict=True)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    weights = {key: value.clone() for key, value in policy.state_dict().items()}
    roles = np.random.default_rng(297).permutation(np.repeat(
        ['tau_gt', 'dense_fk', 'knots48', 'knots48_repeat'], 4)).tolist()
    args.output.mkdir(parents=True)
    manifest = dict(schema='ref2dex.trajectory-decoder-execution.v1', status='RUNNING',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        physical_gpu=args.gpu, gpu_before=before, seed=297, roles=roles, input_sha256=hashes,
        inference_contract='Oracle future HAND-derived knots only; live object/q; frozen R; not H-to-c inference',
        replan_interval=8, controls=542, envs=16, dt=1/30,
        claim='Single-motion decoder/executor coverage Probe, not trained policy or Cm benefit')
    write(args.output / 'manifest.json', manifest)
    install_reset_patch()
    reset = DexploreTask._reset_ref_state_init
    def aligned(task, ids):
        align_native_reference_tables(task)
        return reset(task, ids)
    DexploreTask._reset_ref_state_init = aligned
    loader = native.torch_ext.load_checkpoint
    def load(path):
        value = loader(path)
        compiled = [key.startswith('_orig_mod.') for key in value['model']]
        if any(compiled) and not all(compiled):
            raise ValueError('mixed compiled keys')
        if all(compiled):
            value = dict(value, model={key[10:]: val for key, val in value['model'].items()})
        return value
    native.torch_ext.load_checkpoint = load
    started = time.monotonic()

    class Player(native.EvalPlayer):
        def run(self):
            task = self.env.task
            n = task.num_envs
            if n != 16 or self.is_rnn or len(task.motion_file) != 1 or str(task.device) != 'cpu' or not task.gym.get_sim_params(task.sim).physx.use_gpu:
                raise ValueError('sixteen-env single-motion CPU exchange/GPU PhysX required')
            task._enable_early_termination = False
            task._adaptive_kappa_enabled = False
            task._state_init = DexploreTask.StateInit.Start
            task._hybrid_init_prob = 1.
            self.model.eval()
            obs = self.env_reset(torch.arange(n, device=task.device))
            self.get_batch_size(obs['obs'], 1)
            if (task.start_times != 0).any() or not np.array_equal(initial['dof_position'], task._dof_pos[0].numpy()):
                raise ValueError('actual frame0 initialization mismatch')
            names = task.gym.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles[0])
            key_ids = [names.index(name) for name in HAND_LINKS]
            offset, scale = task._pd_action_offset.to(device), task._pd_action_scale.to(device)
            props = task.gym.get_actor_dof_properties(task.envs[0], task.humanoid_handles[0])
            ratio = torch.as_tensor(props['damping'][:6].copy() / props['stiffness'][:6].copy(), device=device)
            dt = task.gym.get_sim_params(task.sim).dt * task.control_freq_inv
            if abs(dt - 1/30) > 1e-8 or not torch.allclose(ratio, torch.full_like(ratio, .1)):
                raise ValueError('fixed native gains/control interval mismatch')
            manifest['native_controller'] = dict(offset=offset.cpu().tolist(), scale=scale.cpu().tolist(),
                gain_ratio=ratio.cpu().tolist())
            global_velocity = reference_velocity(reference_q, dt)
            gt = torch.tensor([r == 'tau_gt' for r in roles], device=device)
            latent_rows = np.flatnonzero(np.isin(roles, ['knots48', 'knots48_repeat']))
            latent_ids = torch.as_tensor(latent_rows, device=device)
            previous = torch.zeros(n, 12, device=device)
            capture = pd_target = None
            original_pre, original_pd = task.pre_physics_step, task._action_to_pd_targets
            def pre(action):
                nonlocal capture
                capture = action.detach().clone().to(device)
                return original_pre(action)
            def pd(action):
                nonlocal pd_target
                pd_target = original_pd(action)
                return pd_target
            task.pre_physics_step, task._action_to_pd_targets = pre, pd
            physical = PhysicalGeometry(task, ROOT / 'third_party/DExplore/dexplore/data/assets', distance_device=device)
            support = TableSupport(ROOT / 'third_party/DExplore/dexplore/data/assets', task.device)
            arrays = {key: [] for key in ('q', 'dq', 'hand', 'obj', 'velocity', 'surface_gap', 'support_gap',
                'table_footprint', 'features', 'latent', 'intended', 'action', 'applied', 'pd_targets', 'clipped')}
            plans = []
            def measure():
                bodies = task._rigid_body_state.view(n, -1, 13)
                return dict(q=task._dof_pos.to(device), dq=task._dof_vel.to(device),
                    hand=bodies[:, key_ids, :3].to(device), obj=poses(task._target_states).to(device),
                    velocity=task._target_states[:, 7:13].to(device))
            def record():
                current = measure()
                _, gap = physical.measure(task)
                sup, foot = support.measure(task, physical)
                current.update(surface_gap=gap, support_gap=sup, table_footprint=foot)
                for key, value in current.items():
                    arrays[key].append(value.detach().cpu().numpy().copy())
            record()
            work_started = time.monotonic()
            for tick in range(542):
                with torch.no_grad():
                    current = measure()
                    if tick % 8 == 0:
                        index = (tick + torch.arange(1, 25, device=device)).clamp_max(542)
                        q = torch.cat((current['q'][:, None], reference_q[index].expand(n, -1, -1)), 1)
                        c = np.zeros((n, 48), dtype=np.float32)
                        c[latent_rows] = decoder.encode(q[latent_ids, 1:].cpu().numpy(),
                            current['q'][latent_ids].cpu().numpy(), current['obj'][latent_ids].cpu().numpy())
                        decoded = decoder.decode(c[latent_rows], current['q'][latent_ids].cpu().numpy(),
                            current['obj'][latent_ids].cpu().numpy(), dt)
                        q[latent_ids] = decoded['q']
                        root = torch.zeros(n * 24, 13, device=device)
                        root[:, 6] = 1
                        future = decoder.fk.positions(q[:, 1:].reshape(-1, 18), root).reshape(n, 24, 11, 3)
                        future[gt] = reference_hand[index]
                        velocity = torch.stack([future_reference_velocity(value, dt) for value in q])
                        velocity[gt, 1:] = global_velocity[index]
                        plans.append(dict(tick=tick, c=c.copy(), q=q.cpu().numpy().copy(),
                            hand=future.cpu().numpy().copy(), velocity=velocity.cpu().numpy().copy(),
                            query_obj=current['obj'].cpu().numpy().copy()))
                    slot = tick % 8
                    shifted = (torch.arange(24, device=device) + slot).clamp_max(23)
                    next_q = q[:, slot + 1]
                    conditioning_tau = future[:, shifted].clone()
                    conditioning_tau[gt] = reference_hand[(tick + torch.arange(1, 25, device=device)).clamp_max(542)]
                    x = tau_features(current['q'], current['dq'], current['hand'], current['obj'],
                        current['velocity'], conditioning_tau, next_q, previous)
                    latent = policy.actor(x)
                    base = wrist_feedforward(next_q, velocity[:, slot + 1], ratio)
                    intended = native_action(policy.target(base, latent), current['q'], offset, scale)
                    command = intended.clamp(-1, 1)
                    if not all(torch.isfinite(value).all() for value in (x, latent, intended)):
                        raise FloatingPointError('nonfinite execution chain')
                    obs, _, done, _ = self.env_step(self.env, command.clone())
                    if capture is None or not torch.equal(capture, command):
                        raise ValueError('requested/applied command mismatch')
                    for key, value in dict(features=x, latent=latent, intended=intended, action=command,
                        applied=capture, pd_targets=pd_target,
                        clipped=(intended-command).abs().amax(-1) > 1e-6).items():
                        arrays[key].append(value.detach().cpu().numpy().copy())
                    previous = torch.tanh(latent)
                    record()
                    if tick < 541 and done.any():
                        raise ValueError('unexpected early reset/termination')
                if tick == 0 or (tick + 1) % 128 == 0:
                    state = gpu_state(args.gpu)
                    for row in state['processes'].splitlines():
                        pid, memory = row.split(',')
                        if int(pid) not in initial_pids | {os.getpid()} and int(memory) > 512:
                            raise RuntimeError('foreign GPU compute detected')
                    eta = (time.monotonic()-work_started)/(tick+1)*(542-tick-1)
                    monitor = dict(control=tick+1, eta_s=eta, gpu=state)
                    print(json.dumps(monitor), flush=True)
                    with (args.output / 'monitor.jsonl').open('a') as stream:
                        stream.write(json.dumps(monitor) + '\n')
            packed = {key: np.stack(value) for key, value in arrays.items()}
            np.savez_compressed(args.output / 'trajectory.npz', **packed)
            np.savez_compressed(args.output / 'plans.npz', **{key: np.stack([plan[key] for plan in plans]) for key in plans[0]})
            outcomes = []
            for env, role in enumerate(roles):
                outcome = episode_outcome(dict(object_pose=packed['obj'][:, env],
                    object_velocity=packed['velocity'][:, env], **{key: packed[key][:, env]
                    for key in ('surface_gap', 'support_gap', 'table_footprint')}))
                terminal_held = (packed['surface_gap'][-1, env] <= .01
                    and not (packed['table_footprint'][-1, env] and abs(packed['support_gap'][-1, env]) <= .02)
                    and packed['obj'][-1, env, 2, 3] - packed['obj'][0, env, 2, 3] >= .03)
                outcome.update(role=role, env=env, terminal_held=bool(terminal_held),
                    clipping_count=int(packed['clipped'][:, env].sum()))
                outcomes.append(outcome)
            summary = {}
            for role in sorted(set(roles)):
                rows = [row for row in outcomes if row['role'] == role]
                summary[role] = dict(episodes=4, long_held_terminal=sum(
                    row['maximum_held_frames'] >= 433 and row['terminal_held'] for row in rows),
                    terminal=sum(row['terminal_held'] for row in rows),
                    median_held=float(np.median([row['maximum_held_frames'] for row in rows])),
                    clipping_rate=sum(row['clipping_count'] for row in rows)/(4*542))
            calibrated = summary['tau_gt']['long_held_terminal'] >= 3 and summary['dense_fk']['long_held_terminal'] >= 3
            passes = [summary[role]['long_held_terminal'] >= 3 and summary[role]['clipping_rate'] < .01
                      for role in ('knots48', 'knots48_repeat')]
            status = 'PROMISING' if calibrated and all(passes) else ('UNPROMISING' if calibrated and not any(passes) else 'UNCLEAR')
            if any(sha(path) != digest for path, digest in hashes.items()) or any(
                    not torch.equal(value, weights[key]) for key, value in policy.state_dict().items()):
                raise ValueError('frozen source/weight identity drift')
            write(args.output / 'result.json', dict(status=status, summary=summary, outcomes=outcomes,
                audit_required=True, claim=manifest['claim']))
            manifest.update(status='COMPLETED', elapsed_s=time.monotonic()-started,
                torch_peak_bytes=torch.cuda.max_memory_allocated())
            write(args.output / 'manifest.json', manifest)

    native.EvalPlayer = Player
    sys.argv = [sys.argv[0], '--task', 'Dexplore_Inspire', '--cfg_env', cfg['cfg_env'],
        '--cfg_train', cfg['cfg_train'], '--checkpoint', cfg['actor'], '--motion_file', cfg['motions'],
        '--headless', '--num_envs', '16', '--seed', '297', '--sim_device', 'cuda:0', '--rl_device', 'cuda:0',
        '--pipeline', 'cpu', '--graphics_device_id', '0', '--num_threads', '1', '--disable-early-termination',
        '--output', str(args.output / 'native-unused.json'), '--output_path', str(args.output / 'native')]
    cwd = Path.cwd()
    def deadline(signum, frame):
        raise TimeoutError('decoder execution exceeded300s')
    handler = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(300)
    try:
        os.chdir(ROOT / 'third_party/DExplore')
        native.main()
    except BaseException as error:
        manifest.update(status='FAILED', elapsed_s=time.monotonic()-started, error=repr(error))
        write(args.output / 'manifest.json', manifest)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, handler)
        os.chdir(cwd)


if __name__ == '__main__':
    main()
