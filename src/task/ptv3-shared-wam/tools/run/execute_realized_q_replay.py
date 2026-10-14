"""Bounded diagnostic replay of recorded native, PD and q targets.

No learned R/forecaster or actor inference. The legacy player is only a runtime
shell. Uses the unchanged native controller and records actual physics arrays.
"""
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
RUNTIME = Path('/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train')
ROLES = ['recorded_native', 'recorded_PD', 'realized_q_full',
         'realized_q_start', 'realized_q_middle', 'realized_q_late']


def reference_chunk(data, query, kind, velocity_feedforward=False, oracle_fingers=False):
    """Absolute joint targets. Padding is only an unconsumed terminal suffix."""
    if kind == 'PD':
        return data['pd_targets'][np.minimum(query + np.arange(8), len(data['pd_targets']) - 1)].copy()
    if kind == 'q':
        indices = np.minimum(query + np.arange(1, 9), len(data['q']) - 1)
        result = data['q'][indices].copy()
        if velocity_feedforward:
            # Kd/Kp=.1s for wrist and fingers;30Hz finite differences use only
            # the same realized-q plan, never teacher actions or measured dq.
            result += 3 * (data['q'][indices] - data['q'][np.maximum(indices-1, 0)])
        if oracle_fingers:
            action_indices = np.minimum(query+np.arange(8), len(data['pd_targets'])-1)
            result[:, 6:] = data['pd_targets'][action_indices, 6:]
        return result
    if kind == 'reference_q':
        # The geometric DExplore file stores the reference robot trajectory
        # in raw columns 373:391.  The RL export overwrites those columns with
        # realized simulation q, so this array is loaded separately below.
        indices = np.minimum(query + np.arange(1, 9), len(data['reference_q']) - 1)
        return data['reference_q'][indices].copy()
    raise ValueError(kind)


def align_reference_branch(target, live_q, rotation_cls, nearest_euler, torch_module):
    """Choose the equivalent XYZ wrist Euler branch nearest the live joints.

    Geometric q_ref is an absolute pose label, not a servo winding.  PhysX can
    wrap the continuous wrist coordinates between two controls, so the branch
    is selected at every consumption tick rather than once per eight-step
    chunk.  Translation and fingers remain untouched; native coupling happens
    after this conversion.
    """
    result = target.clone()
    rotation = rotation_cls.from_euler(
        'XYZ', target[:, 3:6].detach().cpu().numpy())
    angles = nearest_euler(rotation, live_q[:, 3:6].detach().cpu().numpy())
    result[:, 3:6] = torch_module.as_tensor(angles, device=target.device, dtype=target.dtype)
    return result


def window_queries(record):
    ticks = np.asarray(record['admitted_ticks'], dtype=np.int64)
    if not len(ticks):
        raise ValueError('admitted windows required')
    return [int(ticks[i]) for i in (0, len(ticks) // 2, len(ticks) - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment-card', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--seed', type=int, default=148)
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--velocity-feedforward', action='store_true')
    parser.add_argument('--oracle-fingers', action='store_true', help='diagnostic only: recorded finger PD with q wrist')
    parser.add_argument('--q-source', choices=('realized', 'reference'), default='realized',
                        help='q target source: RL realized state or geometric DExplore reference')
    parser.add_argument('--reference-branch', choices=('raw', 'nearest-live'), default='raw',
                        help='Euler branch for reference q_ref; nearest-live is a diagnostic oracle')
    args = parser.parse_args()
    out, source = args.output.resolve(), args.source.resolve()
    if (Path(sys.prefix).resolve() != RUNTIME.resolve() or out.exists()
            or ROOT / 'outputs/ptv3-shared-wam' not in out.parents
            or not 1 <= args.seconds <= 600):
        raise ValueError('pinned runtime, fresh Task output and bounded deadline required')
    import yaml
    card = yaml.safe_load(args.experiment_card.read_text().split('---', 2)[1])
    if args.seed not in card['seeds'] or card['task'] != TASK.name:
        raise ValueError('assigned card seed and task required')
    corpus = json.loads((source / 'manifest.json').read_text())
    records = [r for r in corpus['records'] if r['sequence'] == args.sequence]
    if len(records) != 1 or records[0]['split'] != 'val' or corpus['status'] != 'COMPLETED':
        raise ValueError('one frozen dev record required')
    record = records[0]
    with np.load(source / record['path'] / 'trajectory.npz') as data:
        original = {key: data[key].copy() for key in data.files}
    controls = 4 if args.smoke else len(original['applied'])
    queries = window_queries(record)
    if subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('assigned GPU occupied')
    if os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize < 20 * 1024**3:
        raise RuntimeError('less than20GiB free')
    out.mkdir(parents=True)
    scratch = ROOT / 'tmp' / out.name
    scratch.mkdir(exist_ok=False)
    motion_meta = json.loads((Path(corpus['source_membership']) / 'manifest.json').read_text())
    motion = Path(motion_meta['source_geometric_root']) / args.sequence
    reference_file = motion / 'interaction_hand_inspire.pt'
    roles = (['recorded_native', 'recorded_PD', 'reference_q_full',
              'reference_q_start', 'reference_q_middle', 'reference_q_late']
             if args.q_source == 'reference' else ROLES)
    motions = out / 'motions'; motions.mkdir()
    (motions / args.sequence).symlink_to(motion)
    cwd = out / 'gym_runtime'
    assets = cwd / 'dexplore/data/assets'; assets.parent.mkdir(parents=True)
    assets.symlink_to('/home2/wyy/oyx_ws/dexplore/dexplore/data/assets')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(scratch), OMP_NUM_THREADS='2',
        PYTHONDONTWRITEBYTECODE='1', TORCH_EXTENSIONS_DIR=str(ROOT / 'tmp/consequence-official-generator/torch201-extensions'))
    sys.dont_write_bytecode = True
    native_root = ROOT / 'third_party/DExplore/dexplore'
    old = ROOT / 'src/task/consequence-evaluator'
    sys.path[:0] = ['/home2/wyy/isaac-gym/isaacgym/python', str(native_root), str(ROOT),
        str(old / 'src'), str(old / 'tools/audit'), str(ROOT / 'src/task/trajectory-policy/src'),
        str(ROOT / 'src/task/CmResidual/tools')]
    sys.path.append('/home2/wyy/.local/lib/python3.8/site-packages')
    from isaacgym import gymtorch  # must precede torch
    import dexplore_ddp_rank_bootstrap
    import torch
    from scipy.spatial.transform import Rotation
    sys.path.insert(0, str(TASK / 'src'))
    from ptv3_shared_wam.action_targets import _nearest_euler
    reference_raw = torch.load(reference_file, map_location='cpu', weights_only=False).numpy()
    if (reference_raw.ndim != 2 or reference_raw.shape[1] < 391
            or reference_raw.shape[0] != len(original['q'])):
        raise ValueError('geometric reference motion must match RL length and contain raw q_ref columns 373:391')
    original['reference_q'] = reference_raw[:, 373:391].copy()
    if not np.isfinite(original['reference_q']).all():
        raise ValueError('reference q contains non-finite values')
    import run as native
    from learning import common_player, dexplore_players
    from rl_games.common import player as base_player
    from probe_official_generator import install_legacy_player_compat
    from consequence_evaluator.native_reset import install_reset_patch
    from consequence_evaluator.physical_geometry import PhysicalGeometry, poses
    from consequence_evaluator.value_geometry import TableSupport
    from trajectory_policy.pd_retargeter import command, couple, pd_from_action
    if torch.__version__ != '2.0.1+cu118':
        raise ValueError('pinned torch required')
    torch.set_num_threads(2)
    install_legacy_player_compat(native, common_player, base_player)
    native.Runner.create_player = lambda runner: runner.player_factory.create(runner.algo_name, config=runner.config)
    # Archived Runner.run unconditionally passes load_path to player.restore.
    # This replay has no policy checkpoint; restore below is intentionally inert.
    native.Runner.load_path = ''
    install_reset_patch()
    started = time.monotonic()
    manifest = dict(schema='ref2dex.rl660-q-replay.v1', status='RUNNING', source=str(source),
        sequence=args.sequence, record=record, roles=roles, controls=controls,
        execution_steps=4, window_queries=queries, window_controls=8,
        seed=args.seed, physical_gpu=args.gpu, engineering_only=args.smoke,
        training_allowed=False, actor_inference=False,
        nominal_targets=('geometric q_ref[t+1]' if args.q_source == 'reference'
                         else 'recorded realized q[t+1]'),
        wrist_stiffness=200, wrist_damping=20, finger_stiffness=100, finger_damping=10,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        experiment_id=card['experiment_id'], pid=os.getpid())
    manifest['q_velocity_feedforward'] = 'finite-difference-0.1s.v1' if args.velocity_feedforward else None
    manifest['oracle_finger_PD'] = args.oracle_fingers
    manifest['q_source'] = args.q_source
    manifest['reference_branch'] = args.reference_branch if args.q_source == 'reference' else None
    manifest['reference_q_source'] = str(reference_file)
    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + '\n')
    write('manifest.json', manifest)
    logs = {}

    class Player(dexplore_players.DexplorePlayerContinuous):
        def restore(self, filename):
            # No policy is called: the player exists only to construct env/step.
            pass

        @torch.no_grad()
        def _run_export_rl(self, task):
            if task.num_envs != len(roles) or len(task.motion_file) != 1:
                raise ValueError('six paired rows and one unique motion required')
            task._adaptive_kappa_enabled = False
            task._enable_early_termination = False
            task._hybrid_init_prob = 1.
            task.max_episode_length[:] = len(original['q'])
            task.rollout_length = len(original['q'])
            self.has_batch_dimension = True
            self.env_reset()
            if (task.start_times != 0).any():
                raise ValueError('frame0 reset required')
            geometry = PhysicalGeometry(task, assets, distance_device=task.device)
            support = TableSupport(assets, task.device)
            offset, scale = task._pd_action_offset.clone(), task._pd_action_scale.clone()
            actual_controller = dict(offset=offset.tolist(), scale=scale.tolist())
            if actual_controller != corpus['native_controller']:
                raise ValueError('controller differs from source')
            properties = task.gym.get_actor_dof_properties(task.envs[0], task.humanoid_handles[0])
            assert np.array_equal(properties['stiffness'], [200]*6 + [100]*12)
            assert np.array_equal(properties['damping'], [20]*6 + [10]*12)
            manifest['native_controller'] = actual_controller
            captured = actual_pd = None
            original_pre, original_pd = task.pre_physics_step, task._action_to_pd_targets
            def pre(action):
                nonlocal captured
                captured = action.clone()
                return original_pre(action)
            def pd(action):
                nonlocal actual_pd
                actual_pd = original_pd(action).clone()
                return actual_pd
            task.pre_physics_step, task._action_to_pd_targets = pre, pd
            for key in ('q', 'dq', 'hand', 'obj', 'velocity', 'surface_gap', 'support_gap',
                        'table_footprint', 'contact_force', 'applied', 'intended', 'pd_targets',
                        'requested_pd', 'clipped', 'query_tick', 'chunk_index', 'q_intervention'):
                logs[key] = []
            def measure():
                hand, gap = geometry.measure(task)
                sup, foot = support.measure(task, geometry)
                current = dict(q=task._dof_pos.clone(), dq=task._dof_vel.clone(), hand=hand,
                    obj=poses(task._target_states), velocity=task._target_states[:, 7:13].clone(),
                    surface_gap=gap, support_gap=sup, table_footprint=foot,
                    contact_force=task._contact_forces[:, task._contact_body_ids].clone())
                for key, value in current.items():
                    logs[key].append(value.cpu().numpy().copy())
                return current
            current = measure()
            reset = {key: float(np.abs(logs[key][0] - original[key][0]).max())
                     for key in ('q', 'dq', 'hand', 'obj', 'velocity')}
            write('reset_comparison.json', reset)
            if max(reset.values()) > 1e-4:
                raise ValueError('reset differs from source: ' + str(reset))
            buffers, clock = {}, {}
            fork_errors = {}
            for tick in range(controls):
                intended = torch.as_tensor(np.repeat(original['applied'][tick:tick+1], len(roles), axis=0), device=task.device)
                requested = pd_from_action(intended, current['q'], offset, scale)
                clipped = torch.zeros(len(roles), dtype=torch.bool, device=task.device)
                query = np.full(len(roles), -1, np.int64)
                index = query.copy(); intervention = np.zeros(len(roles), bool)
                for env in range(1, len(roles)):
                    start = 0 if env < 3 else queries[env - 3]
                    active = env < 3 or start <= tick < start + 8
                    if not active:
                        continue
                    if env >= 3 and tick == start:
                        fork_errors[roles[env]] = {key: float((current[key][env] - current[key][0]).abs().max())
                                                   for key in ('q', 'dq', 'hand', 'obj', 'velocity')}
                    if (tick - start) % 4 == 0:
                        if env == 1:
                            kind = 'PD'
                        else:
                            kind = 'reference_q' if args.q_source == 'reference' else 'q'
                        buffers[env] = torch.as_tensor(reference_chunk(original, tick, kind, args.velocity_feedforward, args.oracle_fingers), device=task.device)
                        clock[env] = tick
                    query[env], index[env] = clock[env], tick - clock[env]
                    target = buffers[env][index[env]:index[env]+1]
                    if (args.q_source == 'reference' and args.reference_branch == 'nearest-live'
                            and env != 1):
                        target = align_reference_branch(target, current['q'][env:env+1], Rotation,
                                                        _nearest_euler, torch)
                    target = couple(target)
                    raw, _, clip = command(target, current['q'][env:env+1], offset, scale)
                    intended[env], requested[env], clipped[env] = raw[0], target[0], clip[0]
                    intervention[env] = env != 1
                action = intended.clamp(-1, 1)
                self.env_step(self.env, action.clone())
                if not torch.equal(captured, action):
                    raise ValueError('actual command differs')
                expected = pd_from_action(action, current['q'], offset, scale)
                if (actual_pd - expected).abs().max() > 1e-6:
                    raise ValueError('same-tick PD conversion differs')
                for key, value in dict(applied=action, intended=intended, requested_pd=requested,
                    pd_targets=actual_pd, clipped=clipped).items():
                    logs[key].append(value.cpu().numpy().copy())
                for key, value in dict(query_tick=query, chunk_index=index, q_intervention=intervention).items():
                    logs[key].append(value.copy())
                current = measure()
                if tick == 0 or (tick + 1) % 32 == 0:
                    pids = set(map(int, subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                        '--query-compute-apps=pid', '--format=csv,noheader'], text=True).split()))
                    if pids - {os.getpid()}:
                        raise RuntimeError('foreign GPU process')
                    elapsed = time.monotonic() - started
                    row = dict(control=tick+1, controls=controls, elapsed_s=elapsed,
                        eta_s=elapsed/(tick+1)*(controls-tick-1),
                        gpu=subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                            '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'], text=True).strip())
                    write('progress.json', row); print(json.dumps(row), flush=True)
                    if os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize < 20 * 1024**3:
                        raise RuntimeError('free space below20GiB')
            manifest['fork_errors'] = fork_errors

    dexplore_players.DexplorePlayerContinuous = Player
    cfg = ROOT / 'outputs/consequence-evaluator/baseline-transfer-inputs-20261007-r1/cfg/inspire.yaml'
    sys.argv = [sys.argv[0], '--task', 'Dexplore_Inspire', '--cfg_env', str(cfg), '--cfg_train',
        str(native_root / 'data/cfg/train/rlg/inspire.yaml'), '--test', '--headless', '--motion_file', str(motions),
        '--num_envs', str(len(roles)), '--export_rl', '--export_output_dir', str(out), '--seed', str(args.seed),
        '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--pipeline', 'gpu', '--graphics_device_id', '0',
        '--num_threads', '1', '--output_path', str(out / 'native')]
    def timeout(number, frame):
        raise TimeoutError('bounded replay deadline')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(args.seconds)
    try:
        os.chdir(cwd)
        native.main()
        manifest.update(status='COMPLETED', elapsed_s=time.monotonic()-started)
    except BaseException as error:
        manifest.update(status='FAILED', error=repr(error), elapsed_s=time.monotonic()-started)
        raise
    finally:
        signal.alarm(0)
        if logs and logs.get('q'):
            np.savez_compressed(out / 'trajectory.npz', **{key: np.stack(value) for key, value in logs.items() if value})
        write('manifest.json', manifest)


if __name__ == '__main__':
    main()
