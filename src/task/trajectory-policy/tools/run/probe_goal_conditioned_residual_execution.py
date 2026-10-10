"""Execute a frozen goal-conditioned residual policy through the shared executor.

The nominal q/velocity plan is read from the retained geometry cache.  That is
an explicitly privileged nominal-reference integration diagnostic: it tests the
student residual policy and executor wiring, not a deployable no-future-q
high-level policy.
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
    for key in ('inputs', 'reference', 'geometry', 'residual-checkpoint',
                'residual-manifest', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--goal-env', type=int, default=0)
    parser.add_argument('--seed', type=int, default=314)
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
    from consequence_evaluator.reference_tracking import reference_velocity, RESIDUAL_LIMITS
    from consequence_evaluator.native_reset import install_reset_patch
    from consequence_evaluator.physical_geometry import PhysicalGeometry, poses
    from consequence_evaluator.value_geometry import TableSupport
    from consequence_evaluator.gate1 import episode_outcome
    from trajectory_policy.goal_executor import GoalConditionedExecutor
    from trajectory_policy.goal_policy import (
        RESIDUAL_GOAL_HORIZON, GoalConditionedResidualPolicy,
        ResidualStandardizer, residual_goal_features,
    )
    from trajectory_policy.inputs import load_geometry
    from oracle_y_utility import align_native_reference_tables
    import evaluate as native
    from env.tasks.base_dexplore_task import DexploreTask
    from rl_games.torch_runner import Runner as RLRunner

    if torch.__version__ != '2.4.1+cu121':
        raise ValueError('pinned graspenv runtime required')
    # The pinned rl_games package calls player_factory.create(..., config=...),
    # while this checkout's historical DExplore evaluator registers its player
    # builder with the older ``params`` keyword.  Keep the compatibility shim
    # local to this Probe rather than mutating the external evaluator source.
    def create_player_compat(runner):
        return runner.player_factory.create(runner.algo_name, params=runner.config)
    RLRunner.create_player = create_player_compat
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0')
    urdf = ROOT / 'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    geometry_data, initial, _, hashes = load_geometry(args.geometry, args.reference, urdf)
    cfg = json.loads(args.inputs.read_text())
    essential = [cfg['actor'], cfg['cfg_env'], cfg['cfg_train']]
    essential += [str(ROOT / 'third_party/DExplore/dexplore' / file) for file in (
        'evaluate.py', 'env/tasks/base_dexplore_task.py', 'env/tasks/dexplore_inspire.py')]
    essential += [path for path in cfg['input_sha256'] if path.endswith('interaction_hand_inspire.pt')]
    for path in essential:
        if sha(path) != cfg['input_sha256'][path]:
            raise ValueError('native input drift: ' + path)
    historical_native_drift = {}
    historical_native_allowlist = {
        str((OLD / 'tools/run/run_hand_bridge_rollout.py').resolve()),
    }
    for path, digest in cfg['input_sha256'].items():
        current = sha(path)
        if current != digest:
            if path in historical_native_allowlist:
                historical_native_drift[path] = dict(declared=digest, current=current)
            else:
                raise ValueError('native input provenance drift: ' + path)
    residual_manifest = json.loads(args.residual_manifest.read_text())
    residual_result = json.loads((args.residual_checkpoint.parent / 'result.json').read_text())
    if (residual_manifest.get('status') != 'COMPLETED'
            or residual_manifest.get('schema') != 'ref2dex.goal-conditioned-residual-bc-run.v1'
            or residual_result.get('status') != 'PROMISING'
            or not residual_result.get('offline_only')):
        raise ValueError('completed promising residual BC Probe required')
    if sha(args.residual_checkpoint) != residual_result.get('checkpoint_sha256'):
        raise ValueError('residual checkpoint hash mismatch')
    for path, digest in residual_manifest.get('input_sha256', {}).items():
        if sha(path) != digest:
            raise ValueError('residual source/input drift: ' + path)
    payload = torch.load(args.residual_checkpoint, map_location=device, weights_only=False)
    if payload.get('schema') != GoalConditionedResidualPolicy.schema:
        raise ValueError('residual checkpoint schema mismatch')
    if payload.get('input_dim') != residual_manifest.get('input_dim'):
        raise ValueError('residual input dimension provenance mismatch')
    if payload.get('goal_horizon') != RESIDUAL_GOAL_HORIZON:
        raise ValueError('residual horizon must be 24')
    policy = GoalConditionedResidualPolicy(width=int(payload['width'])).to(device)
    policy.load_state_dict(payload['state_dict'], strict=True)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    weights = {key: value.clone() for key, value in policy.state_dict().items()}
    standardizer = ResidualStandardizer(
        np.asarray(payload['standardizer']['mean'], np.float32),
        np.asarray(payload['standardizer']['scale'], np.float32),
    )
    feature_mean = torch.as_tensor(standardizer.mean, device=device)
    feature_scale = torch.as_tensor(standardizer.scale, device=device)
    reference_q = torch.as_tensor(geometry_data['q'], device=device)
    if reference_q.shape != (543, 18) or not torch.isfinite(reference_q).all():
        raise ValueError('geometry cache must contain a finite 543-frame q plan')
    roles = ['goal_residual'] * 16
    with args.reference.open('rb') as stream:
        import pickle
        goal_packet = pickle.load(stream)
    if goal_packet.get('engineering_only') is not True or goal_packet.get('training_allowed') is not False:
        raise ValueError('goal reference must remain goal-only engineering packet')
    if args.goal_env < 0 or args.goal_env >= int(goal_packet['hand_keypoints'].shape[1]):
        raise ValueError('goal-env outside reference packet')
    goal_hand = torch.as_tensor(goal_packet['hand_keypoints'][:, args.goal_env], device=device)
    goal_obj = torch.as_tensor(goal_packet['object_pose'][:, args.goal_env], device=device)
    if goal_hand.shape != (543, 11, 3) or goal_obj.shape != (543, 4, 4):
        raise ValueError('goal reference must contain 543 hand/object frames')
    if not torch.isfinite(goal_hand).all() or not torch.isfinite(goal_obj).all():
        raise ValueError('nonfinite goal reference')
    hashes.update({
        str(Path(path).resolve()): historical_native_drift.get(path, {}).get('current', digest)
        for path, digest in cfg['input_sha256'].items()
    })
    hashes.update({
        str(args.inputs.resolve()): sha(args.inputs),
        str(args.reference.resolve()): sha(args.reference),
        str(args.residual_checkpoint.resolve()): sha(args.residual_checkpoint),
        str(args.residual_manifest.resolve()): sha(args.residual_manifest),
        str((args.residual_checkpoint.parent / 'result.json').resolve()):
            sha(args.residual_checkpoint.parent / 'result.json'),
        str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
        str((TASK / 'src/trajectory_policy/goal_policy.py').resolve()):
            sha(TASK / 'src/trajectory_policy/goal_policy.py'),
        str((TASK / 'src/trajectory_policy/goal_policy_data.py').resolve()):
            sha(TASK / 'src/trajectory_policy/goal_policy_data.py'),
        str((TASK / 'src/trajectory_policy/goal_executor.py').resolve()):
            sha(TASK / 'src/trajectory_policy/goal_executor.py'),
    })
    args.output.mkdir(parents=True)
    manifest = dict(schema='ref2dex.goal-conditioned-residual-execution.v1', status='RUNNING',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        physical_gpu=args.gpu, gpu_before=before, seed=args.seed, roles=roles,
        input_sha256=hashes, goal_env=args.goal_env,
        input_dim=payload['input_dim'], residual_dim=12,
        goal_horizon=RESIDUAL_GOAL_HORIZON, execute_steps=8,
        inference_contract=payload['inference_contract'],
        previous_residual_contract=payload['previous_residual_contract'],
        nominal_reference_contract=(
            'privileged geometry cache q and position-derived velocity; integration diagnostic only; '
            'not deployable no-future-q nominal reference'),
        goal_authorization='engineering_only=true, training_allowed=false packet used only for hand/object goals',
        action_contract='raw residual latent -> tanh bounded residual -> shared executor target/coupling/native clamp',
        replan_interval=1, controls=542, envs=16, dt=1/30,
        gate=dict(long_held_terminal_min=12, clipping_rate_max=.01),
        claim='privileged nominal-reference residual-policy integration Probe; no deployable H or Cm claim')
    manifest['historical_native_source_drift'] = historical_native_drift
    manifest['rl_games_compatibility'] = 'local Runner.create_player shim maps config to historical params builder'
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
            offset, native_scale = task._pd_action_offset.to(device), task._pd_action_scale.to(device)
            props = task.gym.get_actor_dof_properties(task.envs[0], task.humanoid_handles[0])
            ratio = torch.as_tensor(props['damping'][:6].copy() / props['stiffness'][:6].copy(), device=device)
            dt = task.gym.get_sim_params(task.sim).dt * task.control_freq_inv
            if abs(dt - 1/30) > 1e-8 or not torch.allclose(ratio, torch.full_like(ratio, .1)):
                raise ValueError('fixed native gains/control interval mismatch')
            manifest['native_controller'] = dict(offset=offset.cpu().tolist(), scale=native_scale.cpu().tolist(),
                gain_ratio=ratio.cpu().tolist())
            executor = GoalConditionedExecutor(offset, native_scale, ratio, RESIDUAL_LIMITS,
                                                execute_steps=8)
            nominal_velocity = reference_velocity(reference_q, dt)
            manifest['native_controller'].update(
                residual_limits=list(RESIDUAL_LIMITS), control_dt=float(dt),
                privileged_nominal_q_source=str((args.geometry / 'geometry.npz').resolve()),
                privileged_nominal_velocity_contract='reference_velocity(geometry q, control_dt)')
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
            arrays = {key: [] for key in (
                'q', 'dq', 'hand', 'obj', 'velocity', 'surface_gap', 'support_gap',
                'table_footprint', 'features', 'previous_residual', 'latent',
                'bounded_residual', 'nominal_q', 'nominal_velocity', 'target',
                'intended', 'action', 'applied', 'pd_targets', 'clipped')}
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
                    index = (tick + 1 + torch.arange(RESIDUAL_GOAL_HORIZON,
                                                       device=device)).clamp_max(542)
                    hand_goal = goal_hand[index][None].expand(n, -1, -1, -1)
                    obj_goal = goal_obj[index][None].expand(n, -1, -1, -1)
                    raw_features = residual_goal_features(
                        current['q'], current['dq'], current['hand'], current['obj'],
                        current['velocity'], previous, hand_goal, obj_goal)
                    encoded = (raw_features - feature_mean) / feature_scale
                    latent = policy(encoded)
                    bounded = torch.tanh(latent)
                    nominal_q = reference_q[tick + 1].expand(n, -1)
                    nominal_v = nominal_velocity[tick + 1].expand(n, -1)
                    executed = executor.step(current['q'], nominal_q, nominal_v, latent)
                    target = executed['target']
                    intended, command = executed['intended'], executed['command']
                    clipped = executed['clipped']
                    if not all(torch.isfinite(value).all() for value in (
                            raw_features, encoded, latent, bounded, nominal_q,
                            nominal_v, target, intended, command)):
                        raise FloatingPointError('nonfinite execution chain')
                    obs, _, done, _ = self.env_step(self.env, command.clone())
                    if capture is None or not torch.equal(capture, command):
                        raise ValueError('requested/applied command mismatch')
                    if pd_target is None:
                        raise RuntimeError('native PD target capture was not populated')
                    for key, value in dict(
                            features=raw_features, previous_residual=previous,
                            latent=latent, bounded_residual=bounded,
                            nominal_q=nominal_q, nominal_velocity=nominal_v,
                            target=target, intended=intended, action=command,
                            applied=capture, pd_targets=pd_target, clipped=clipped).items():
                        arrays[key].append(value.detach().cpu().numpy().copy())
                    previous = bounded
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
            outcomes = []
            for env, role in enumerate(roles):
                outcome = episode_outcome(dict(object_pose=packed['obj'][:, env],
                    object_velocity=packed['velocity'][:, env], **{key: packed[key][:, env]
                    for key in ('surface_gap', 'support_gap', 'table_footprint')}))
                terminal_held = (packed['surface_gap'][-1, env].max() <= .01
                    and not (packed['table_footprint'][-1, env].any()
                             and abs(packed['support_gap'][-1, env].min()) <= .02)
                    and packed['obj'][-1, env, 2, 3] - packed['obj'][0, env, 2, 3] >= .03)
                outcome.update(role=role, env=env, terminal_held=bool(terminal_held),
                    clipping_count=int(packed['clipped'][:, env].sum()))
                outcomes.append(outcome)
            summary = dict(
                episodes=len(outcomes),
                long_held_terminal=sum(row['maximum_held_frames'] >= 433 and row['terminal_held']
                                       for row in outcomes),
                hold45=sum(row['maximum_held_frames'] >= 45 for row in outcomes),
                terminal=sum(row['terminal_held'] for row in outcomes),
                median_held=float(np.median([row['maximum_held_frames'] for row in outcomes])),
                clipping_rate=sum(row['clipping_count'] for row in outcomes)/(len(outcomes)*542),
            )
            status = ('PROMISING' if summary['long_held_terminal'] >= 12
                      and summary['clipping_rate'] < .01 else 'UNPROMISING')
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
        '--headless', '--num_envs', '16', '--seed', str(args.seed), '--sim_device', 'cuda:0', '--rl_device', 'cuda:0',
        '--pipeline', 'cpu', '--graphics_device_id', '0', '--num_threads', '1', '--disable-early-termination',
        '--output', str(args.output / 'native-unused.json'), '--output_path', str(args.output / 'native')]
    cwd = Path.cwd()
    def deadline(signum, frame):
        raise TimeoutError('residual execution exceeded300s')
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
