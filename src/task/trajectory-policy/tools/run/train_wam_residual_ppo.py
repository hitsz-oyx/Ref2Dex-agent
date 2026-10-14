"""Phase-randomized residual PPO on the frozen context WAM and old chunk R.

The base command is the frozen pair. PPO adds a 12-D residual and is scored
on the airplane outcome, not on an expert action.
"""
import argparse
import json
import os
import random
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
OLD = ROOT / 'src/task/consequence-evaluator'
RUNTIME = Path('/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train')
GRASPENV = '/home2/wyy/miniconda3/envs/graspenv/bin/python'
EXP = 'P-20261013-wam-residual-ppo'
RUN = 'wam-ref2-execution-rppo-20261014-r6'
HEAD = ROOT / 'outputs/trajectory-policy/wam-ref2-joint-multitakeover-20261012-r1'
CHUNK = ROOT / 'outputs/trajectory-policy/wam-v2-control-fit-20261012-r1'
EFFECT = ROOT / 'outputs/goal-point-wam/wam-effect-ddp-20261013-r3/train/checkpoint_step50500.pt'
CONTROL = ROOT / 'outputs/goal-point-wam/wam-control-context-20261013-r3/stage2_step4500.pt'
N = 4
DIAGNOSTIC = 96
UPDATES = 20
ROLLOUT = 96
DEADLINE_S = 14400
ACTOR_LR = 3e-4
EVAL = 542
PHASES = (0, 32, 48, 62)
CURRICULUM_EXP = 'P-20261014-wam-residual-curriculum'
MAX_UPDATES = 100


def rpc_budget(updates, envs=N, phases=PHASES, eval_every=0):
    """One packet per control step. The server counts RPCs, not environments."""
    return query_budget(updates, envs, phases, eval_every, ROLLOUT, DIAGNOSTIC, EVAL)
sys.path[:0] = [str(TASK / 'src'), str(TASK / 'tools/audit'), str(TASK / 'tools/run')]
from audit_wam_ref2_shadow import sha, write
from trajectory_policy.wam_ipc import SCHEMA, send, receive, peer_pid
from collect_competent_pd import OFFICIAL, OFFICIAL_SHA, gpu_state
from trajectory_policy.residual_curriculum import (
    curriculum_phases, diagnostic_phases, query_budget, contact_hold_summary,
)


def window(rows, env):
    if len(rows) < 4:
        raise ValueError('phase context must contain three real past measurements and current state')
    frames = [row[env] for row in rows]
    return np.stack(frames[-4:])


def label_probe(outcomes, close_rise, thrown, residual_rms):
    hold = max(item['maximum_held_frames'] for item in outcomes)
    if hold >= 45 or (close_rise > 0.0425 and not thrown):
        return 'PROMISING'
    if residual_rms < 0.05:
        return 'UNCLEAR'
    return 'UNPROMISING'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--envs', type=int, default=N)
    parser.add_argument('--updates', type=int, default=UPDATES)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--curriculum', choices=('uniform', 'contact'), default='uniform')
    parser.add_argument('--allow-curriculum-change', action='store_true')
    parser.add_argument('--eval-every', type=int, default=0,
                        help='additional training updates between full frame-0 evaluations; 0 means final only')
    parser.add_argument('--experiment-id', choices=(EXP, CURRICULUM_EXP), default=EXP)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/trajectory-policy' / RUN)
    args = parser.parse_args()
    if not 1 <= args.envs <= 16 or not 1 <= args.updates <= MAX_UPDATES or args.eval_every < 0:
        raise ValueError('bounded probe stays within 16 envs and 100 additional updates')
    if args.allow_curriculum_change and args.resume is None:
        raise ValueError('curriculum-change continuation requires a v2 parent checkpoint')
    args.rpcs = rpc_budget(args.updates, args.envs, curriculum_phases(args.curriculum, args.envs), args.eval_every)
    if args.rpcs > 200000:
        raise ValueError('server query budget exceeded')
    out = args.output.resolve()
    scratch = ROOT / 'tmp' / out.name
    address = scratch / 'w.sock'
    if (out.exists() or scratch.exists() or not out.name.startswith('wam-ref2-execution-')
            or ROOT / 'outputs/trajectory-policy' not in out.parents
            or len(os.fsencode(str(address))) >= 108
            or Path(sys.prefix).resolve() != RUNTIME.resolve()):
        raise ValueError('fresh owned output, short socket, and pinned native runtime required')
    if subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid',
                                '--format=csv,noheader,nounits'], text=True).strip():
        raise RuntimeError('GPU occupied before residual PPO')
    scratch.mkdir(mode=0o700)
    out.mkdir()
    server_script = TASK / 'tools/run/wam_native_server.py'
    server_log = (out / 'server.log').open('w')
    server = subprocess.Popen(
        [GRASPENV, str(server_script), '--gpu', str(args.gpu), '--controls', str(args.rpcs),
         '--experiment-id', args.experiment_id, '--head-dir', str(HEAD), '--socket', str(address),
         '--output', str(out / 'server'), '--chunk-dir', str(CHUNK),
         '--effect-checkpoint', str(EFFECT), '--control-checkpoint', str(CONTROL), '--open-queries'],
        cwd=ROOT, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    def stop_server():
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)

    try:
        for _ in range(240):
            if server.poll() is not None:
                raise RuntimeError('inference server exited before READY')
            readiness = out / 'server/status.json'
            if readiness.exists() and json.loads(readiness.read_text())['status'] == 'READY':
                break
            time.sleep(0.5)
        else:
            raise TimeoutError('inference server not READY within 120s')
        run_native(args, out, address, server)
        server.wait(timeout=60)
        if server.returncode != 0:
            raise RuntimeError('server terminal exit failure')
    finally:
        stop_server()
        server_log.close()


def run_native(args, out, address, server):
    N = args.envs
    UPDATES = args.updates
    QUERIES = args.rpcs
    phases = curriculum_phases(args.curriculum, N)
    native_root = ROOT / 'third_party/DExplore/dexplore'
    sys.path[:0] = [str(Path('/home2/wyy/isaac-gym/isaacgym/python')), str(OLD / 'src'),
                    str(OLD / 'tools/audit'), str(ROOT), str(native_root), str(ROOT / 'src/task/CmResidual/tools')]
    sys.path.append('/home2/wyy/.local/lib/python3.8/site-packages')
    from isaacgym import gymtorch  # must initialize before Torch
    import dexplore_ddp_rank_bootstrap
    import torch
    import evaluate as native
    from learning import common_player
    from rl_games.common import player as base_player
    from probe_official_generator import install_legacy_player_compat, model_state
    from consequence_evaluator.native_reset import install_reset_patch
    from consequence_evaluator.physical_geometry import PhysicalGeometry, poses
    from consequence_evaluator.value_geometry import TableSupport
    from consequence_evaluator.gate1 import episode_outcome
    from consequence_evaluator.reset_kinematics import task_kinematics
    from env.tasks.base_dexplore_task import DexploreTask
    from trajectory_policy.chunk_retargeter import load_chunk
    from trajectory_policy.pd_retargeter import pd_from_action
    from trajectory_policy.residual_ppo import (
        ALPHA, ResidualActor, ResidualValue, apply_residual, object_distance, observation,
        residual_reward, rollout_bootstrap, residual_execution_stats, update,
    )
    from trajectory_policy.residual_reset import reset_context, teacher_prefix_action
    from trajectory_policy.retargeter_data import ACTIVE, features
    from trajectory_policy.robot_context import control_context
    from trajectory_policy.wam_effect_admission import chunk_admission, effect_admission
    from trajectory_policy.wam_control_admission import control_admission
    from wam_native_server import scene_patch_admission
    del gymtorch, dexplore_ddp_rank_bootstrap, model_state
    if torch.__version__ != '2.0.1+cu118' or sha(OFFICIAL) != OFFICIAL_SHA:
        raise ValueError('pinned native runtime/teacher identity required')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(296)
    actor = ResidualActor().cuda()
    value_net = ResidualValue().cuda()
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)
    resume_updates = 0
    resumed = None
    if args.resume is not None:
        resumed = torch.load(args.resume, map_location='cuda:0')
        if (resumed.get('schema') != 'ref2dex.wam-residual-ppo.v2'
                or resumed.get('alpha') != ALPHA or len(resumed.get('phases', ())) != N
                or (resumed.get('phases') != list(phases) and not args.allow_curriculum_change)
                or any(key not in resumed for key in ('actor_optimizer', 'value_optimizer', 'rng', 'prefix'))):
            raise ValueError('v2 training state and matching env count required; phase changes must be explicit')
        if max(phases) >= len(resumed['prefix']['obj']):
            raise ValueError('parent measured prefix does not cover requested reset phases')
        actor.load_state_dict(resumed['actor'])
        value_net.load_state_dict(resumed['value'])
        actor_optimizer.load_state_dict(resumed['actor_optimizer'])
        value_optimizer.load_state_dict(resumed['value_optimizer'])
        resume_updates = int(resumed['update'])
    actor_initial = {key: item.detach().clone() for key, item in actor.state_dict().items()}
    torch.manual_seed(292)
    random.seed(296)
    np.random.seed(296)
    noise = torch.Generator(device='cuda')
    noise.manual_seed(296)
    run_dir = ROOT / 'outputs/consequence-evaluator/baseline-transfer-s3-20261007-r2/airplane_base'
    cfg = json.loads((run_dir / 'config.json').read_text())
    reference_path = ROOT / 'outputs/trajectory-policy/competent-reference-20261011-r1/reference.npz'
    with np.load(reference_path, allow_pickle=False) as stream:
        reference = {key: stream[key].copy() for key in stream.files}
    r_path, r_identity, _hashes = chunk_admission(CHUNK)
    chunk_packet = torch.load(r_path, map_location='cuda:0')
    retargeter = load_chunk(chunk_packet, 'cuda:0')
    for parameter in retargeter.parameters():
        parameter.requires_grad_(False)
    head_packet = torch.load(HEAD / 'head.pt', map_location='cpu')
    head_manifest = json.loads((HEAD / 'manifest.json').read_text())
    head_result = json.loads((HEAD / 'result.json').read_text())
    patch_identity = scene_patch_admission(head_packet, head_manifest, head_result)
    effect_identity, _values = effect_admission(EFFECT, HEAD, CHUNK)
    control_identity, _packet = control_admission(CONTROL, effect_identity, r_identity)
    del head_packet, _values, _packet
    base_path = ROOT / 'outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt'
    base_hash = sha(base_path)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(180)
    connection.connect(str(address))
    if peer_pid(connection) != server.pid:
        raise ValueError('wrong local inference server PID')
    handshake = receive(connection)
    if (handshake['schema'] != SCHEMA or handshake['pid'] != server.pid
            or handshake['physical_gpu'] != args.gpu or handshake.get('controls') != QUERIES
            or handshake.get('experiment_id') != args.experiment_id or handshake['base_sha256'] != base_hash
            or handshake.get('scene_patch_identity') != patch_identity
            or handshake.get('retargeter_identity') != r_identity
            or handshake.get('effect_identity') != effect_identity
            or handshake.get('control_identity') != control_identity):
        raise ValueError('WAM inference peer identity differs')
    started = time.monotonic()
    queries = 0
    manifest = dict(schema='ref2dex.wam-residual-ppo.v2', status='RUNNING', seed=292,
                    ppo_seed=296, physical_gpu=args.gpu, experiment_id=args.experiment_id, run_id=out.name,
                    alpha=ALPHA, phases=list(phases), envs=N, queries=QUERIES, updates=UPDATES,
                    resume_updates=resume_updates,
                    resume=None if args.resume is None else str(args.resume),
                    batched_wam=True, rollout=ROLLOUT, diagnostic_steps=DIAGNOSTIC, eval_steps=EVAL,
                    frozen_world=True, frozen_wam=True, frozen_R=True,
                    reward='object progress, contact, hold, slip, residual penalty',
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    server_pid=server.pid, native_pid=os.getpid(),
                    retargeter_identity=r_identity, effect_identity=effect_identity,
                    control_identity=control_identity)
    manifest.update(gae_lambda=1., gamma=0.99, bootstrap='V(actual post-control observation), terminal masked',
                    reset_context='measured teacher prefix; current reset includes declared noise',
                    residual_parameterization='Gaussian latent; clip(base + alpha * latent)',
                    active_dofs=list(ACTIVE), checkpoint='optimizer, RNG, prefix and update state')
    manifest.update(curriculum=args.curriculum, eval_every=args.eval_every,
                    allow_curriculum_change=args.allow_curriculum_change,
                    parent_phases=None if resumed is None else resumed['phases'],
                    diagnostic_phases=list(diagnostic_phases(phases)))
    write(out / 'manifest.json', manifest)
    install_legacy_player_compat(native, common_player, base_player)
    install_reset_patch()
    metrics = []

    class Player(native.EvalPlayer):
        def run(self):
            nonlocal queries
            task = self.env.task
            if task.num_envs != N or self.is_rnn or str(task.device) != 'cuda:0':
                raise ValueError('requested nonrecurrent GPU env count required')
            task._state_init = DexploreTask.StateInit.Start
            task._hybrid_init_prob = 1.
            task._adaptive_kappa_enabled = False
            task._enable_early_termination = False
            self.env_reset(torch.arange(N, device=task.device))
            if (task.start_times != 0).any() or (task.progress_buf != 0).any():
                raise ValueError('frame0 reset required')
            geometry = PhysicalGeometry(task, native_root / 'data/assets', distance_device=task.device)
            support = TableSupport(native_root / 'data/assets', task.device)
            offset, scale = task._pd_action_offset.clone(), task._pd_action_scale.clone()
            if dict(offset=offset.tolist(), scale=scale.tolist()) != chunk_packet['native_controller']:
                raise ValueError('native Chunk-R control units differ')
            names = task.gym.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles[0])
            if len(names) < 11:
                raise ValueError('hand body names missing')
            actual_pd = capture = None
            original_pre, original_pd = task.pre_physics_step, task._action_to_pd_targets

            def pre(action):
                nonlocal capture
                capture = action.clone()
                return original_pre(action)

            def pd(action):
                nonlocal actual_pd
                actual_pd = original_pd(action).clone()
                return actual_pd

            task.pre_physics_step, task._action_to_pd_targets = pre, pd

            def measure():
                hand, gap = geometry.measure(task)
                sup, foot = support.measure(task, geometry)
                if hand.shape != (N, 11, 3):
                    raise ValueError('11 native hand points required')
                return dict(q=task._dof_pos.clone(), dq=task._dof_vel.clone(), hand=hand,
                            obj=poses(task._target_states), velocity=task._target_states[:, 7:13].clone(),
                            surface_gap=gap, support_gap=sup, table_footprint=foot)

            # Capture an actual, bounded teacher prefix once. Restoring a
            # demonstration q alone cannot reconstruct measured H4/actions.
            teacher_rows = []
            teacher_actions = []
            teacher_snapshots = []
            obs = self.env_reset(torch.arange(N, device=task.device))
            for tick in range(max(PHASES) + 1):
                teacher_rows.append({key: item.detach().cpu().numpy().copy() for key, item in measure().items()})
                teacher_snapshots.append({key: item.detach().cpu().numpy().copy() for key, item in dict(
                    q=task._dof_pos, dq=task._dof_vel, root=task._humanoid_root_states,
                    target=task._target_states, table=task._table_states).items()})
                if tick == max(PHASES):
                    break
                with torch.no_grad():
                    teacher = teacher_prefix_action(self, obs, N)
                obs, _, done, _ = self.env_step(self.env, teacher.clone())
                if done.any() or capture is None or not torch.equal(capture, teacher):
                    raise ValueError('teacher prefix terminated or applied command differed')
                teacher_actions.append(capture.detach().cpu().numpy().copy())
            prefix = {key: np.stack([row[key] for row in teacher_rows]) for key in teacher_rows[0]}
            prefix['applied'] = np.stack(teacher_actions)
            prefix['snapshots'] = {key: np.stack([row[key] for row in teacher_snapshots])
                                   for key in teacher_snapshots[0]}
            if resumed is not None:
                prefix = resumed['prefix']
            np.savez_compressed(out / 'reset-prefix.npz', **{key: item for key, item in prefix.items() if key != 'snapshots'},
                                **{'snapshot_' + key: item for key, item in prefix['snapshots'].items()})

            def place(frames, perturb=True):
                frames = torch.as_tensor(frames, dtype=torch.long, device=task.device)
                ids = torch.arange(N, device=task.device)
                task.ref_index[:] = 0
                task.progress_buf[:] = frames
                task.start_times[:] = frames
                task._hist_obs[:] = 0
                task.contact_reset[:] = 0
                selected = frames.cpu().numpy(), np.arange(N)
                for key, destination in dict(q=task._dof_pos, dq=task._dof_vel, root=task._humanoid_root_states,
                                             target=task._target_states, table=task._table_states).items():
                    destination[:] = torch.as_tensor(prefix['snapshots'][key][selected], device=task.device)
                if perturb:
                    task._dof_pos += (torch.rand(task._dof_pos.shape, device=task.device, generator=noise) - 0.5) * 0.02
                    task._target_states[:, 0:3] += (torch.rand(N, 3, device=task.device, generator=noise) - 0.5) * 0.01
                bodies = task._rigid_body_state.view(N, -1, 13)
                bodies[:, :task.num_bodies] = task_kinematics(task).states(
                    task._dof_pos, task._dof_vel, task._humanoid_root_states)
                bodies[:, task.num_bodies] = task._table_states
                bodies[:, task.num_bodies + 1] = task._target_states
                task._contact_forces[:] = 0
                task._tar_contact_forces[:] = 0
                task._reset_env_tensors(ids)
                task._compute_observations(ids)

            def ask(logs, history, previous, starts, step):
                nonlocal queries
                rows, refs, states = [], [], []
                for env in range(N):
                    phase = int(min(int(starts[env]) + step, 542))
                    obj = window(history['obj'], env)
                    hand = window(history['hand'], env)
                    goal = reference['obj'][min(phase + 24, 542)]
                    ref = reference['obj'][np.minimum(phase + np.array([8, 16, 24]), 542)]
                    previous_action = previous[env].detach().cpu().numpy()
                    current = {key: logs[key][-1][env] for key in ('q', 'dq')}
                    prop, robot_goal = control_context(current['q'], current['dq'], previous_action, obj[-1], ref)
                    state, _unused = features(
                        current['q'], current['dq'], hand[-1], obj[-1], logs['velocity'][-1][env],
                        np.broadcast_to(hand[-1], (24, 11, 3)))
                    rows.append(dict(env_id=env, tick=phase, object_history=obj.tolist(),
                                     hand_history=hand.tolist(), planned_object_goal=goal.tolist(),
                                     proprio=prop.tolist(), robot_goal=robot_goal.tolist(),
                                     retarget_state=np.asarray(state, np.float32).tolist()))
                    refs.append(ref)
                    states.append(np.asarray(state, np.float32))
                send(connection, dict(batch=rows))
                reply = receive(connection)
                queries += 1
                found = {item['env_id']: item for item in reply.get('responses', ())}
                if reply.get('schema') != SCHEMA or reply.get('batch') is not True or set(found) != set(range(N)):
                    raise ValueError('batched inference reply does not cover every env')
                taus, bases = [], []
                for env in range(N):
                    item = found[env]
                    tau = np.asarray(item['tau'], np.float32)
                    base = np.asarray(item['base_action'], np.float32)
                    anchor = np.asarray(item['anchor_object_pose'], np.float32)
                    if (item['tick'] != rows[env]['tick'] or tau.shape != (24, 33) or base.shape != (18,)
                            or not np.isfinite(tau).all() or not np.isfinite(base).all()
                            or not np.array_equal(anchor, np.asarray(rows[env]['object_history'][-1], np.float32))):
                        raise ValueError('inference reply does not match this live phase')
                    taus.append(tau)
                    bases.append(base)
                taus, bases = np.stack(taus), np.stack(bases)
                if queries == 1:
                    with torch.no_grad():
                        encoded = retargeter(torch.as_tensor(np.stack(states), device=task.device),
                                             torch.as_tensor(taus, device=task.device))
                    local = np.zeros_like(bases)
                    local[:, list(ACTIVE)] = encoded[:, 0].detach().cpu().numpy()
                    error = float(np.max(np.abs(local - bases)))
                    manifest['cross_runtime_base_max_error'] = error
                    write(out / 'manifest.json', manifest)
                    print(json.dumps(dict(cross_runtime_base_max_error=error)), flush=True)
                    if error > 1e-3:
                        raise ValueError('server chunk base disagrees with native R: ' + str(error))
                return taus, np.stack(refs), bases

            def make_observation(current, tau, refs, previous):
                tau_tensor = torch.as_tensor(tau, device=task.device)
                ref_tensor = torch.as_tensor(refs, device=task.device)
                return observation(current['q'], current['dq'], previous, current['obj'], current['velocity'],
                                   current['hand'], tau_tensor, ref_tensor)

            def command(current, tau, refs, previous, deterministic, base, zero_residual=False):
                base = torch.as_tensor(np.asarray(base, np.float32), device=task.device)
                obs = make_observation(current, tau, refs, previous)
                with torch.no_grad():
                    distribution = actor.distribution(obs)
                    if zero_residual:
                        delta = torch.zeros(N, 12, device=task.device)
                    else:
                        delta = distribution.loc if deterministic else distribution.sample()
                    logp = distribution.log_prob(delta).sum(-1)
                    mean, std = distribution.loc, distribution.scale
                applied = apply_residual(base, delta)
                return applied, obs, delta, logp, mean, std

            def episode(steps, starts, deterministic, collect, zero_residual=False):
                logs = {key: [] for key in ('q', 'dq', 'hand', 'obj', 'velocity', 'surface_gap',
                                            'support_gap', 'table_footprint', 'applied', 'delta', 'base')}
                rows = []

                def record():
                    current = measure()
                    for key in ('q', 'dq', 'hand', 'obj', 'velocity', 'surface_gap', 'support_gap', 'table_footprint'):
                        logs[key].append(current[key].detach().cpu().numpy().copy())
                    return current

                current = record()
                history, previous_np, z0_np = reset_context(
                    prefix, starts, {key: current[key].detach().cpu().numpy() for key in ('obj', 'hand')})
                previous = torch.as_tensor(previous_np, device=task.device)
                z0 = torch.as_tensor(z0_np, device=task.device)
                phase = torch.as_tensor([min(int(start), 542) for start in starts], device=task.device)
                ref = torch.as_tensor(reference['obj'][phase.cpu().numpy()], device=task.device)
                previous_distance = object_distance(current['obj'], ref)
                supported = current['table_footprint'] & (current['support_gap'].abs() <= 0.02)
                _, previous_held, _ = residual_reward(
                    previous_distance, previous_distance, current['surface_gap'], current['obj'][:, 2, 3] - z0,
                    supported, torch.zeros(N, dtype=torch.bool, device=task.device),
                    torch.zeros(N, 12, device=task.device))
                np.savez_compressed(out / f'reset-{queries:06d}.npz', starts=np.asarray(starts),
                                    object_history=np.stack(history['obj']), hand_history=np.stack(history['hand']),
                                    previous=previous_np, z0=z0_np, previous_held=previous_held.cpu().numpy())
                for step in range(steps):
                    before = time.monotonic()
                    tau, refs, base = ask(logs, history, previous, starts, step)
                    applied, obs, delta, logp, mean, std = command(
                        current, tau, refs, previous, deterministic, base, zero_residual=zero_residual)
                    if not torch.isfinite(applied).all():
                        raise FloatingPointError('nonfinite residual command')
                    expected = pd_from_action(applied, current['q'], offset, scale)
                    _obs, _reward, done, _info = self.env_step(self.env, applied.clone())
                    if capture is None or not torch.equal(capture, applied) or (actual_pd - expected).abs().max() > 1e-6:
                        raise ValueError('native applied command/actual PD capture mismatch')
                    done = done.bool()
                    if step < steps - 1 and done.any():
                        raise ValueError('early reset invalidates the residual episode')
                    logs['applied'].append(applied.detach().cpu().numpy().copy())
                    logs['delta'].append(delta.detach().cpu().numpy().copy())
                    logs['base'].append(base.copy())
                    nxt = record()
                    for key in history:
                        history[key].append(logs[key][-1])
                        history[key] = history[key][-4:]
                    post = np.minimum(np.asarray(starts) + step + 1, 542)
                    distance = object_distance(
                        nxt['obj'], torch.as_tensor(reference['obj'][post], device=task.device))
                    lift = nxt['obj'][:, 2, 3] - z0
                    supported = nxt['table_footprint'] & (nxt['support_gap'].abs() <= 0.02)
                    reward, held, terms = residual_reward(
                        distance, previous_distance, nxt['surface_gap'], lift, supported, previous_held, delta)
                    if collect:
                        with torch.no_grad():
                            value = value_net(obs)
                        rows.append(dict(observation=obs, delta=delta.detach(), reward=reward.detach(),
                                         value=value, done=done.detach(),
                                         logp=logp.detach(), mean=mean.detach(), std=std.detach(),
                                         terms={key: item.detach().cpu().numpy() for key, item in terms.items()}))
                    previous_distance, previous_held, previous = distance, held, applied.detach()
                    current = nxt
                    if step == 0 or (step + 1) % 64 == 0 or step + 1 == steps:
                        state = gpu_state(args.gpu)
                        if set(state['pids']) - {os.getpid(), server.pid}:
                            raise RuntimeError('foreign GPU process during residual PPO')
                        print(json.dumps(dict(step=step + 1, steps=steps, rpcs=queries, queries=queries * N,
                                              elapsed_s=time.monotonic() - started,
                                              step_s=time.monotonic() - before,
                                              eta_s=(time.monotonic() - started) / max(queries, 1) * (QUERIES - queries),
                                              gpu=state)), flush=True)
                bootstrap = None
                if collect:
                    # A new WAM query is required: V sees the real s_T plan,
                    # measured H4, previous command and planned goals at T.
                    tau, refs, _base = ask(logs, history, previous, starts, steps)
                    final_obs = make_observation(current, tau, refs, previous)
                    bootstrap = rollout_bootstrap(value_net, final_obs, done)
                    global_update = resume_updates + len(metrics) + 1
                    np.savez_compressed(out / f'bootstrap{global_update:04d}.npz',
                                        observation=final_obs.cpu().numpy(), bootstrap=bootstrap.cpu().numpy(),
                                        done=done.cpu().numpy(), phases=np.asarray(starts),
                                        q=current['q'].cpu().numpy(), dq=current['dq'].cpu().numpy(),
                                        previous=previous.cpu().numpy(), obj=current['obj'].cpu().numpy(),
                                        hand=current['hand'].cpu().numpy(), velocity=current['velocity'].cpu().numpy(),
                                        tau=tau, refs=refs,
                                        reward=np.stack([row['reward'].cpu().numpy() for row in rows]),
                                        value=np.stack([row['value'].cpu().numpy() for row in rows]))
                    torch.save(value_net.state_dict(), out / f'bootstrap-value{global_update:04d}.pt')
                return logs, rows, bootstrap

            diagnostics = []
            diagnostic_starts = diagnostic_phases(phases)
            for group in range((len(diagnostic_starts) + N - 1) // N):
                starts = [diagnostic_starts[(group * N + env) % len(diagnostic_starts)] for env in range(N)]
                place(starts, perturb=False)
                logs, _rows, _ = episode(DIAGNOSTIC, starts, True, False, zero_residual=True)
                packed = {key: np.stack(item) for key, item in logs.items()}
                np.savez_compressed(out / f'diagnostic{group}.npz', phases=np.asarray(starts), **packed)
                for env, phase in enumerate(starts):
                    diagnostics.append(dict(phase=phase, env=env,
                        min_gap_m=float(packed['surface_gap'][1:, env].min()),
                        held_fraction=float(np.mean((packed['surface_gap'][1:, env] <= 0.01)
                            & ~(packed['table_footprint'][1:, env] & (np.abs(packed['support_gap'][1:, env]) <= 0.02))
                            & (packed['obj'][1:, env, 2, 3] - prefix['obj'][0, env, 2, 3] >= 0.03)))))
            diagnostic_gap = min(item['min_gap_m'] for item in diagnostics if item['phase'] == 0)
            manifest['phase_diagnostics'] = diagnostics
            manifest['diagnostic_min_gap_m'] = diagnostic_gap
            write(out / 'manifest.json', manifest)
            print(json.dumps(dict(diagnostic_min_gap_m=diagnostic_gap)), flush=True)
            if any(item['min_gap_m'] > 0.03 for item in diagnostics):
                raise RuntimeError('zero-residual phase gate lost contact; PPO not started')
            def restore_rng(rng):
                torch.set_rng_state(rng['torch'].cpu())
                torch.cuda.set_rng_state_all([item.cpu() for item in rng['cuda']])
                noise.set_state(rng['phase'].cpu())
                np.random.set_state(rng['numpy'])
                random.setstate(rng['python'])

            def training_rng():
                return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                            phase=noise.get_state(), numpy=np.random.get_state(), python=random.getstate())

            if resumed is not None:
                restore_rng(resumed['rng'])

            def save_checkpoint(path, count):
                torch.save(dict(schema='ref2dex.wam-residual-ppo.v2', update=count,
                                actor=actor.state_dict(), value=value_net.state_dict(), alpha=ALPHA,
                                actor_optimizer=actor_optimizer.state_dict(), value_optimizer=value_optimizer.state_dict(),
                                phases=list(phases), prefix=prefix,
                                rng=training_rng()), path)

            def evaluate(global_update, final=False):
                saved_rng = training_rng()
                was_training = actor.training
                self.env_reset(torch.arange(N, device=task.device))
                if (task.start_times != 0).any():
                    raise ValueError('frame0 evaluation reset required')
                actor.eval()
                logs, _rows, _ = episode(EVAL, [0] * N, True, False)
                packed = {key: np.stack(logs[key]) for key in logs}
                np.savez_compressed(out / ('eval.npz' if final else f'eval{global_update:04d}.npz'), **packed)
                outcomes = []
                close_rise, thrown, residual_values = 0., False, []
                for env in range(N):
                    outcome = episode_outcome(dict(
                        object_pose=packed['obj'][:, env], object_velocity=packed['velocity'][:, env],
                        surface_gap=packed['surface_gap'][:, env], support_gap=packed['support_gap'][:, env],
                        table_footprint=packed['table_footprint'][:, env]))
                    height = packed['obj'][:, env, 2, 3] - packed['obj'][0, env, 2, 3]
                    close = packed['surface_gap'][:, env] <= 0.01
                    close_rise = max(close_rise, float(height[close].max()) if close.any() else 0.)
                    thrown = thrown or float(height.min()) < -0.05 or float(height[-1]) < -0.05
                    outcome.update(env=env, min_gap_m=float(packed['surface_gap'][:, env].min()),
                                   max_height_m=float(height.max()), terminal_height_m=float(height[-1]))
                    outcomes.append(outcome)
                residual_rms = float(np.sqrt(np.mean(np.square(packed['delta']))))
                conclusion = label_probe(outcomes, close_rise, thrown, residual_rms)
                actor.train(was_training)
                restore_rng(saved_rng)
                result = dict(status=conclusion, update=global_update, outcomes=outcomes,
                              close_rise_m=close_rise, thrown=thrown, eval_residual_rms=residual_rms)
                write(out / f'eval{global_update:04d}.json', result)
                print(json.dumps(dict(evaluation=result)), flush=True)
                return result

            for index in range(UPDATES):
                # A KL reject may quarter the step size inside one update.
                # The next update starts again at the declared rate.
                actor_optimizer.param_groups[0]['lr'] = ACTOR_LR
                # Each update is its own phase reset. Continuing the previous
                # physics state walks off the end of the 543-frame clip.
                place(phases)
                _logs, rows, bootstrap = episode(ROLLOUT, phases, False, True)
                batch = {key: torch.stack([row[key] for row in rows]) for key in
                         ('observation', 'delta', 'reward', 'value', 'done', 'logp', 'mean', 'std')}
                batch['phase'] = torch.as_tensor(phases, device=task.device)[None].expand(ROLLOUT, -1)
                result = update(actor, value_net, batch, bootstrap,
                                actor_optimizer, value_optimizer, gae_lambda=1.)
                with torch.no_grad():
                    actor.log_std.clamp_(-5., -1.2)
                global_update = resume_updates + index + 1
                result.update(update=global_update, mean_reward=float(batch['reward'].mean()),
                              residual_rms=float(batch['delta'].square().mean().sqrt()),
                              elapsed_s=time.monotonic() - started, rpcs=queries, queries=queries * N,
                              reward_terms={key: float(np.mean([row['terms'][key] for row in rows]))
                                            for key in rows[0]['terms']},
                              residual_mean_by_dof=batch['delta'].mean((0, 1)).tolist(),
                              residual_std_by_dof=batch['delta'].std((0, 1), unbiased=False).tolist(),
                              phase_returns={str(phase): float(batch['reward'][:, [env for env, p in enumerate(phases)
                                             if p == phase]].sum(0).mean()) for phase in set(phases)},
                              phase_held_fraction={str(phase): float(np.mean([row['terms']['held'][env]
                                             for row in rows for env, p in enumerate(phases) if p == phase]))
                                             for phase in set(phases)},
                              **residual_execution_stats(torch.as_tensor(np.stack(_logs['base']), device=task.device),
                                  batch['delta'], torch.as_tensor(np.stack(_logs['applied']), device=task.device)))
                result['grasp_trials'] = contact_hold_summary(
                    {key: np.stack(_logs[key]) for key in ('obj', 'surface_gap', 'support_gap', 'table_footprint')},
                    phases, prefix['obj'][0, :, 2, 3])
                metrics.append(result)
                print(json.dumps(result), flush=True)
                with (out / 'monitor.jsonl').open('a') as stream:
                    stream.write(json.dumps(result) + '\n')
                if global_update % 5 == 0:
                    save_checkpoint(out / f'update{global_update:04d}.pt', global_update)
                if args.eval_every and (index + 1) % args.eval_every == 0 and index + 1 < UPDATES:
                    checkpoint = out / f'update{global_update:04d}.pt'
                    if not checkpoint.exists():
                        save_checkpoint(checkpoint, global_update)
                    evaluate(global_update)
            save_checkpoint(out / 'final.pt', resume_updates + UPDATES)
            evaluation = evaluate(resume_updates + UPDATES, final=True)
            if queries != QUERIES:
                raise ValueError('server query count drifted: ' + str(queries))
            change = float(torch.sqrt(sum((item - actor_initial[key]).square().sum()
                                          for key, item in actor.state_dict().items())))
            report = dict(**evaluation, engineering_pass=True, diagnostic_min_gap_m=diagnostic_gap,
                          actor_parameter_change=change,
                          updates=resume_updates + UPDATES, rpcs=queries, queries=queries * N, metrics=metrics,
                          final_checkpoint_sha256=sha(out / 'final.pt'),
                          claim='one airplane frame-0 residual rollout; probe label only')
            write(out / 'result.json', report)
            manifest.update(status='COMPLETED', elapsed_s=time.monotonic() - started,
                            result_sha256=sha(out / 'result.json'))
            write(out / 'manifest.json', manifest)
            print(json.dumps(report), flush=True)

    native.EvalPlayer = Player
    train_cfg = native_root / 'data/cfg/train/rlg/inspire.yaml'
    sys.argv = [sys.argv[0], '--task', 'Dexplore_Inspire', '--cfg_env', cfg['cfg_env'],
                '--cfg_train', str(train_cfg), '--motion_file', cfg['motion_root'],
                '--checkpoint', str(OFFICIAL), '--headless', '--num_envs', str(N), '--seed', '292',
                '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--pipeline', 'gpu',
                '--graphics_device_id', '0', '--num_threads', '1', '--disable-early-termination',
                '--output', str(out / 'unused.json'), '--output_path', str(out / 'native')]
    cwd = Path.cwd()

    def deadline(signum, frame):
        raise TimeoutError('residual PPO exceeded the 14400s cap')

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(DEADLINE_S)
    try:
        os.chdir(ROOT / 'third_party/DExplore')
        native.main()
    except BaseException as error:
        manifest.update(status='FAILED', error=repr(error), elapsed_s=time.monotonic() - started,
                        queries=queries)
        write(out / 'manifest.json', manifest)
        raise
    finally:
        signal.alarm(0)
        os.chdir(cwd)
        connection.close()


if __name__ == '__main__':
    main()
