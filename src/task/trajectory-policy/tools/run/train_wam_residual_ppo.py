"""Phase-randomized residual PPO on the frozen context WAM and old chunk R.

The base command is the frozen pair. PPO adds a 12-D residual and is scored
on the airplane outcome, not on an expert action.
"""
import argparse
import json
import os
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
RUN = 'wam-ref2-execution-rppo-20261013-r4'
HEAD = ROOT / 'outputs/trajectory-policy/wam-ref2-joint-multitakeover-20261012-r1'
CHUNK = ROOT / 'outputs/trajectory-policy/wam-v2-control-fit-20261012-r1'
EFFECT = ROOT / 'outputs/goal-point-wam/wam-effect-ddp-20261013-r3/train/checkpoint_step50500.pt'
CONTROL = ROOT / 'outputs/goal-point-wam/wam-control-context-20261013-r3/stage2_step4500.pt'
N = 4
DIAGNOSTIC = 64
UPDATES = 200
ROLLOUT = 96
DEADLINE_S = 14400
ACTOR_LR = 3e-4
EVAL = 542
PHASES = (0, 32, 48, 62)
QUERIES = N * (DIAGNOSTIC + UPDATES * ROLLOUT + EVAL)
sys.path[:0] = [str(TASK / 'src'), str(TASK / 'tools/audit'), str(TASK / 'tools/run')]
from audit_wam_ref2_shadow import sha, write
from trajectory_policy.wam_ipc import SCHEMA, send, receive, peer_pid
from collect_competent_pd import OFFICIAL, OFFICIAL_SHA, gpu_state


def window(rows, env):
    frames = [row[env] for row in rows]
    while len(frames) < 4:
        frames.insert(0, frames[0])
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
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/trajectory-policy' / RUN)
    args = parser.parse_args()
    if QUERIES != 79224:
        raise ValueError('query budget drifted from the long-run card')
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
        [GRASPENV, str(server_script), '--gpu', str(args.gpu), '--controls', str(QUERIES),
         '--experiment-id', EXP, '--head-dir', str(HEAD), '--socket', str(address),
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
        residual_reward, update,
    )
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
    actor_initial = {key: item.detach().clone() for key, item in actor.state_dict().items()}
    torch.manual_seed(292)
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
            or handshake.get('experiment_id') != EXP or handshake['base_sha256'] != base_hash
            or handshake.get('scene_patch_identity') != patch_identity
            or handshake.get('retargeter_identity') != r_identity
            or handshake.get('effect_identity') != effect_identity
            or handshake.get('control_identity') != control_identity):
        raise ValueError('WAM inference peer identity differs')
    started = time.monotonic()
    queries = 0
    manifest = dict(schema='ref2dex.wam-residual-ppo.v1', status='RUNNING', seed=292,
                    ppo_seed=296, physical_gpu=args.gpu, experiment_id=EXP, run_id=out.name,
                    alpha=ALPHA, phases=list(PHASES), queries=QUERIES, updates=UPDATES,
                    rollout=ROLLOUT, diagnostic_steps=DIAGNOSTIC, eval_steps=EVAL,
                    frozen_world=True, frozen_wam=True, frozen_R=True,
                    reward='object progress, contact, hold, slip, residual penalty',
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    server_pid=server.pid, native_pid=os.getpid(),
                    retargeter_identity=r_identity, effect_identity=effect_identity,
                    control_identity=control_identity)
    write(out / 'manifest.json', manifest)
    install_legacy_player_compat(native, common_player, base_player)
    install_reset_patch()
    metrics = []

    class Player(native.EvalPlayer):
        def run(self):
            nonlocal queries
            task = self.env.task
            if task.num_envs != N or self.is_rnn or str(task.device) != 'cuda:0':
                raise ValueError('four-env nonrecurrent GPU pipeline required')
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

            def place(frames):
                frames = torch.as_tensor(frames, dtype=torch.long, device=task.device)
                ids = torch.arange(N, device=task.device)
                task.ref_index[:] = 0
                task.progress_buf[:] = frames
                task.start_times[:] = frames
                task._hist_obs[:] = 0
                task.contact_reset[:] = 0
                dof = task.hoi_refs[task.data_id, 0, frames, 119:119 + task.num_dof]
                vel = task.hoi_refs[task.data_id, 0, frames, 119 + task.num_dof:119 + 2 * task.num_dof]
                task._set_env_state(ids, dof, vel)
                task._reset_target(ids)
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

            def ask(logs, starts, step):
                nonlocal queries
                taus, refs = [], []
                for env in range(N):
                    phase = int(min(int(starts[env]) + step, 542))
                    obj = window(logs['obj'], env)
                    hand = window(logs['hand'], env)
                    goal = reference['obj'][min(phase + 24, 542)]
                    ref = reference['obj'][np.minimum(phase + np.array([8, 16, 24]), 542)]
                    previous = logs['applied'][-1][env] if logs['applied'] else np.zeros(18, np.float32)
                    current = {key: logs[key][-1][env] for key in ('q', 'dq')}
                    prop, robot_goal = control_context(current['q'], current['dq'], previous, obj[-1], ref)
                    request = dict(tick=phase, object_history=obj.tolist(), hand_history=hand.tolist(),
                                   planned_object_goal=goal.tolist(), proprio=prop.tolist(),
                                   robot_goal=robot_goal.tolist())
                    send(connection, request)
                    reply = receive(connection)
                    queries += 1
                    tau = np.asarray(reply['tau'], np.float32)
                    if (reply['schema'] != SCHEMA or reply['tick'] != phase or tau.shape != (24, 33)
                            or not np.isfinite(tau).all()
                            or not np.array_equal(np.asarray(reply['anchor_object_pose'], np.float32), obj[-1])):
                        raise ValueError('inference reply does not match this live phase')
                    taus.append(tau)
                    refs.append(ref)
                return np.stack(taus), np.stack(refs)

            def command(current, tau, refs, previous, deterministic):
                states = []
                for env in range(N):
                    values = [current[key][env].detach().cpu().numpy() for key in ('q', 'dq', 'hand', 'obj', 'velocity')]
                    state, _tau = features(*values, np.broadcast_to(values[2], (24, 11, 3)))
                    states.append(state)
                state_tensor = torch.as_tensor(np.stack(states), device=task.device)
                tau_tensor = torch.as_tensor(tau, device=task.device)
                ref_tensor = torch.as_tensor(refs, device=task.device)
                with torch.no_grad():
                    encoded = retargeter(state_tensor, tau_tensor)
                base = torch.zeros(N, 18, device=task.device)
                base[:, list(ACTIVE)] = encoded[:, 0]
                obs = observation(current['q'], current['dq'], previous, current['obj'], current['velocity'],
                                  current['hand'], tau_tensor, ref_tensor)
                with torch.no_grad():
                    distribution = actor.distribution(obs)
                    delta = distribution.loc if deterministic else distribution.sample()
                    logp = distribution.log_prob(delta).sum(-1)
                    mean, std = distribution.loc, distribution.scale
                applied = apply_residual(base, delta)
                return applied, obs, delta, logp, mean, std

            def episode(steps, starts, deterministic, collect):
                logs = {key: [] for key in ('q', 'dq', 'hand', 'obj', 'velocity', 'surface_gap',
                                            'support_gap', 'table_footprint', 'applied', 'delta')}
                rows = []

                def record():
                    current = measure()
                    for key in ('q', 'dq', 'hand', 'obj', 'velocity', 'surface_gap', 'support_gap', 'table_footprint'):
                        logs[key].append(current[key].detach().cpu().numpy().copy())
                    return current

                current = record()
                z0 = current['obj'][:, 2, 3].clone()
                phase = torch.as_tensor([min(int(start), 542) for start in starts], device=task.device)
                ref = torch.as_tensor(reference['obj'][phase.cpu().numpy()], device=task.device)
                previous_distance = object_distance(current['obj'], ref)
                previous_held = torch.zeros(N, dtype=torch.bool, device=task.device)
                previous = torch.zeros(N, 18, device=task.device)
                for step in range(steps):
                    before = time.monotonic()
                    tau, refs = ask(logs, starts, step)
                    applied, obs, delta, logp, mean, std = command(
                        current, tau, refs, previous, deterministic)
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
                    nxt = record()
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
                                         terms={key: float(item.mean()) for key, item in terms.items()}))
                    previous_distance, previous_held, previous = distance, held, applied.detach()
                    current = nxt
                    if step == 0 or (step + 1) % 64 == 0 or step + 1 == steps:
                        state = gpu_state(args.gpu)
                        if set(state['pids']) - {os.getpid(), server.pid}:
                            raise RuntimeError('foreign GPU process during residual PPO')
                        print(json.dumps(dict(step=step + 1, steps=steps, queries=queries,
                                              elapsed_s=time.monotonic() - started,
                                              step_s=time.monotonic() - before, gpu=state)), flush=True)
                return logs, rows

            logs, _rows = episode(DIAGNOSTIC, [0] * N, True, False)
            gaps = np.stack(logs['surface_gap'][1:])
            diagnostic_gap = float(gaps.min())
            manifest['diagnostic_min_gap_m'] = diagnostic_gap
            write(out / 'manifest.json', manifest)
            print(json.dumps(dict(diagnostic_min_gap_m=diagnostic_gap)), flush=True)
            if diagnostic_gap > 0.03:
                raise RuntimeError('zero-residual base lost the context contact; PPO not started')
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
            value_optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)
            for index in range(UPDATES):
                # A KL reject may quarter the step size inside one update.
                # The next update starts again at the declared rate.
                actor_optimizer.param_groups[0]['lr'] = ACTOR_LR
                # Each update is its own phase reset. Continuing the previous
                # physics state walks off the end of the 543-frame clip.
                place(PHASES)
                _logs, rows = episode(ROLLOUT, PHASES, False, True)
                batch = {key: torch.stack([row[key] for row in rows]) for key in
                         ('observation', 'delta', 'reward', 'value', 'done', 'logp', 'mean', 'std')}
                result = update(actor, value_net, batch, torch.zeros(N, device=task.device),
                                actor_optimizer, value_optimizer)
                with torch.no_grad():
                    actor.log_std.clamp_(-5., -1.2)
                result.update(update=index + 1, mean_reward=float(batch['reward'].mean()),
                              residual_rms=float(batch['delta'].square().mean().sqrt()),
                              elapsed_s=time.monotonic() - started, queries=queries,
                              reward_terms={key: float(np.mean([row['terms'][key] for row in rows]))
                                            for key in rows[0]['terms']})
                metrics.append(result)
                print(json.dumps(result), flush=True)
                with (out / 'monitor.jsonl').open('a') as stream:
                    stream.write(json.dumps(result) + '\n')
                if (index + 1) % 25 == 0:
                    torch.save(dict(schema='ref2dex.wam-residual-ppo.v1', update=index + 1,
                                    actor=actor.state_dict(), value=value_net.state_dict(), alpha=ALPHA),
                               out / f'update{index + 1:04d}.pt')
            self.env_reset(torch.arange(N, device=task.device))
            if (task.start_times != 0).any():
                raise ValueError('frame0 evaluation reset required')
            actor.eval()
            logs, _rows = episode(EVAL, [0] * N, True, False)
            packed = {key: np.stack(logs[key]) for key in logs}
            np.savez_compressed(out / 'eval.npz', **packed)
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
            if queries != QUERIES:
                raise ValueError('server query count drifted: ' + str(queries))
            change = float(torch.sqrt(sum((item - actor_initial[key]).square().sum()
                                          for key, item in actor.state_dict().items())))
            torch.save(dict(schema='ref2dex.wam-residual-ppo.v1', actor=actor.state_dict(),
                            value=value_net.state_dict(), alpha=ALPHA), out / 'final.pt')
            report = dict(status=conclusion, engineering_pass=True, diagnostic_min_gap_m=diagnostic_gap,
                          outcomes=outcomes, close_rise_m=close_rise, thrown=thrown,
                          eval_residual_rms=residual_rms, actor_parameter_change=change,
                          updates=UPDATES, queries=queries, metrics=metrics,
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
