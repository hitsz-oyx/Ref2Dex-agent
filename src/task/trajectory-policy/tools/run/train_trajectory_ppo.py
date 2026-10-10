"""Pure-history trajectory PPO with measured task reward and a frozen executor."""
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
    for key in ('inputs', 'reference', 'checkpoint', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--actor-fit', type=Path, required=True, help='completed independent actor BC folder')
    parser.add_argument('--updates', type=int, default=24)
    parser.add_argument('--high-steps', type=int, default=16)
    parser.add_argument('--seed', type=int, default=293)
    parser.add_argument('--seconds', type=int, default=480)
    parser.add_argument('--engineering-smoke', action='store_true')
    args = parser.parse_args()
    if not (1 <= args.updates <= 24 and 2 <= args.high_steps <= 16):
        raise ValueError('bounded rollout/update count required')
    if args.engineering_smoke and not (args.seed < 100 and args.updates == 2 and args.high_steps == 2):
        raise ValueError('engineering smoke is exactly two updates/two chunks/debug seed')
    episode_controls = 14 if args.engineering_smoke else 542
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
    from trajectory_policy.dense_decoder import DenseDecoder
    from trajectory_policy.history import measured_history
    from trajectory_policy.actor import load_actor
    from trajectory_policy.ppo import HistoryValue, task_reward, update as ppo_update
    from trajectory_policy.actor import SCHEMA as ACTOR_SCHEMA, HISTORY_SCHEMA
    import pickle
    from oracle_y_utility import align_native_reference_tables
    import evaluate as native
    from env.tasks.base_dexplore_task import DexploreTask

    if torch.__version__ != '2.4.1+cu121':
        raise ValueError('pinned graspenv runtime required')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0')
    urdf = ROOT / 'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    with args.reference.open('rb') as stream:
        reference_packet = pickle.load(stream)
    initial_q = reference_packet['dof_position'][0, 0].copy()
    del reference_packet  # No future labels are extracted by online training.
    hashes = {str(path.resolve()): sha(path) for path in (args.reference, urdf)}
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
    fit_manifest = json.loads((args.actor_fit/'manifest.json').read_text())
    if fit_manifest['status'] != 'COMPLETED' or sha(args.actor_fit/'best.pt') != fit_manifest['checkpoint_sha256']:
        raise ValueError('completed frozen actor fit required')
    for path, digest in fit_manifest['input_sha256'].items():
        if sha(path) != digest:
            raise ValueError('actor fit source/input drift: '+path)
    hashes.update(fit_manifest['input_sha256'])
    for name in ('manifest.json', 'result.json', 'best.pt'):
        path = args.actor_fit/name
        hashes[str(path.resolve())] = sha(path)
    actor = load_actor(torch.load(args.actor_fit/'best.pt', map_location=device, weights_only=False), device).train()
    actor_weights = {key: value.clone() for key, value in actor.state_dict().items()}
    decoder = DenseDecoder(urdf, device)
    policy = TauTracker().to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if payload['schema'] != SCHEMA:
        raise ValueError('tau controller schema mismatch')
    policy.load_state_dict(payload['state_dict'], strict=True)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    weights = {key: value.clone() for key, value in policy.state_dict().items()}
    args.output.mkdir(parents=True)
    manifest = dict(schema='ref2dex.trajectory-ppo-run.v1', status='RUNNING',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        physical_gpu=args.gpu, gpu_before=before, seed=args.seed, input_sha256=hashes,
        updates=args.updates, high_steps=args.high_steps, envs=16, episode_controls=episode_controls,
        engineering_smoke=args.engineering_smoke, gamma=.99, gae_lambda=.95,
        actor_lr=1e-6, value_lr=3e-4, clip_ratio=.2, epochs=4, target_joint_kl=.02,
        actor_backtrack_factor=.25, actor_backtrack_trials=4,
        inference_contract='Measured H328 -> independent sampled c288 -> D -> tau -> frozen R; no future reference, phase, force or WM',
        reward_contract='Current physical proximity/lift/held/loss/native control cost; native reference reward ignored',
        claim='Bounded trajectory PPO Probe, not Validation or Cm utility')
    write(args.output/'manifest.json', manifest)
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
                raise ValueError('fixed native environment contract required')
            task._enable_early_termination = False
            task._adaptive_kappa_enabled = False
            task._state_init = DexploreTask.StateInit.Start
            task._hybrid_init_prob = 1.
            # Runner changes global flags before Player.run; fix actual PPO precision.
            torch.set_float32_matmul_precision('highest')
            manifest['matmul_precision'] = torch.get_float32_matmul_precision()
            self.model.eval()  # Owned loader model is never queried for actions.
            obs = self.env_reset(torch.arange(n, device=task.device))
            self.get_batch_size(obs['obs'], 1)
            if (task.start_times != 0).any() or not np.array_equal(initial_q, task._dof_pos[0].numpy()):
                raise ValueError('frame0 reset mismatch')
            names = task.gym.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles[0])
            key_ids = [names.index(name) for name in HAND_LINKS]
            offset, scale = task._pd_action_offset.to(device), task._pd_action_scale.to(device)
            props = task.gym.get_actor_dof_properties(task.envs[0], task.humanoid_handles[0])
            ratio = torch.as_tensor(props['damping'][:6].copy()/props['stiffness'][:6].copy(), device=device)
            dt = task.gym.get_sim_params(task.sim).dt*task.control_freq_inv
            if abs(dt-1/30) > 1e-8 or not torch.allclose(ratio, torch.full_like(ratio, .1)):
                raise ValueError('fixed controller contract required')
            manifest['native_controller'] = dict(offset=offset.cpu().tolist(), scale=scale.cpu().tolist(), gain_ratio=ratio.cpu().tolist())
            value_net = HistoryValue(actor.history_mean, actor.history_scale).to(device)
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-6)
            value_optimizer = torch.optim.Adam(value_net.parameters(), lr=3e-4)
            physical = PhysicalGeometry(task, ROOT/'third_party/DExplore/dexplore/data/assets', distance_device=device)
            support = TableSupport(ROOT/'third_party/DExplore/dexplore/data/assets', task.device)
            previous = torch.zeros(n, 12, device=device)
            last_command = torch.zeros(n, 18, device=device)
            previous_held = torch.zeros(n, dtype=torch.bool, device=device)
            held_run = torch.zeros(n, dtype=torch.long, device=device)
            maximum_held = held_run.clone()
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
            def measure():
                bodies = task._rigid_body_state.view(n, -1, 13)
                return dict(q=task._dof_pos.to(device), dq=task._dof_vel.to(device),
                    hand=bodies[:, key_ids, :3].to(device), obj=poses(task._target_states).to(device),
                    velocity=task._target_states[:, 7:13].to(device))
            def numpy_state(state):
                return {key: value.detach().cpu().numpy().copy() for key, value in state.items()}
            current = measure()
            history = [numpy_state(current)]*4
            initial_z = current['obj'][:, 2, 3].clone()
            def observation():
                return torch.as_tensor(measured_history(*(np.stack([state[key] for state in history])
                    for key in ('obj', 'hand', 'q', 'dq', 'velocity'))), device=device)
            def packet(update_index):
                return dict(schema=ACTOR_SCHEMA, history_schema=HISTORY_SCHEMA, width=512,
                    model={key: value.detach().cpu().clone() for key, value in actor.state_dict().items()},
                    value_model={key: value.detach().cpu().clone() for key, value in value_net.state_dict().items()},
                    update=update_index, input_sha256=hashes, git_commit=manifest['git_commit'])
            (args.output/'actors').mkdir()
            low_records, high_records, metrics, episodes = [], [], [], []
            episode_tick = episode_index = low_index = episode_start = 0
            work_started = time.monotonic()
            for update_index in range(args.updates):
                torch.save(packet(update_index), args.output/'actors'/('u%04d.pt'%update_index))
                records = {key: [] for key in ('history', 'c', 'mean', 'std', 'logp', 'value', 'reward', 'done', 'duration')}
                for chunk in range(args.high_steps):
                    with torch.no_grad():
                        h = observation()
                        distribution = actor.distribution(h)
                        c = distribution.sample()
                        v = value_net(h)
                        logp = distribution.log_prob(c).sum(-1)
                        decoded = decoder.decode(c.cpu().numpy(), current['q'].cpu().numpy(), current['obj'].cpu().numpy(), dt)
                        q, future, velocity = (decoded[key] for key in ('q', 'hand', 'velocity'))
                        duration = min(8, episode_controls-episode_tick)
                        total_reward = torch.zeros(n, device=device)
                        high = dict(update=update_index, chunk=chunk, low_index=low_index,
                            history=h.cpu().numpy().copy(), c=c.cpu().numpy().copy(),
                            mean=distribution.loc.cpu().numpy().copy(), std=distribution.scale.cpu().numpy().copy(),
                            logp=logp.cpu().numpy().copy(), value=v.cpu().numpy().copy(),
                            q=q.cpu().numpy().copy(), hand=future.cpu().numpy().copy(), velocity=velocity.cpu().numpy().copy())
                        for slot in range(duration):
                            shifted = (torch.arange(24, device=device)+slot).clamp_max(23)
                            next_q = q[:, slot+1]
                            x = tau_features(current['q'], current['dq'], current['hand'], current['obj'],
                                current['velocity'], future[:, shifted], next_q, previous)
                            latent = policy.actor(x)
                            base = wrist_feedforward(next_q, velocity[:, slot+1], ratio)
                            intended = native_action(policy.target(base, latent), current['q'], offset, scale)
                            command = intended.clamp(-1, 1)
                            if not all(torch.isfinite(value).all() for value in (h, c, x, latent, intended)):
                                raise FloatingPointError('nonfinite execution')
                            before_state = numpy_state(current)
                            obs, ignored_native_reward, raw_done, _ = self.env_step(self.env, command.clone())
                            if not torch.equal(command, capture):
                                raise ValueError('requested/applied mismatch')
                            after = measure()
                            _, gap = physical.measure(task)
                            sup, foot = support.measure(task, physical)
                            gap, sup, foot = (value.to(device) for value in (gap, sup, foot))
                            supported = foot & (abs(sup) <= .02)
                            lift = after['obj'][:, 2, 3]-initial_z
                            clipped = (intended-command).abs().amax(-1) > 1e-6
                            r, held, terms = task_reward(gap, lift, supported, previous_held, clipped, command-last_command)
                            total_reward += .99**slot*r
                            terminal = episode_tick+1 == episode_controls
                            if raw_done.any() and not terminal:
                                raise ValueError('unexpected early native termination')
                            low = dict(before_state, update=update_index, chunk=chunk, slot=slot,
                                episode_start=episode_start, episode_tick=episode_tick,
                                next_obj=after['obj'].cpu().numpy().copy(), next_velocity=after['velocity'].cpu().numpy().copy(),
                                initial_z=initial_z.cpu().numpy().copy(), surface_gap=gap.cpu().numpy().copy(),
                                support_gap=sup.cpu().numpy().copy(), table_footprint=foot.cpu().numpy().copy(),
                                reward=r.cpu().numpy().copy(), held=held.cpu().numpy().copy(),
                                features=x.cpu().numpy().copy(), latent=latent.cpu().numpy().copy(),
                                intended=intended.cpu().numpy().copy(), action=command.cpu().numpy().copy(),
                                applied=capture.cpu().numpy().copy(), pd_targets=pd_target.cpu().numpy().copy(),
                                clipped=clipped.cpu().numpy().copy(), **{'reward_'+key:value.cpu().numpy().copy() for key,value in terms.items()})
                            low_records.append(low)
                            held_run = torch.where(held, held_run+1, torch.zeros_like(held_run))
                            maximum_held = torch.maximum(maximum_held, held_run)
                            previous, previous_held = torch.tanh(latent), held
                            last_command = command
                            current = after
                            history = (history+[numpy_state(after)])[-4:]
                            low_index += 1
                            episode_tick += 1
                        done = torch.full((n,), terminal, dtype=torch.bool, device=device)
                        high.update(reward=total_reward.cpu().numpy().copy(), done=done.cpu().numpy().copy(), duration=np.full(n, duration, np.int64))
                        high_records.append(high)
                        for key in records:
                            val = {'history': h, 'c': c, 'mean': distribution.loc, 'std': distribution.scale,
                                'logp': logp, 'value': v, 'reward': total_reward, 'done': done,
                                'duration': torch.full((n,), duration, dtype=torch.long, device=device)}[key]
                            records[key].append(val.detach())
                        if terminal:
                            episodes.append(dict(episode=episode_index, maximum_held=maximum_held.cpu().tolist(),
                                terminal_held=held.cpu().tolist(), low_end=low_index))
                            obs = self.env_reset(torch.arange(n, device=task.device))
                            current = measure()
                            if (task.start_times != 0).any():
                                raise ValueError('reset start frame drift')
                            history = [numpy_state(current)]*4
                            initial_z = current['obj'][:, 2, 3].clone()
                            previous.zero_()
                            last_command = torch.zeros_like(last_command)
                            previous_held = torch.zeros_like(previous_held)
                            held_run.zero_()
                            maximum_held.zero_()
                            episode_tick = 0
                            episode_index += 1
                            episode_start = low_index
                batch = {key: torch.stack(values) for key, values in records.items()}
                with torch.no_grad():
                    bootstrap_history = observation()
                    bootstrap = value_net(bootstrap_history).detach()
                with torch.enable_grad():
                    result = ppo_update(actor, value_net, batch, bootstrap, actor_optimizer, value_optimizer)
                for chunk in range(args.high_steps):
                    high_records[-args.high_steps+chunk].update(
                        advantage=result['advantage'][chunk].cpu().numpy().copy(), returns=result['returns'][chunk].cpu().numpy().copy(),
                        bootstrap=bootstrap.cpu().numpy().copy(), bootstrap_history=bootstrap_history.cpu().numpy().copy())
                with torch.no_grad():
                    actor.log_std.clamp_(-4.605170186, -1.609437912)
                state = gpu_state(args.gpu)
                for row in state['processes'].splitlines():
                    pid, memory = map(int, row.split(','))
                    if pid != os.getpid() and memory > 512:
                        raise RuntimeError('foreign GPU compute detected')
                monitor = dict(update=update_index+1, low_controls=low_index, environment_interactions=low_index*n,
                    mean_discounted_chunk_reward=float(batch['reward'].mean()),
                    actor_steps=result['actor_steps'], policy_loss=result['policy_loss'], value_loss=result['value_loss'],
                    joint_action_kl=result['joint_action_kl'], actor_lr=result['actor_lr'], behavior_logprob_error=result['behavior_logprob_error'],
                    elapsed_s=time.monotonic()-started, eta_s=(time.monotonic()-work_started)/(update_index+1)*(args.updates-update_index-1), gpu=state)
                metrics.append(monitor)
                print(json.dumps(monitor), flush=True)
                with (args.output/'monitor.jsonl').open('a') as stream:
                    stream.write(json.dumps(monitor)+'\n')
            if any(sha(path) != digest for path, digest in hashes.items()) or any(
                not torch.equal(value, weights[key]) for key, value in policy.state_dict().items()):
                raise ValueError('source/input/frozen R drift')
            for key in ('history_mean', 'history_scale', 'action_mean', 'action_scale'):
                if not torch.equal(actor.state_dict()[key], actor_weights[key]):
                    raise ValueError('normalization changed during PPO')
            torch.save(packet(args.updates), args.output/'final.pt')
            np.savez_compressed(args.output/'low.npz', **{key:np.stack([record[key] for record in low_records]) for key in low_records[0]})
            np.savez_compressed(args.output/'high.npz', **{key:np.stack([record[key] for record in high_records]) for key in high_records[0]})
            parameter_change = float(torch.sqrt(sum((value-actor_weights[key]).square().sum() for key,value in actor.state_dict().items() if key.startswith('mean_net.'))))
            report = dict(status='UNCLEAR', engineering_pass=True, actor_parameter_change=parameter_change,
                updates=args.updates, controls=low_index, environment_interactions=low_index*n,
                completed_episodes=episodes, held_steps=int(sum(record['held'].sum() for record in low_records)),
                clipping_rate=float(np.mean([record['clipped'] for record in low_records])),
                final_checkpoint_sha256=sha(args.output/'final.pt'),
                claim='Actual PPO updates and physical traces; independent frozen evaluation/audit required')
            write(args.output/'result.json', report)
            manifest.update(status='COMPLETED', elapsed_s=time.monotonic()-started,
                torch_peak_bytes=torch.cuda.max_memory_allocated(), final_checkpoint_sha256=report['final_checkpoint_sha256'])
            write(args.output/'manifest.json', manifest)

    native.EvalPlayer = Player
    sys.argv = [sys.argv[0], '--task', 'Dexplore_Inspire', '--cfg_env', cfg['cfg_env'],
        '--cfg_train', cfg['cfg_train'], '--checkpoint', cfg['actor'], '--motion_file', cfg['motions'],
        '--headless', '--num_envs', '16', '--seed', str(args.seed), '--sim_device', 'cuda:0', '--rl_device', 'cuda:0',
        '--pipeline', 'cpu', '--graphics_device_id', '0', '--num_threads', '1', '--disable-early-termination',
        '--output', str(args.output/'native-unused.json'), '--output_path', str(args.output/'native')]
    cwd = Path.cwd()
    def deadline(signum, frame):
        raise TimeoutError('trajectory PPO exceeded declared wall-time cap')
    handler = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.seconds)
    try:
        os.chdir(ROOT/'third_party/DExplore')
        native.main()
    except BaseException as error:
        manifest.update(status='FAILED', elapsed_s=time.monotonic()-started, error=repr(error))
        write(args.output/'manifest.json', manifest)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, handler)
        os.chdir(cwd)


if __name__ == '__main__':
    main()
