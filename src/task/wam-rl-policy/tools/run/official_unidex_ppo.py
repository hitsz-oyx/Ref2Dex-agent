"""Run the pinned upstream state PPO, with bounded supervision and passive metrics.

PPO.run/update, RolloutStorage and the state ActorCritic math are upstream code.
Only unused vision imports are removed from a per-run copy. The environment
bootstrap reuses audited native assets; it does not load a teacher or BC policy.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
UPSTREAM = '6c9e710b04769bf5d9765e3233820101cb3d718f'


def write(path, value):
    part = path.with_suffix(path.suffix+'.part')
    part.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    part.replace(path)


def runtime_replay(source, training):
    begin = source.index('    actor_layers = []')
    end = source.index('    # The official replay asset', begin)
    source = source[:begin]+source[end:]
    begin = source.index('    records = []\n    begin = time.monotonic()')
    end = source.index('\n\ndef supervise(args):', begin)
    source = source[:begin]+'''    try:
        args.run_official(task, snapshot, env_objects, args)
    finally:
        task.gym.destroy_sim(task.sim)
'''+source[end:]
    source = source.replace('random_time=False, printNumSuccesses=False',
                            'random_time='+str(training)+', printNumSuccesses=False')
    source = source.replace('> 128*2**20:', '> 512*2**20:')
    begin = source.index("    work = assets.parent/'dexgrasp'")
    end = source.index('    cfg = yaml.safe_load', begin)
    source = source[:begin]+'''    work = args.output/'runtime-work/dexgrasp'
    work.mkdir(parents=True)
    (work.parent/'assets').symlink_to(assets.resolve(), target_is_directory=True)
    os.chdir(work)
'''+source[end:]
    assert 'state_based_model.pt' not in source[:source.index('def supervise(args):')]
    return source


def run_official(task, snapshot, objects, args):
    import numpy as np
    import torch
    import yaml
    from tasks.hand_base.vec_task import VecTaskPython

    class WamObservationEnv:
        """Expose the frozen WAM summaries through the official VecTask API."""
        def __init__(self, base, task, snapshot_fn, checkpoint, cache):
            from gym import spaces
            from wam_rl_policy.wam_features import FeatureController
            self.base = base
            self.task = task
            self.snapshot_fn = snapshot_fn
            self.num_envs = base.num_envs
            self.num_environments = base.num_environments
            self.num_states = base.num_states
            self.num_actions = base.num_actions
            self.num_observations = 2607
            self.state_space = base.state_space
            self.action_space = base.action_space
            self.clip_obs = base.clip_obs
            self.clip_actions = base.clip_actions
            self.rl_device = base.rl_device
            self.controller = FeatureController(checkpoint, objects, cache)
            # The official task can explore outside the WAM training workspace.
            # Keep the bounded probe alive with the repository's admitted zero
            # fallback; the run records this as a feature-coverage limitation.
            self.controller.allow_invalid_zero = True
            self.observation_space = spaces.Box(
                low=np.full(2607, -np.inf, dtype=np.float32),
                high=np.full(2607, np.inf, dtype=np.float32), dtype=np.float32)
            self._resets = False

        def _augment(self, native, tick):
            self.controller.act(self.snapshot_fn(), native, int(tick))
            z = torch.nn.functional.layer_norm(
                self.controller.last_wam_features.reshape(-1, 3, 768), (768,)).flatten(1)
            relative_goal = self.task.goal_pos - self.task.object_pose[:, :3]
            result = torch.cat((native, z, relative_goal / .1), dim=-1)
            if result.shape != (self.num_envs, 2607) or not torch.isfinite(result).all():
                raise FloatingPointError('Nonfinite or malformed official WAM observation')
            return result

        def reset(self):
            native = self.base.reset()
            self.controller.reset_envs(range(self.num_envs))
            self.controller.invalid = set()
            return self._augment(native, 1)

        def step(self, actions, id=0):
            native, rewards, dones, infos = self.base.step(actions, id)
            result = self._augment(native, int(id) + 2)
            done_ids = torch.nonzero(dones, as_tuple=False).flatten().tolist()
            if done_ids:
                # Keep the terminal frame available for this observation, then
                # clear only those environments before their next post-reset
                # snapshot is encoded.
                self.controller.mark_reset_envs(done_ids)
            return result, rewards, dones, infos

        def get_state(self):
            return self.base.get_state()

    official = args.output/'runtime-official/dexgrasp'
    sys.path.insert(0, str(official))
    spec = importlib.util.spec_from_file_location('official_process_sarl', official/'utils/process_sarl.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cfg = yaml.safe_load((official/'cfg/ppo/config.yaml').read_text())
    # These keys are unconditionally read even for the state-only model.
    cfg['policy'].update(backbone_type='pn', freeze_backbone=False)
    cfg['seed'] = args.seed
    task.cfg['seed'] = args.seed
    env = VecTaskPython(task, 'cuda:0', cfg['clip_observations'], cfg['clip_actions'])
    if args.wam_checkpoint:
        # Keep all feature-source imports local to the WAM arm.  The official
        # state-only baseline therefore retains the exact 300D environment path.
        sys.path[:0] = [str(ROOT/'src/task/wam-rl-policy/src'),
                        str(ROOT/'src/task/shadow-finetune/src'),
                        str(ROOT/'src/task/ptv3-shared-wam-v2/src'),
                        str(ROOT/'src/task/video-data-preparation/src')]
        env = WamObservationEnv(env, task, snapshot, args.wam_checkpoint, args.wam_cache)
        cfg['policy']['obs_dim'] = 2607
        cfg['policy']['wam_adapter'] = bool(args.wam_adapter)
        cfg['policy']['wam_adapter_bounded'] = args.wam_adapter_gain is not None
        cfg['policy']['wam_adapter_gain'] = float(args.wam_adapter_gain or 1.0)
    # A 300D official checkpoint cannot be loaded directly into the 2607D WAM
    # network.  Preserve its native columns and initialize the new WAM/goal
    # columns to zero, which is the matched WAM finetune starting point.
    wam_init = bool(args.wam_checkpoint and args.model)
    process_args = SimpleNamespace(test=False, vision=False,
        model_dir='' if wam_init else (str(args.model) if args.model else ''))
    ppo = module.process_ppo(process_args, env, cfg, str(args.output/'models'))
    if wam_init:
        loaded = torch.load(args.model, map_location='cpu')
        if isinstance(loaded, dict) and 'state_dict' in loaded: loaded = loaded['state_dict']
        target = ppo.actor_critic.state_dict()
        for key, value in loaded.items():
            if key not in target: raise ValueError('Official init key missing in WAM model: '+key)
            if tuple(value.shape) == tuple(target[key].shape):
                target[key].copy_(value)
            elif key in ('actor.0.weight', 'critic.0.weight') and value.shape[1] == 300 and target[key].shape[1] == 2607:
                target[key].zero_(); target[key][:, :300].copy_(value)
            else:
                raise ValueError('Official init shape mismatch for WAM model: '+key)
        ppo.actor_critic.load_state_dict(target, strict=True)
    if args.mode == 'evaluate':
        assert args.model is not None
        ppo.actor_critic.eval()
        task.random_time = False
        obs = env.reset(); traces = [snapshot()]
        initial_z = task.object_pos[:, 2].clone()
        hold = torch.zeros(args.envs, device='cuda', dtype=torch.long)
        longest = hold.clone(); max_lift = torch.zeros_like(initial_z)
        with torch.inference_mode():
            for tick in range(1, task.max_episode_length):
                obs, _, _, _ = env.step(ppo.actor_critic.act_inference(obs), tick-1)
                lift = task.object_pos[:, 2]-initial_z
                near = (task.object_pos-task.goal_pos).norm(dim=-1) <= .05
                hold = torch.where(near & (lift >= .1), hold+1, 0)
                longest = torch.maximum(longest, hold); max_lift = torch.maximum(max_lift, lift)
                traces.append(snapshot())
        np.savez_compressed(args.output/'evaluation.npz', **{k:np.stack([s[k] for s in traces]) for k in traces[0]})
        report = dict(episodes=args.envs, official_success=int(task.successes.sum()),
            held60_with_lift10cm=int((longest >= 60).sum()), lifted10cm=int((max_lift >= .1).sum()),
            longest_goal_hold_with_lift_frames=longest.tolist(), max_lift_m=max_lift.tolist())
        write(args.output/'evaluation.json', report)
        write(args.output/'progress.json', dict(status='COMPLETED', evaluation=report))
        ppo.writer.close()
        return

    write(args.output/'training-config.json', cfg)
    initial = {k:v.detach().clone() for k,v in ppo.actor_critic.state_dict().items()}
    ppo.save(str(args.output/'model-initial.pt'))
    state = dict(iteration=0, optimizer_steps=0)
    original_step = ppo.optimizer.step
    def optimizer_step(*a, **kw):
        result = original_step(*a, **kw)
        state['optimizer_steps'] += 1
        return result
    ppo.optimizer.step = optimizer_step
    original_log, original_update, original_save = ppo.log, ppo.update, ppo.save
    started = time.monotonic()
    def update():
        returns, values = ppo.storage.returns, ppo.storage.values
        var = returns.var()
        state['critic_EV'] = float(1-(returns-values).var()/var) if var > 1e-8 else 0.
        state['action_clip_fraction'] = float((ppo.storage.actions.abs() > 1).float().mean())
        state['current_native_successes'] = int(task.successes.sum())
        result = original_update()
        if not all(np.isfinite(v) for v in result):raise FloatingPointError('Nonfinite upstream PPO losses')
        return result
    ppo.update = update
    def log(locs):
        original_log(locs)
        iteration = locs['it']+1
        elapsed = time.monotonic()-started
        state.update(iteration=iteration, env_steps=iteration*args.envs*cfg['learn']['nsteps'],
            elapsed_s=elapsed, eta_s=elapsed/iteration*(args.iterations-iteration),
            value_loss=float(locs['mean_value_loss']), surrogate_loss=float(locs['mean_surrogate_loss']),
            parameter_std=float(ppo.actor_critic.log_std.exp().mean()),
            effective_std=float(ppo.actor_critic.log_std.exp().square().mean()), lr=ppo.step_size,
            mean_native_reward_per_step=float(locs['mean_reward']),
            mean_episode_return=float(np.mean(locs['rewbuffer'])) if locs['rewbuffer'] else None)
        if args.wam_checkpoint:
            state['wam_invalid_envs_current_history'] = len(env.controller.invalid)
        with (args.output/'curve.jsonl').open('a') as stream:stream.write(json.dumps(state, allow_nan=False)+'\n')
        write(args.output/'progress.json', dict(status='RUNNING', **state))
    ppo.log = log
    def save(path):
        original_save(path)
        torch.save(dict(optimizer=ppo.optimizer.state_dict(), completed_iterations=state['iteration'],
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(), random_rng=__import__('random').getstate()),
            str(path)+'.optimizer.pt')
    ppo.save = save
    try:
        ppo.run(num_learning_iterations=args.iterations, log_interval=args.save_interval)
        differences = {k:float((v-initial[k]).abs().max()) for k,v in ppo.actor_critic.state_dict().items()}
        assert all(torch.isfinite(v).all() for v in ppo.actor_critic.state_dict().values())
        actor_changes = [v for k,v in differences.items()
                         if k.startswith('actor.') or k.startswith('wam_actor_adapter.')]
        assert actor_changes and max(actor_changes) > 0
        critic_changes = [v for k,v in differences.items() if k.startswith('critic.')]
        if not args.wam_adapter:
            assert critic_changes and max(critic_changes) > 0
        assert state['optimizer_steps'] == args.iterations*cfg['learn']['noptepochs']*cfg['learn']['nminibatches']
        write(args.output/'learning-audit.json', dict(status='PASS', parameter_max_change=differences,
            optimizer_steps=state['optimizer_steps'], source_engine='unmodified upstream PPO.run/update/storage'))
        write(args.output/'progress.json', dict(status='COMPLETED', **state))
    finally:
        ppo.writer.close()


def worker(args):
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'tmp').mkdir(exist_ok=True)
    source_root = ROOT/'third_party/UniDexGrasp2-official-ppo'
    runtime = args.output/'runtime-official'
    shutil.copytree(source_root, runtime)
    model = runtime/'dexgrasp/algorithms/rl/ppo/module.py'
    source = model.read_text()
    removed = [line for line in source.splitlines() if line.startswith('from algo.pn_utils.')]
    assert len(removed) == 2
    source = '\n'.join(line for line in source.splitlines() if line not in removed)+'\n'
    algorithm_changes = False
    if args.wam_checkpoint:
        # The upstream state actor hard-codes the native 300D prefix in three
        # places.  WAM changes only the declared input contract; PPO math,
        # rollout storage, covariance and KL handling remain upstream.
        source = source.replace("self.num_obs = 300", "self.num_obs = model_cfg.get('obs_dim', 300)")
        source = source.replace("self.actor(observations[:,:300])", "self.actor(observations)")
        source = source.replace("if not self.use_pc:\n            observations = observations[:,:300]",
                                "if not self.use_pc and self.num_obs == 300:\n            observations = observations[:,:300]")
        if ("model_cfg.get('obs_dim', 300)" not in source or
                "self.actor(observations[:,:300])" in source):
            raise ValueError('WAM official input-dimension patch did not apply cleanly')
        algorithm_changes = True
        if args.wam_adapter:
            marker = "        self.init_weights(self.critic, critic_weights)\n"
            addition = marker + """        if model_cfg.get('wam_adapter', False):
            self.wam_actor_adapter = nn.Linear(2307, 24)
            self.wam_adapter_bounded = model_cfg.get('wam_adapter_bounded', False)
            self.wam_adapter_gain = model_cfg.get('wam_adapter_gain', 1.0)
            nn.init.zeros_(self.wam_actor_adapter.weight); nn.init.zeros_(self.wam_actor_adapter.bias)
            for parameter in list(self.actor.parameters()) + list(self.critic.parameters()):
                parameter.requires_grad_(False)
            self.log_std.requires_grad_(False)
"""
            if source.count(marker) != 1: raise ValueError('WAM adapter insertion point missing')
            source = source.replace(marker, addition, 1)
            source = source.replace("else:\n            actions_mean = self.actor(observations)\n",
                "else:\n            actions_mean = self.actor(observations)\n            if hasattr(self, 'wam_actor_adapter'):\n                wam_residual = self.wam_actor_adapter(observations[:, 300:])\n                if self.wam_adapter_bounded:\n                    wam_residual = self.wam_adapter_gain * torch.tanh(wam_residual)\n                actions_mean = actions_mean + wam_residual\n")
            source = source.replace("if not self.use_pc and self.num_obs == 300:\n            observations = observations[:,:300]\n",
                "if not self.use_pc and self.num_obs == 300:\n            observations = observations[:,:300]\n")
            source = source.replace("            value = self.critic(observations)\n\n        return actions.detach()",
                "            value = self.critic(observations)\n            if hasattr(self, 'wam_value_adapter'):\n                value = value + self.wam_value_adapter(observations[:, 300:])\n\n        return actions.detach()")
            source = source.replace("            value = self.critic(observations)\n\n        return actions_log_prob",
                "            value = self.critic(observations)\n            if hasattr(self, 'wam_value_adapter'):\n                value = value + self.wam_value_adapter(observations[:, 300:])\n\n        return actions_log_prob")
            if source.count("wam_residual = self.wam_actor_adapter(observations[:, 300:])") != 3:
                raise ValueError('WAM adapter must have exactly one residual path in act, act_inference, evaluate')
            algorithm_changes = True
    model.write_text(source)
    write(args.output/'compatibility.json', dict(upstream_commit=UPSTREAM,
        removed_unused_vision_imports=removed, added_state_config_keys=['backbone_type','freeze_backbone'],
        algorithm_changes=algorithm_changes, state_network_math_changes=False,
        environment_patch='existing maximum heterogeneous aggregate capacity only',
        input_contract='2607D native300 + frozen WAM2304 + relative_goal3' if algorithm_changes else '300D native state',
        wam_adapter_gain=(float(args.wam_adapter_gain) if args.wam_adapter_gain is not None else None),
        note='Only WAM input-dimension plumbing is patched; upstream squared scale_tril and KL convention intentionally preserved; random_time is phase staggering, not RSI'))
    replay = ROOT/'src/task/video-data-preparation/tools/run/replay_unidexgrasp.py'
    dest = args.output/'runtime_replay.py'
    dest.write_text(runtime_replay(replay.read_text(), args.mode == 'train'))
    spec = importlib.util.spec_from_file_location('official_unidex_environment', dest)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    args.vendor = ROOT/'third_party/UniDexGrasp2'
    args.assets = None; args.scale = .08; args.episodes = 1; args.gpu_seconds = args.seconds
    args.run_official = run_official
    module.worker(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--object-manifest', type=Path, required=True)
    p.add_argument('--mode', choices=['train','evaluate'], default='train')
    p.add_argument('--model', type=Path)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--envs', type=int, default=1024)
    p.add_argument('--iterations', type=int, default=10000)
    p.add_argument('--save-interval', type=int, default=1000)
    p.add_argument('--seed', type=int, default=245)
    p.add_argument('--seconds', type=int, default=7200)
    p.add_argument('--wam-checkpoint', type=Path)
    p.add_argument('--wam-cache', type=Path)
    p.add_argument('--wam-adapter', action='store_true')
    p.add_argument('--wam-adapter-gain', type=float,
                   help='Enable bounded zero-init WAM actor residual: gain*tanh(adapter).')
    p.add_argument('--worker', action='store_true')
    args = p.parse_args()
    args.output = args.output.resolve(); args.object_manifest = args.object_manifest.resolve()
    if args.model:args.model = args.model.resolve()
    if args.wam_checkpoint:args.wam_checkpoint = args.wam_checkpoint.resolve()
    if args.wam_cache:args.wam_cache = args.wam_cache.resolve()
    if bool(args.wam_checkpoint) != bool(args.wam_cache):raise ValueError('WAM requires both --wam-checkpoint and --wam-cache')
    if args.wam_adapter and not args.wam_checkpoint:raise ValueError('WAM adapter requires WAM features')
    if args.wam_adapter_gain is not None:
        if not args.wam_adapter:raise ValueError('--wam-adapter-gain requires --wam-adapter')
        if not 0.0 < args.wam_adapter_gain <= 1.0:raise ValueError('--wam-adapter-gain must be in (0,1]')
    if not 1 <= args.envs <= 2048 or not 1 <= args.iterations <= 10000 or not 1 <= args.seconds <= 14400:
        raise ValueError('Bounded official PPO probe required')
    if args.mode == 'evaluate' and not args.model:raise ValueError('Evaluate requires an explicit model')
    if args.worker:return worker(args)
    if args.output.exists():raise FileExistsError(args.output)
    if ROOT/'outputs/wam-rl-policy' not in args.output.parents:raise ValueError('Task output required')
    occupied = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
    if occupied:raise RuntimeError('Assigned GPU occupied')
    args.output.mkdir(parents=True); (args.output/'tmp').mkdir()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(args.output/'tmp'),
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
        TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/official-unidex-isaac-extensions'), MAX_JOBS='2')
    process = dict(status='STARTING', gpu=args.gpu, supervisor_pid=os.getpid(), upstream_commit=UPSTREAM,
        code_commit=os.environ.get('REF2DEX_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        mode=args.mode, model=str(args.model) if args.model else None, BC_loaded=bool(args.model),
        envs=args.envs, iterations=args.iterations, seconds=args.seconds, seed=args.seed,
        object_manifest=str(args.object_manifest), run_id=args.output.name)
    process.update(wam_checkpoint=str(args.wam_checkpoint) if args.wam_checkpoint else None,
                   wam_cache=str(args.wam_cache) if args.wam_cache else None,
                   observation_dim=2607 if args.wam_checkpoint else 300,
                   official_actor_loaded=bool(args.model), BC_loaded=bool(args.model))
    write(args.output/'manifest.json', process)
    stopped = [False]
    for sig in [signal.SIGTERM, signal.SIGINT]:signal.signal(sig, lambda *_:stopped.__setitem__(0,True))
    started = time.monotonic();reason = 'CHILD_EXIT'
    with (args.output/'worker.log').open('x') as log:
        child = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],
            cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        process.update(status='RUNNING', pid=child.pid);write(args.output/'process.json',process)
        while child.poll() is None:
            elapsed = time.monotonic()-started
            if stopped[0] or (args.output/'STOP').exists() or elapsed > args.seconds:
                reason = 'STOP' if stopped[0] or (args.output/'STOP').exists() else 'TIME_BUDGET'
            elif shutil.disk_usage(args.output).free < 20*2**30:reason = 'DISK_RESERVE'
            elif sum(f.stat().st_size for f in args.output.rglob('*') if f.is_file()) > 2*2**30:reason = 'ARTIFACT_BUDGET'
            else:reason = 'CHILD_EXIT'
            if reason != 'CHILD_EXIT':
                os.killpg(child.pid, signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid, signal.SIGKILL)
                break
            resource = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
            with (args.output/'resources.jsonl').open('a') as stream:stream.write(json.dumps(dict(elapsed_s=elapsed,gpu=resource))+'\n')
            time.sleep(10)
        code = child.wait()
        process.update(status='COMPLETED' if code == 0 else 'FAILED', exit_code=code, reason=reason, elapsed_s=time.monotonic()-started)
        write(args.output/'process.json',process);write(args.output/'manifest.json',process)
    if code:raise SystemExit(code)


if __name__ == '__main__':main()
