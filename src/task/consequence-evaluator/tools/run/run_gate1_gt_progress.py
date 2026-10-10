"""Fresh complete-prefix native replay, causal GT phase value, real mixed controls.

The fixed first Probe intervenes only at four early-contact decisions, then
continues the self-trained actor to the full reference end. No mid-state restore.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]; ROOT = TASK.parents[2]
sys.path[:0] = [str(ROOT), str(TASK / 'src')]
from consequence_evaluator.contracts import HAND_LINKS, K, K_EXEC, is_within
from consequence_evaluator.gate1 import SEEDS, QUERY_TICKS, CANDIDATES, candidate_plan, choose_candidate, episode_outcome, paired_counts
from consequence_evaluator.provenance import self_trained_ancestry, sha
from consequence_evaluator.twin import REQUIRED_NATIVE_STATE_KEYS, capture_native_rng, fingerprint

NATIVE_PYTHON = '/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python'


def write(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temporary.replace(path)


def load(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def save(path, value):
    with Path(path).open('xb') as stream:
        pickle.dump(value, stream, protocol=4)


def ensure_free(gpu):
    pids = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip()
    if pids:
        raise RuntimeError('GPU is occupied: ' + pids)


def native_worker(a):
    ensure_free(a.gpu)
    helper = TASK / 'tools/audit/probe_official_generator.py'
    spec = importlib.util.spec_from_file_location('official_runtime_compat', helper)
    compat = importlib.util.module_from_spec(spec); spec.loader.exec_module(compat)
    if Path(sys.prefix).resolve() != compat.RUNTIME:
        raise RuntimeError('native worker requires the verified archived runtime')
    scratch = ROOT / 'tmp/consequence-official-generator'; scratch.mkdir(exist_ok=True, parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), TMPDIR=str(ROOT / 'tmp'),
        TORCH_EXTENSIONS_DIR=str(scratch / 'torch201-extensions'), OMP_NUM_THREADS='2')
    native_root = ROOT / 'third_party/DExplore/dexplore'
    sys.path[:0] = [str(compat.ISAAC), str(native_root), str(ROOT / 'src/task/CmResidual/tools'),
                   str(ROOT / 'src/task/cm-interaction-oracle/src')]
    sys.path.append('/home2/wyy/.local/lib/python3.8/site-packages')
    # Bootstrap installs old NumPy aliases; evaluate imports Gym before Torch.
    import dexplore_ddp_rank_bootstrap
    import evaluate as native
    import torch
    from learning import common_player
    from rl_games.common import player as base_player
    from env.tasks.base_dexplore_task import DexploreTask
    from consequence_evaluator.native_reset import install_reset_patch
    from consequence_evaluator.physical_geometry import PhysicalGeometry
    from consequence_evaluator.value_geometry import TableSupport
    from consequence_evaluator.collection import pose_matrix
    from oracle_y_utility import align_native_reference_tables
    if torch.__version__ != '2.0.1+cu118':
        raise RuntimeError('archived Torch was shadowed')
    compat.install_legacy_player_compat(native, common_player, base_player)
    install_reset_patch(); torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    original_reset = DexploreTask._reset_ref_state_init
    def aligned_reset(task, ids):
        align_native_reference_tables(task); return original_reset(task, ids)
    DexploreTask._reset_ref_state_init = aligned_reset
    properties_spec = importlib.util.spec_from_file_location('native_property_codec', TASK / 'tools/audit/native_twin_probe.py')
    properties_module = importlib.util.module_from_spec(properties_spec); properties_spec.loader.exec_module(properties_module)
    normalize_properties = properties_module.normalize_properties
    config = json.loads((a.run_dir / 'config.json').read_text())
    trained = json.loads((a.run_dir / 'run_manifest.json').read_text())
    expected = load(a.prefix) if a.prefix else None
    query = a.query_tick
    if expected is not None and query > len(expected['actions']):
        raise ValueError('missing committed prefix controls')
    base = native.EvalPlayer
    started = time.monotonic()

    class GatePlayer(base):
        def restore(self, filename):
            value = native.torch_ext.load_checkpoint(filename)
            self.model.load_state_dict(compat.model_state(value['model']), strict=True)
            if self.normalize_input:
                self.running_mean_std.load_state_dict(value['running_mean_std'], strict=True)
            if self._normalize_amp_input:
                self._amp_input_mean_std.load_state_dict(value['amp_input_mean_std'], strict=True)

        @torch.no_grad()
        def run(self):
            self.is_deterministic = self.is_determenistic
            task = self.env.task
            if (self.is_rnn or task.num_envs != 1 or abs(task.dt - 1 / 30) > 1e-8
                    or abs(task.sim_params.dt - 1 / 60) > 1e-8 or task.control_freq_inv != 2):
                raise ValueError('fixed single-environment30Hz deterministic policy required')
            task._state_init = DexploreTask.StateInit.Start; task._hybrid_init_prob = 1.
            task._adaptive_kappa_enabled = False; task._enable_early_termination = False
            if task.dr_randomizations or task._motion_sampler is not None or task.projtype != 'None':
                raise ValueError('randomized/sampler/projectile replay unsupported')
            geometry = PhysicalGeometry(task, native_root / 'data/assets')
            table = TableSupport(native_root / 'data/assets', task.device)
            obs = self.env_reset(torch.arange(1, device=task.device))
            self.get_batch_size(obs['obs'], 1)
            if task.gym.get_frame_count(task.sim) != 0 or (task.start_times != 0).any():
                raise ValueError('fresh unstepped full-frame0 reset required')
            total = int(task.max_episode_length[task.data_id[0]]) - 1
            if total != 542:
                raise ValueError('frozen full543state reference required')
            physics = {'sim_params': normalize_properties(task.gym.get_sim_params(task.sim)), 'actors': []}
            for env in task.envs:
                actors = []
                for actor_index in range(task.gym.get_actor_count(env)):
                    handle = task.gym.get_actor_handle(env, actor_index)
                    actors.append(dict(name=task.gym.get_actor_name(env, actor_index),
                        shape=normalize_properties(task.gym.get_actor_rigid_shape_properties(env, handle)),
                        body=normalize_properties(task.gym.get_actor_rigid_body_properties(env, handle)),
                        dof=normalize_properties(task.gym.get_actor_dof_properties(env, handle))))
                physics['actors'].append(actors)
            controller = dict(model={k: v.cpu().numpy().copy() for k, v in self.model.state_dict().items()},
                normalize_input=bool(self.normalize_input), checkpoint_sha256=trained['checkpoint_sha256'])
            for key, normalizer in [('rms', getattr(self, 'running_mean_std', None)),
                                    ('amp_rms', getattr(self, '_amp_input_mean_std', None))]:
                controller[key] = None if normalizer is None else {k: v.cpu().numpy().copy() for k, v in normalizer.state_dict().items()}
            identity = dict(physics_hash=fingerprint(physics), controller_hash=fingerprint(controller))
            if expected is not None and expected['replay_identity'] != identity:
                raise ValueError('native physics/controller identity changed')
            state_keys = sorted(REQUIRED_NATIVE_STATE_KEYS | {k for k in ('reset_buf', 'actions', 'real_pd_tar') if isinstance(getattr(task, k, None), torch.Tensor)})
            fields = ('object_pose', 'hand_keypoints', 'surface_gap', 'support_gap',
                      'table_footprint', 'object_velocity', 'history')
            rows = {k: [] for k in fields}; hashes = []; rng_hashes = []; actions = []; ended = []
            def observe():
                points, gap = geometry.measure(task); support, footprint = table.measure(task, geometry)
                values = dict(object_pose=pose_matrix(task._target_states[0].cpu().numpy()),
                    hand_keypoints=points[0].cpu().numpy(), surface_gap=gap[0].cpu().numpy(),
                    support_gap=support[0].cpu().numpy(), table_footprint=footprint[0].cpu().numpy(),
                    object_velocity=task._target_states[0, 7:13].cpu().numpy(),
                    history=obs['obs'][0].cpu().numpy())
                for k, v in values.items(): rows[k].append(np.asarray(v).copy())
                state = {k: getattr(task, k).cpu().numpy().copy() for k in state_keys}
                hashes.append(fingerprint(state)); rng_hashes.append(fingerprint(capture_native_rng(torch)))
                index = len(hashes) - 1
                if task.gym.get_frame_count(task.sim) != index * task.control_freq_inv:
                    raise ValueError('unexpected simulator step count at tick%d' % index)
                if expected is not None and index <= query:
                    if hashes[-1] != expected['state_hashes'][index] or rng_hashes[-1] != expected['rng_hashes'][index]:
                        raise ValueError('declared native/RNG prefix replay mismatch at tick%d' % index)
                    for k in fields:
                        if not np.array_equal(rows[k][-1], expected[k][index]):
                            raise ValueError('measured prefix replay differs at tick%d: %s' % (index, k))
            executed = None; original_pre = task.pre_physics_step
            def capture(control):
                nonlocal executed
                executed = control.detach().cpu().numpy().copy()
                return original_pre(control)
            task.pre_physics_step = capture
            observe(); plan = candidate_plan(a.candidate)
            stop = total if a.finish else query + K
            for tick in range(stop):
                if time.monotonic() - started > 150:
                    raise TimeoutError('bounded native worker deadline')
                control = self.get_action(obs, True).clamp(-1, 1)
                if tick < query:
                    control = torch.as_tensor(expected['actions'][tick:tick + 1], device=self.device)
                elif not a.finish:
                    control = (control + torch.as_tensor(plan[tick - query], device=self.device)).clamp(-1, 1)
                obs, _, done, info = self.env_step(self.env, control.clone())
                if not isinstance(obs, dict): obs = {'obs': obs}
                self._post_step(info)
                actual = executed[0].copy()
                if not np.isfinite(actual).all() or np.abs(actual).max() > 1 + 1e-6:
                    raise ValueError('nonfinite/out-of-bounds native command')
                if tick < query and not np.array_equal(actual, expected['actions'][tick]):
                    raise ValueError('executed prefix control changed')
                if tick < query and bool(done[0]) != bool(expected['done'][tick]):
                    raise ValueError('prefix done signal changed')
                actions.append(actual); ended.append(bool(done[0])); observe()
                if done.any() and tick < stop - 1:
                    raise ValueError('native episode terminated before full requested window')
            packet = {k: np.asarray(v) for k, v in rows.items()}
            packet.update(actions=np.asarray(actions), done=np.asarray(ended),
                state_hashes=hashes, rng_hashes=rng_hashes, timestamps=np.arange(stop + 1) / 30,
                seed=a.seed, query_tick=query, candidate=a.candidate, residual_plan=plan,
                checkpoint_sha256=trained['checkpoint_sha256'], fresh_prefix_replay=True,
                replay_identity=identity, canonical_state_keys=state_keys)
            save(a.worker_output, packet)
            write(a.worker_output.with_suffix('.json'), dict(status='COMPLETED', steps=stop,
                prefix_steps=query, all_prefix_native_rng_and_measurements_exact=True,
                elapsed_s=time.monotonic() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                outcome=episode_outcome(packet) if a.finish else None))
    native.EvalPlayer = GatePlayer
    sys.argv = [sys.argv[0], '--task', 'Dexplore_Inspire', '--cfg_env', config['cfg_env'],
        '--cfg_train', str(native_root / 'data/cfg/train/rlg/inspire.yaml'),
        '--motion_file', config['motion_root'], '--checkpoint', trained['checkpoint'],
        '--headless', '--disable-early-termination', '--num_envs', '1', '--seed', str(a.seed),
        '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--pipeline', 'gpu', '--graphics_device_id', '0',
        '--output', str(a.worker_output.with_suffix('.unused.json')),
        '--output_path', str(a.worker_output.with_suffix('.runtime'))]
    os.chdir(ROOT / 'third_party/DExplore'); native.main()


def score_worker(a):
    ensure_free(a.gpu); os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    import torch
    from consequence_evaluator.reference_bank import load_reference_features, BANK_SCHEMA
    from consequence_evaluator.reference_progress import trajectory_features
    from consequence_evaluator.temporal_phase import LearnedReferenceProgress, ENCODER_SCHEMA
    torch.set_num_threads(1)
    meta = json.loads((a.reference / 'manifest.json').read_text())
    trained = json.loads((a.encoder / 'manifest.json').read_text())
    checkpoint = Path(trained['checkpoint'])
    if (meta['schema'] != BANK_SCHEMA or trained['schema'] != ENCODER_SCHEMA
            or trained['status'] != 'COMPLETED' or trained['reference_sha256'] != meta['reference_sha256']
            or sha(checkpoint) != trained['checkpoint_sha256']):
        raise ValueError('exact frozen physical-bank phase encoder required')
    if any(sha(p) != h for p, h in {**meta['sources'], **trained['sources']}.items()):
        raise ValueError('frozen phase model/reference source drift')
    references, _ = load_reference_features(a.reference, meta)
    matcher = LearnedReferenceProgress(references, torch.load(checkpoint, map_location='cpu', weights_only=False), 'cuda:0')
    packets = [load(p) for p in a.score_inputs]; values = []; progress_start = []
    for packet in packets:
        trace = matcher.align(trajectory_features(packet['object_pose'], packet['hand_keypoints'], packet['timestamps']))
        tick = packet['query_tick']; end = tick + K
        if len(packet['actions']) != end:
            raise ValueError('candidate GT future must end exactly at t+24')
        values.append(float(trace['progress'][end] - trace['progress'][tick]))
        progress_start.append(float(trace['progress'][tick]))
    if max(progress_start) - min(progress_start) > 1e-9:
        raise ValueError('candidate prefixes have different current progress')
    write(a.score_output, dict(values=values, progress_start=progress_start,
        choice=choose_candidate(values), model_role='phase-only; no evaluator',
        horizon=K, execution_horizon=K_EXEC, no_candidate_future_after_horizon=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--encoder', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--seconds', type=int, default=900)
    p.add_argument('--episodes', type=int, default=4)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--worker-output', type=Path)
    p.add_argument('--prefix', type=Path)
    p.add_argument('--query-tick', type=int, default=0)
    p.add_argument('--seed', type=int, default=282)
    p.add_argument('--candidate', type=int, default=0)
    p.add_argument('--finish', action='store_true')
    p.add_argument('--score-inputs', type=Path, nargs='+')
    p.add_argument('--score-output', type=Path)
    a = p.parse_args()
    for name in ('run_dir', 'reference', 'encoder', 'output'):
        setattr(a, name, getattr(a, name).resolve())
        if not is_within(getattr(a, name), ROOT / 'outputs/consequence-evaluator'):
            p.error('all inputs/outputs must be task-owned')
    if not 0 <= a.gpu <= 7 or not 1 <= a.episodes <= 4 or not 30 <= a.seconds <= 900:
        p.error('bounded one-GPU/<=4episode/<=900s Probe required')
    if a.worker:
        native_worker(a); return
    if a.score_inputs:
        score_worker(a); return
    if a.output.exists() or shutil.disk_usage(ROOT).free < 20 * 2**30:
        p.error('fresh owned output and20GiB reserve required')
    ensure_free(a.gpu); frozen = self_trained_ancestry(a.run_dir, ROOT / 'outputs/consequence-evaluator')
    config = json.loads((a.run_dir / 'config.json').read_text())
    native_root = ROOT / 'third_party/DExplore/dexplore'
    files = [Path(__file__).resolve(), TASK / 'docs/experiments/probes/P-20261008-gate1-gt-progress.md',
        a.run_dir / 'config.json', Path(config['input_manifest']),
        Path(config['cfg_env']), TASK / 'tools/audit/probe_official_generator.py',
        TASK / 'tools/audit/native_twin_probe.py',
        ROOT / 'src/task/CmResidual/tools/dexplore_ddp_rank_bootstrap.py',
        ROOT / 'src/task/cm-interaction-oracle/src/oracle_y_utility.py',
        *sorted((TASK / 'src/consequence_evaluator').glob('*.py')),
        *sorted(native_root.rglob('*.py')),
        *sorted(p for p in (native_root / 'data/assets').rglob('*') if p.is_file()),
        *sorted(Path(config['motion_root']).glob('*/interaction_hand_inspire.pt'))]
    for directory in (a.reference, a.encoder):
        meta = json.loads((directory / 'manifest.json').read_text())
        frozen.update(meta['sources']); files.append(directory / 'manifest.json')
        files.append(Path(meta['checkpoint']) if directory == a.encoder else directory / 'reference.npz')
    frozen.update({str(f): sha(f) for f in files})
    if any(sha(k) != v for k, v in frozen.items()):
        raise ValueError('Gate1 input drift')
    a.output.mkdir(parents=True); started = time.monotonic(); pairs = []; events = []
    manifest = dict(schema='ref2dex.consequence-gate1.gt-progress.v1', status='RUNNING',
        experiment_id='P-20261008-gate1-gt-progress', run_id=a.output.name, pid=os.getpid(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        sources=frozen, physical_gpu=a.gpu, seconds_budget=a.seconds, training_allowed=False,
        seeds=list(SEEDS[:a.episodes]), candidates=list(CANDIDATES), query_ticks=list(QUERY_TICKS),
        horizon=K, execution_horizon=K_EXEC, scope='four early-contact rolling decisions then full actor continuation',
        actor_role='owned self-trained Cm-off actor, not official reference generator',
        solver='fresh_simulator_complete_executed_prefix_replay; no mid-state restore',
        outcome_metric='recoverable_hold45_controlled_place_settle15', events=events, pairs=pairs)
    def check():
        if time.monotonic() - started >= a.seconds:
            raise TimeoutError('Gate1 campaign deadline')
        if shutil.disk_usage(ROOT).free < 20 * 2**30 or sum(f.stat().st_size for f in a.output.rglob('*') if f.is_file()) > 2**30:
            raise RuntimeError('Gate1 disk budget exceeded')
        if any(sha(k) != v for k, v in frozen.items()):
            raise ValueError('Gate1 source/input drift during execution')
    common = ['--run-dir', str(a.run_dir), '--reference', str(a.reference), '--encoder', str(a.encoder),
              '--output', str(a.output), '--gpu', str(a.gpu)]
    def launch(name, args, native=True):
        check(); command = [NATIVE_PYTHON if native else sys.executable, str(Path(__file__).resolve()), *common, *args]
        env = os.environ.copy(); env.update(PYTHONDONTWRITEBYTECODE='1',
            TMPDIR=str(ROOT / 'tmp'), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        if native: env['PYTHONNOUSERSITE'] = '1'
        else: env.pop('PYTHONNOUSERSITE', None)
        with (a.output / (name + '.log')).open('x') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=min(160, max(1, a.seconds - (time.monotonic() - started))), check=True)
        check()
        events.append(dict(worker=name, elapsed_s=time.monotonic() - started))
        manifest['elapsed_s'] = time.monotonic() - started
        write(a.output / 'manifest.json', manifest)
        print(json.dumps(events[-1]), flush=True)
    write(a.output / 'manifest.json', manifest)
    try:
        for seed in SEEDS[:a.episodes]:
            baseline_path = a.output / ('s%d-baseline.pkl' % seed)
            launch('s%d-baseline' % seed, ['--worker', '--worker-output', str(baseline_path), '--seed', str(seed), '--finish'])
            baseline = load(baseline_path); prefix_path = baseline_path; decisions = []
            for tick in QUERY_TICKS:
                paths = []
                for candidate in range(3):
                    name = 's%d-t%d-c%d' % (seed, tick, candidate); path = a.output / (name + '.pkl')
                    launch(name, ['--worker', '--worker-output', str(path), '--seed', str(seed),
                        '--prefix', str(prefix_path), '--query-tick', str(tick), '--candidate', str(candidate)])
                    paths.append(path)
                packets = [load(path) for path in paths]
                # First zero branch must reproduce the separately run baseline.
                if tick == QUERY_TICKS[0]:
                    if packets[0]['state_hashes'] != baseline['state_hashes'][:tick + K + 1]:
                        raise ValueError('candidate-zero continuation differs from frozen baseline')
                # Repeated zero branch at the first query protects hidden solver state.
                if seed == SEEDS[0] and tick == QUERY_TICKS[0]:
                    repeat = a.output / 'first-zero-repeat.pkl'
                    launch('first-zero-repeat', ['--worker', '--worker-output', str(repeat), '--seed', str(seed),
                        '--prefix', str(prefix_path), '--query-tick', str(tick)])
                    repeated = load(repeat)
                    if fingerprint(repeated) != fingerprint(packets[0]):
                        raise ValueError('first candidate-zero repeat is not exact')
                score = a.output / ('s%d-t%d-score.json' % (seed, tick))
                launch('s%d-t%d-score' % (seed, tick), ['--score-inputs', *map(str, paths), '--score-output', str(score)], native=False)
                decision = json.loads(score.read_text()); decisions.append(dict(tick=tick, **decision))
                # Only controls from the chosen8step prefix are committed. The
                # NEXT fresh native worker executes this full mixed prefix and
                # verifies every native/RNG/measured state, never joins suffix Z.
                chosen = packets[decision['choice']]; stop = tick + K_EXEC
                committed = {}
                for key, value in chosen.items():
                    if key in ('actions', 'done'): committed[key] = value[:stop].copy()
                    elif key in ('state_hashes', 'rng_hashes'): committed[key] = value[:stop + 1]
                    elif key in ('object_pose', 'hand_keypoints', 'surface_gap', 'support_gap',
                                 'table_footprint', 'object_velocity', 'history', 'timestamps'):
                        committed[key] = value[:stop + 1].copy()
                    else: committed[key] = value
                prefix_path = a.output / ('s%d-committed-through%d.pkl' % (seed, stop)); save(prefix_path, committed)
            terminal = a.output / ('s%d-rolling.pkl' % seed)
            launch('s%d-rolling' % seed, ['--worker', '--worker-output', str(terminal), '--seed', str(seed),
                '--prefix', str(prefix_path), '--query-tick', str(QUERY_TICKS[-1] + K_EXEC), '--finish'])
            actual = load(terminal)
            pairs.append(dict(seed=seed, baseline=episode_outcome(baseline), rolling=episode_outcome(actual),
                initial_state_hash=baseline['state_hashes'][0], decisions=decisions,
                every_committed_prefix_reexecuted_and_exact=True))
            print(json.dumps(dict(paired=paired_counts(pairs))), flush=True)
        result = paired_counts(pairs)
        unique = len({p['initial_state_hash'] for p in pairs})
        # This panel is a bounded Probe; no formal Gate1 claim from four episodes.
        status = 'PROMISING' if result['rescued'] > result['harmed'] else 'UNCLEAR'
        result.update(probe_status=status, unique_initial_native_states=unique, Gate1_formal_pass=False,
            scope=manifest['scope'], training_allowed=False, pairs=pairs,
            limitation='four early-contact decisions, few deterministic seeds; no whole-episode MPC or formal validation')
        write(a.output / 'result.json', result); manifest.update(status='COMPLETED', result=result)
    except BaseException as error:
        manifest.update(status='FAILED', error=repr(error)); raise
    finally:
        manifest['elapsed_s'] = time.monotonic() - started; write(a.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
