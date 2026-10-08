"""Bounded engineering check of fresh native prefix replay and twin controls.

Each arm runs in a new process/simulator. Outputs always remain engineering
artifacts, never an evaluator training dataset or a six-expert qualification.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / 'src'))
from consequence_evaluator.contracts import K, HAND_LINKS, is_within
from consequence_evaluator.collection import pose_matrix, smooth_residual
from consequence_evaluator.provenance import self_trained_ancestry, sha
from consequence_evaluator.twin import (
    REQUIRED_NATIVE_STATE_KEYS, TwinBranch, capture_native_snapshot,
    fingerprint, replay_provenance, validate_pair,
)


def write(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def trace_error(actual, expected):
    """Maximum numerical disagreement in the complete canonical prefix trace."""
    if len(actual) != len(expected):
        raise ValueError('prefix trace length changed')
    error = 0.
    for current, reference in zip(actual, expected):
        if set(current) != set(reference):
            raise ValueError('prefix trace state inventory changed')
        for name in current:
            first, second = np.asarray(current[name]), np.asarray(reference[name])
            if (first.shape != second.shape or first.dtype != second.dtype
                    or not np.isfinite(first).all() or not np.isfinite(second).all()):
                raise ValueError('prefix trace shape/dtype/finite mismatch: ' + name)
            if first.size:
                error = max(error, float(np.max(np.abs(first.astype('float64') - second))))
    return error


def normalize_properties(value):
    # Isaac structured arrays have unspecified alignment padding. Serialize
    # named fields, preserving physical values rather than padding bytes.
    if isinstance(value, np.ndarray) and value.dtype.names:
        return {name: normalize_properties(value[name]) for name in value.dtype.names}
    if isinstance(value, dict):
        return {key: normalize_properties(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize_properties(item) for item in value]
    return value


def ensure_gpu_free(gpu):
    occupied = subprocess.check_output(
        ['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader'],
        text=True).strip()
    memory = int(subprocess.check_output(
        ['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        text=True).strip())
    if occupied or memory > 1024:
        raise RuntimeError('GPU occupied before native probe: %s (%d MiB)' % (occupied, memory))


def worker(a, output, run, trained, config):
    ensure_gpu_free(a.gpu)
    scratch = ROOT / 'tmp/consequence-twin-probe'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), TMPDIR=str(scratch),
                      TORCH_EXTENSIONS_DIR=str(scratch / 'torch-extensions'),
                      PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
    sys.dont_write_bytecode = True
    native_root = ROOT / 'third_party/DExplore/dexplore'
    sys.path[:0] = [str(native_root), str(ROOT), str(ROOT / 'src/task/CmResidual/tools')]
    # Isaac must initialize before Torch; install legacy NumPy aliases first.
    import dexplore_ddp_rank_bootstrap  # noqa: F401
    import evaluate as native
    from env.tasks.base_dexplore_task import DexploreTask
    import torch
    from consequence_evaluator.native_reset import install_reset_patch
    from src.task.CmResidual.paired_evaluation import physical_property_value

    install_reset_patch()
    sys.path.insert(0, str(ROOT / 'src/task/cm-interaction-oracle/src'))
    from oracle_y_utility import align_native_reference_tables
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    original_reset = DexploreTask._reset_ref_state_init

    def aligned_reset(task, ids):
        align_native_reference_tables(task)
        return original_reset(task, ids)

    DexploreTask._reset_ref_state_init = aligned_reset
    base = native.EvalPlayer

    class ProbePlayer(base):
        @torch.no_grad()
        def run(self):
            task = self.env.task
            if self.is_rnn or task.num_envs != a.num_envs:
                raise ValueError('fixed non-RNN native engineering probe required')
            task._state_init = DexploreTask.StateInit.Start
            task._hybrid_init_prob = 1.
            task._adaptive_kappa_enabled = False
            task._enable_early_termination = False
            if task.dr_randomizations or task._motion_sampler is not None or task.projtype != 'None':
                raise ValueError('randomized/sampler/projectile twin state is unsupported')
            obs = self.env_reset(torch.arange(task.num_envs, device=task.device))
            self.get_batch_size(obs['obs'], 1)
            if task.gym.get_frame_count(task.sim) != 0 or (task.start_times != 0).any():
                raise ValueError('fresh unstepped full reference reset required')
            names = task.gym.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles[0])
            hand_ids = [names.index(name) for name in HAND_LINKS]

            def canonical():
                return {name: getattr(task, name).detach().cpu().numpy().copy()
                        for name in REQUIRED_NATIVE_STATE_KEYS}

            def anchors():
                bodies = task._rigid_body_state.view(task.num_envs, -1, 13)
                return (pose_matrix(task._target_states[0].detach().cpu().numpy()),
                        bodies[0, hand_ids, :3].detach().cpu().numpy().copy())

            executed = None
            original_pre_physics = task.pre_physics_step

            def capture_control(actions):
                nonlocal executed
                executed = actions.detach().cpu().numpy().copy()
                return original_pre_physics(actions)

            task.pre_physics_step = capture_control

            def step(control):
                nonlocal executed, obs
                executed = None
                obs, _, done, info = self.env_step(self.env, control.clone())
                if not isinstance(obs, dict):
                    obs = {'obs': obs}
                self._post_step(info)
                if executed is None or executed.shape != (task.num_envs, 18):
                    raise ValueError('native executed-action capture did not run')
                if not np.isfinite(executed).all() or np.max(np.abs(executed)) > 1 + 1e-6:
                    raise ValueError('native executed control is invalid')
                return done.detach().cpu().numpy().reshape(-1).astype(bool)

            trace, controls = [canonical()], []
            reference = None
            if a.worker_arm != 'a':
                with (output / 'a/payload.pkl').open('rb') as stream:
                    reference = pickle.load(stream)
            for tick in range(a.prefix_steps):
                control = self.get_action(obs, True).clamp(-1, 1)
                if reference is not None:
                    recorded = reference['prefix_actions'][tick]
                    actual = control.detach().cpu().numpy()
                    if not np.allclose(actual, recorded, atol=1e-6, rtol=0):
                        raise ValueError('fresh prefix policy control differs from reference')
                    control = torch.tensor(recorded, device=self.device)
                done = step(control)
                controls.append(executed.copy())
                trace.append(canonical())
                if done.any():
                    raise ValueError('episode ended within common prefix')
            controls = np.stack(controls) if controls else np.zeros((0, task.num_envs, 18), dtype='float32')
            error = trace_error(trace, reference['prefix_states']) if reference is not None else 0.
            prefix_trace = reference['prefix_states'] if reference is not None else trace
            prefix_controls = reference['prefix_actions'] if reference is not None else controls
            physics = {'sim_params': physical_property_value(task.gym.get_sim_params(task.sim)), 'actors': []}
            for env in task.envs:
                actors = []
                for index in range(task.gym.get_actor_count(env)):
                    handle = task.gym.get_actor_handle(env, index)
                    actors.append({
                        'name': task.gym.get_actor_name(env, index),
                        'shape': physical_property_value(task.gym.get_actor_rigid_shape_properties(env, handle)),
                        'body': physical_property_value(task.gym.get_actor_rigid_body_properties(env, handle)),
                        'dof': physical_property_value(task.gym.get_actor_dof_properties(env, handle)),
                    })
                physics['actors'].append(actors)
            physics = normalize_properties(physics)
            history_contract = {'source': 'raw native policy obs', 'shape': list(obs['obs'].shape[1:]),
                                'player_sha256': sha(native_root / 'learning/common_player.py')}
            controller = {'model': self.model.state_dict(),
                          'rms': self.running_mean_std.state_dict() if self.normalize_input else None,
                          'normalize_input': bool(self.normalize_input),
                          'checkpoint_sha256': trained['checkpoint_sha256']}
            if hasattr(self, '_amp_input_mean_std'):
                controller['amp_rms'] = self._amp_input_mean_std.state_dict()
            replay = replay_provenance(
                prefix_trace, prefix_controls[:, 0], replay_max_abs_error=error,
                physics_properties=physics, history_contract=history_contract,
                controller_identity={'checkpoint_sha256': trained['checkpoint_sha256']},
                physics_dt=task.sim_params.dt, control_dt=task.dt,
                sim_steps_per_control=task.control_freq_inv)
            pose, hand = anchors()
            pair_id = 'native-s%d-prefix%d' % (a.seed, a.prefix_steps)
            snapshot = capture_native_snapshot(
                pair_id, a.prefix_steps, task, controller_state=controller,
                rnn_state=getattr(self, 'states', None), observation=obs['obs'], scalars={},
                reset_ids={'default': task._reset_default_env_ids, 'reference': task._reset_ref_env_ids},
                history=obs['obs'][0], object_pose=pose, hand_keypoints=hand,
                torch_module=torch, replay=replay, is_rnn=self.is_rnn)
            # Both plans are completely specified before the first branch step.
            plan = smooth_residual(np.random.default_rng(a.seed), a.amplitude)
            if a.worker_arm == 'b':
                plan = -plan
            actions, poses, hands, ended = [], [pose], [hand], []
            for tick in range(K):
                base_control = self.get_action(obs, True).clamp(-1, 1)
                residual = torch.tensor(plan[tick], device=self.device).expand_as(base_control)
                done = step((base_control + residual).clamp(-1, 1))
                pose, hand = anchors()
                actions.append(executed[0].copy())
                poses.append(pose)
                hands.append(hand)
                ended.append(done[0])
                if done.any() and tick < K - 1:
                    raise ValueError('episode ended before the complete branch')
            branch = TwinBranch(snapshot.state_hash, pair_id,
                                'b' if a.worker_arm == 'b' else 'a', plan,
                                np.stack(actions), np.stack(poses), np.stack(hands),
                                np.asarray(ended, dtype=bool), snapshot.common_prefix_hash)
            arm_output = output / a.worker_arm
            arm_output.mkdir()
            payload = {'snapshot': snapshot, 'branch': branch,
                       'prefix_states': trace, 'prefix_actions': controls}
            with (arm_output / 'payload.pkl').open('wb') as stream:
                pickle.dump(payload, stream, protocol=4)
            write(arm_output / 'summary.json', {
                'engineering_only': True, 'training_allowed': False,
                'state_hash': snapshot.state_hash, 'rng_hash': snapshot.rng_hash,
                'common_prefix_hash': snapshot.common_prefix_hash,
                'replay': replay, 'prefix_error': error, 'branch_steps': K,
                'field_hashes': {name: fingerprint(value) for name, value in snapshot.state.items()},
                'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            })

    native.EvalPlayer = ProbePlayer
    sys.argv = [sys.argv[0], '--task', 'Dexplore_Inspire', '--cfg_env', config['cfg_env'],
                '--cfg_train', str(native_root / 'data/cfg/train/rlg/inspire.yaml'),
                '--motion_file', config['motion_root'], '--checkpoint', trained['checkpoint'],
                '--headless', '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--pipeline', 'gpu',
                '--graphics_device_id', '0', '--num_envs', str(a.num_envs), '--seed', str(a.seed),
                '--output', str(output / (a.worker_arm + '-unused.json')),
                '--output_path', str(output / (a.worker_arm + '-runtime'))]
    os.chdir(ROOT / 'third_party/DExplore')
    native.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--prefix-steps', type=int, default=8)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--amplitude', type=float, default=.08)
    parser.add_argument('--worker-arm', choices=('a', 'b', 'repeat'))
    a = parser.parse_args()
    output, run = a.output.resolve(), a.run_dir.resolve()
    if (not is_within(output, ROOT / 'outputs/consequence-evaluator')
            or not is_within(run, ROOT / 'outputs/consequence-evaluator')
            or not 0 <= a.prefix_steps <= 128 or a.num_envs != 1
            or not 1 <= a.seconds <= 900 or not 0 < a.amplitude <= .2 or a.gpu < 0 or a.seed < 0):
        parser.error('bounded owned engineering probe requires exactly one environment')
    trained = json.loads((run / 'run_manifest.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    frozen = self_trained_ancestry(run, ROOT / 'outputs/consequence-evaluator')
    if a.worker_arm is not None:
        worker(a, output, run, trained, config)
        return
    if output.exists() or shutil.disk_usage(ROOT).free < 20 * 2**30:
        parser.error('fresh owned output and >=20GiB free disk required')
    ensure_gpu_free(a.gpu)
    native_root = ROOT / 'third_party/DExplore/dexplore'
    paths = [Path(__file__), run / 'config.json', Path(config['cfg_env']),
             native_root / 'data/cfg/train/rlg/inspire.yaml',
             ROOT / 'src/task/CmResidual/tools/dexplore_ddp_rank_bootstrap.py',
             ROOT / 'src/task/CmResidual/paired_evaluation.py',
             ROOT / 'src/task/cm-interaction-oracle/src/oracle_y_utility.py',
             *sorted((TASK / 'src/consequence_evaluator').glob('*.py')),
             *sorted(native_root.rglob('*.py')),
             *sorted(Path(config['motion_root']).glob('*/interaction_hand_inspire.pt')),
             *sorted(p for p in (native_root / 'data/assets').rglob('*') if p.is_file())]
    frozen.update({str(path.absolute()): sha(path) for path in paths})
    output.mkdir(parents=True)
    started = time.monotonic()
    manifest = {'status': 'RUNNING', 'engineering_only': True, 'training_allowed': False,
                'run_id': output.name, 'git_commit': subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'seed': a.seed, 'physical_gpu': a.gpu, 'prefix_steps': a.prefix_steps,
                'num_envs': a.num_envs, 'sources': frozen,
                'seconds_budget': a.seconds, 'output_budget_bytes': 2**30}
    write(output / 'run_manifest.json', manifest)
    try:
        for arm in ('a', 'b', 'repeat'):
            remaining = a.seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError('native twin engineering deadline')
            command = [sys.executable, str(Path(__file__).resolve()), '--run-dir', str(run),
                       '--output', str(output), '--gpu', str(a.gpu), '--seed', str(a.seed),
                       '--prefix-steps', str(a.prefix_steps), '--num-envs', str(a.num_envs),
                       '--amplitude', str(a.amplitude), '--worker-arm', arm]
            with (output / (arm + '.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                               timeout=remaining, check=True, cwd=ROOT)
            if sum(path.stat().st_size for path in output.rglob('*') if path.is_file()) > 2**30:
                raise RuntimeError('native twin engineering output exceeded1GiB')
        payloads = []
        for arm in ('a', 'b', 'repeat'):
            with (output / arm / 'payload.pkl').open('rb') as stream:
                payloads.append(pickle.load(stream))
        first, second, repeat = payloads
        result = {'engineering_only': True, 'training_allowed': False,
                  'changed_state_fields': [name for name in first['snapshot'].state
                      if fingerprint(first['snapshot'].state[name]) != fingerprint(second['snapshot'].state[name])],
                  'repeat_common_prefix_equal': first['snapshot'].common_prefix_hash == repeat['snapshot'].common_prefix_hash,
                  'repeat_actions_equal': np.array_equal(first['branch'].actions, repeat['branch'].actions),
                  'repeat_object_poses_equal': np.array_equal(first['branch'].object_poses, repeat['branch'].object_poses)}
        try:
            result['pair_contract'] = validate_pair(first['snapshot'], second['snapshot'],
                                                   first['branch'], second['branch'])
            result['status'] = 'ENGINEERING_PASS' if all(result[name] for name in (
                'repeat_common_prefix_equal', 'repeat_actions_equal', 'repeat_object_poses_equal')) else 'REPEAT_NOT_EXACT'
        except ValueError as error:
            result.update(status='REPLAY_NOT_EXACT', error=str(error))
        write(output / 'result.json', result)
        if any(sha(path) != expected for path, expected in frozen.items()):
            raise RuntimeError('native twin source/input drift')
        manifest['status'] = 'COMPLETED'
        manifest['engineering_result'] = result['status']
        print(json.dumps(result, indent=2))
    except BaseException as error:
        manifest.update(status='FAILED', error=repr(error))
        raise
    finally:
        manifest['elapsed_s'] = time.monotonic() - started
        write(output / 'run_manifest.json', manifest)


if __name__ == '__main__':
    main()
