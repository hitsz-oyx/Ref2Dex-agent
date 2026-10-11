"""Bounded official actor evaluation on one authenticated corrected reference."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from collect_competent_pd import ROOT, TASK, OLD, RUNTIME, OFFICIAL, OFFICIAL_SHA, sha, write, gpu_state


def summarize(data):
    sys.path.insert(0, str(OLD / 'src'))
    from consequence_evaluator.gate1 import episode_outcome
    rows = []
    ref = data['reference_obj']
    for env in range(data['obj'].shape[1]):
        row = episode_outcome(dict(object_pose=data['obj'][:, env],
            object_velocity=data['velocity'][:, env], surface_gap=data['surface_gap'][:, env],
            support_gap=data['support_gap'][:, env], table_footprint=data['table_footprint'][:, env]))
        error = data['obj'][:, env, :3, 3] - ref[:, :3, 3]
        relative = np.matmul(ref[:, :3, :3].swapaxes(-1, -2), data['obj'][:, env, :3, :3])
        angle = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
        row.update(env=env, reference_position_rmse_m=float(np.sqrt(np.mean(np.sum(error**2, -1)))),
            reference_rotation_mean_rad=float(angle.mean()), endpoint_position_error_m=float(np.linalg.norm(error[-1])),
            hand_position_rmse_m=float(np.sqrt(np.mean(np.sum((data['hand16'][:, env] - data['reference_hand'])**2, -1)))),
            action_saturation_rate=float((np.abs(data['action'][:, env]) >= .999999).any(-1).mean()))
        rows.append(row)
    n = len(rows)
    return dict(status='PROMISING' if sum(r['maximum_held_frames'] >= 45 for r in rows) >= 6 else 'UNPROMISING',
        episodes=n, lift5=sum(r['maximum_held_frames'] >= 5 for r in rows),
        hold45=sum(r['maximum_held_frames'] >= 45 for r in rows),
        controlled_place=sum(r['success'] for r in rows),
        median_held_frames=float(np.median([r['maximum_held_frames'] for r in rows])),
        reference_position_rmse_m=float(np.sqrt(np.mean([r['reference_position_rmse_m']**2 for r in rows]))),
        reference_rotation_mean_rad=float(np.mean([r['reference_rotation_mean_rad'] for r in rows])),
        endpoint_position_error_m=float(np.mean([r['endpoint_position_error_m'] for r in rows])),
        reference_terminal_support=bool(data['reference_terminal_support']),
        reference_terminal_support_gap_m=float(data['reference_terminal_support_gap']),
        reference_maximum_rise_m=float(np.max(ref[:, 2, 3] - ref[0, 2, 3])),
        reference_terminal_rise_m=float(ref[-1, 2, 3] - ref[0, 2, 3]), outcomes=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--idle-context-pid', type=int, action='append', default=[])
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists() or ROOT / 'outputs/trajectory-policy' not in out.parents or Path(sys.prefix).resolve() != RUNTIME.resolve():
        raise ValueError('fresh task output and pinned legacy runtime required')
    before = gpu_state(args.gpu)
    allowed = set(args.idle_context_pid)
    if set(before['pids']) - allowed or before['used_mib'] > 512 or before['utilization'] > 5:
        raise RuntimeError('occupied GPU: ' + repr(before))
    commands = {pid: subprocess.check_output(['ps', '-p', str(pid), '-o', 'args='], text=True).strip()
                for pid in allowed if pid in before['pids']}
    bank_path = ROOT / 'outputs/consequence-evaluator/corrected-expert-inputs-20261007-r2/manifest.json'
    bank = json.loads(bank_path.read_text())
    motion = next(row for row in bank['motion_inputs'] if row['sequence'] == args.sequence)
    if sha(motion['path']) != motion['sha256'] or sha(OFFICIAL) != OFFICIAL_SHA:
        raise ValueError('motion/checkpoint input drift')
    out.mkdir(parents=True)
    motions = out / 'motions'
    motions.mkdir()
    (motions / args.sequence).symlink_to(Path(motion['path']).parent)
    cfg_path = ROOT / 'outputs/consequence-evaluator/baseline-transfer-inputs-20261007-r1/cfg/inspire.yaml'
    scratch = ROOT / 'tmp/trajectory-policy/official-reference-screen'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(scratch), OMP_NUM_THREADS='2',
        PYTHONDONTWRITEBYTECODE='1', TORCH_EXTENSIONS_DIR=str(ROOT / 'tmp/consequence-official-generator/torch201-extensions'))
    sys.dont_write_bytecode = True
    native_root = ROOT / 'third_party/DExplore/dexplore'
    sys.path[:0] = [str(Path('/home2/wyy/isaac-gym/isaacgym/python')), str(TASK / 'src'),
        str(OLD / 'src'), str(OLD / 'tools/audit'), str(ROOT), str(native_root), str(ROOT / 'src/task/CmResidual/tools')]
    sys.path.append('/home2/wyy/.local/lib/python3.8/site-packages')
    from isaacgym import gymtorch  # noqa: F401 -- before torch
    import dexplore_ddp_rank_bootstrap  # noqa: F401
    import torch
    import evaluate as native
    from learning import common_player
    from rl_games.common import player as base_player
    from probe_official_generator import install_legacy_player_compat, model_state
    from consequence_evaluator.native_reset import install_reset_patch
    from consequence_evaluator.physical_geometry import PhysicalGeometry, poses
    from consequence_evaluator.value_geometry import TableSupport
    from env.tasks.base_dexplore_task import DexploreTask
    from trajectory_policy.reference_actor import reference_from_processed
    from trajectory_policy.pd_retargeter import pd_from_action
    if torch.__version__ != '2.0.1+cu118':
        raise ValueError('pinned Torch required')
    torch.set_num_threads(2)
    torch.manual_seed(191)
    torch.set_float32_matmul_precision('highest')
    train_cfg = native_root / 'data/cfg/train/rlg/inspire.yaml'
    assets = native_root / 'data/assets'
    object_name = args.sequence.split('_')[1]
    files = [Path(__file__), Path(__file__).with_name('collect_competent_pd.py'), bank_path, cfg_path,
        train_cfg, OFFICIAL, Path(motion['path']), OLD / 'tools/audit/probe_official_generator.py',
        TASK / 'docs/experiments/probes/P-20261011-official-reference-screen.md',
        TASK / 'src/trajectory_policy/reference_actor.py', TASK / 'src/trajectory_policy/pd_retargeter.py',
        *list((OLD / 'src/consequence_evaluator').glob('*.py')), *list(native_root.rglob('*.py')),
        assets / ('mjcf/' + object_name + '.urdf'), assets / ('mjcf/objects/' + object_name + '/' + object_name + '.obj'),
        assets / 'mjcf/table.urdf', assets / 'mjcf/objects/table/table.obj',
        assets / 'inspire_hand_new/inspire_hand_right.urdf']
    hashes = {str(path.resolve()): sha(path) for path in files}
    manifest = dict(schema='ref2dex.official-reference-screen.v1', status='RUNNING',
        experiment_id='P-20261011-official-reference-screen', task='trajectory-policy', run_id=out.name,
        sequence=args.sequence, motion=motion, checkpoint_sha256=OFFICIAL_SHA, seed=191, num_envs=8,
        physical_gpu=args.gpu, gpu_before=before, preexisting_idle_contexts=commands,
        training_allowed=False, actor_rows=64, early_termination=False, frame0=True,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), input_sha256=hashes)
    write(out / 'manifest.json', manifest)
    install_legacy_player_compat(native, common_player, base_player)
    install_reset_patch()
    started = time.monotonic()

    class Player(native.EvalPlayer):
        def restore(self, filename):
            value = native.torch_ext.load_checkpoint(filename)
            self.model.load_state_dict(model_state(value['model']), strict=True)
            if self.normalize_input:
                self.running_mean_std.load_state_dict(value['running_mean_std'], strict=True)
            if self._normalize_amp_input:
                self._amp_input_mean_std.load_state_dict(value['amp_input_mean_std'], strict=True)

        def run(self):
            task = self.env.task
            # Native hardObjectOversampling duplicates names containing small.
            # Accept only aliases of this exact authenticated input, never a
            # second distinct reference or silently changed sampling mixture.
            if (task.num_envs != 8 or self.is_rnn or task.num_motions < 1
                    or any(Path(path).resolve() != Path(motion['path']).parent.resolve()
                           for path in task.motion_file)):
                raise ValueError('single unique motion/8env/nonrecurrent required')
            task._state_init = DexploreTask.StateInit.Start
            task._hybrid_init_prob = 1.
            task._adaptive_kappa_enabled = False
            task._enable_early_termination = False
            obs = self.env_reset(torch.arange(8, device=task.device))
            self.get_batch_size(obs['obs'], 1)
            frames = int(task.max_episode_length[0])
            if (frames != motion['frames'] or (task.max_episode_length != frames).any()
                    or (task.start_times != 0).any() or task.ball_size != 1
                    or any(not torch.equal(task.hoi_data[0], task.hoi_data[i]) for i in range(task.num_motions))):
                raise ValueError('source duration/frame0/scale mismatch')
            ref = reference_from_processed(task.hoi_data[0, :frames].cpu().numpy())
            geometry = PhysicalGeometry(task, assets, distance_device=task.device)
            support = TableSupport(assets, task.device)
            offset, scale = task._pd_action_offset.clone(), task._pd_action_scale.clone()
            manifest.update(controls=frames - 1, native_loaded_aliases=task.num_motions,
                native_controller=dict(offset=offset.tolist(), scale=scale.tolist()))
            write(out / 'manifest.json', manifest)
            capture = actual_pd = None
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
            logs = {k: [] for k in ('q', 'dq', 'hand16', 'obj', 'velocity', 'surface_gap',
                'support_gap', 'table_footprint', 'action', 'applied', 'pd_targets', 'obs')}

            def record():
                _, gap = geometry.measure(task)
                sup, foot = support.measure(task, geometry)
                bodies = task._rigid_body_state.view(8, -1, 13)
                current = dict(q=task._dof_pos, dq=task._dof_vel,
                    hand16=bodies[:, task._key_body_ids, :3], obj=poses(task._target_states),
                    velocity=task._target_states[:, 7:13], surface_gap=gap, support_gap=sup, table_footprint=foot)
                for key, value in current.items():
                    logs[key].append(value.cpu().numpy().copy())
                return current

            current = record()
            # Measure reference terminal support using the same mesh and table;
            # preserve simulator tensors and do not run a physics step here.
            previous = task._target_states.clone()
            terminal = task.hoi_data[0, frames - 1]
            task._target_states[:, :3] = terminal[106:109]
            task._target_states[:, 3:7] = terminal[109:113]
            terminal_gap, terminal_foot = support.measure(task, geometry)
            task._target_states[:] = previous
            ref_supported = bool((terminal_foot & (terminal_gap.abs() <= .02)).all())
            work = time.monotonic()
            for tick in range(frames - 1):
                with torch.no_grad():
                    raw = obs['obs']
                    action = self.get_action(dict(obs, obs=raw.repeat(8, 1)), True)[:8].clamp(-1, 1).clone()
                    expected = pd_from_action(action, current['q'], offset, scale)
                    if not torch.isfinite(action).all():
                        raise FloatingPointError('nonfinite action')
                    logs['obs'].append(raw.cpu().numpy().copy())
                    obs, _, done, _ = self.env_step(self.env, action.clone())
                    if not isinstance(obs, dict):
                        obs = {'obs': obs}
                    if not torch.equal(capture, action) or (expected - actual_pd).abs().max() > 1e-6:
                        raise ValueError('native command/PD mismatch')
                    for key, value in dict(action=action, applied=capture, pd_targets=actual_pd).items():
                        logs[key].append(value.cpu().numpy().copy())
                    current = record()
                    if tick < frames - 2 and done.any():
                        raise ValueError('unexpected reset/termination')
                if tick == 0 or (tick + 1) % 128 == 0:
                    state = gpu_state(args.gpu)
                    foreign = set(state['pids']) - {os.getpid()}
                    if foreign - allowed or any(state['process_mib'][pid] > 512 for pid in foreign):
                        np.savez_compressed(out / 'partial.npz', **{k: np.stack(v) for k, v in logs.items() if v})
                        raise RuntimeError('foreign GPU work: ' + repr(state))
                    monitor = dict(sequence=args.sequence, control=tick + 1, elapsed_s=time.monotonic() - started,
                        eta_s=(time.monotonic() - work) / (tick + 1) * (frames - 2 - tick), gpu=state)
                    print(json.dumps(monitor), flush=True)
                    with (out / 'monitor.jsonl').open('a') as stream:
                        stream.write(json.dumps(monitor) + '\n')
            data = {k: np.stack(v) for k, v in logs.items()}
            data.update(reference_obj=ref['obj'], reference_hand=ref['hand_pos'],
                reference_terminal_support=ref_supported, reference_terminal_support_gap=float(terminal_gap.mean()))
            if any(not np.isfinite(v).all() for v in data.values()):
                raise FloatingPointError('nonfinite trace')
            np.savez_compressed(out / 'trajectory.npz', **data)
            result = summarize(data)
            write(out / 'result.json', result)
            if any(sha(path) != digest for path, digest in hashes.items()):
                raise ValueError('input drift during execution')
            manifest.update(status='COMPLETED', elapsed_s=time.monotonic() - started,
                trajectory_sha256=sha(out / 'trajectory.npz'), torch_peak_bytes=torch.cuda.max_memory_allocated())
            write(out / 'manifest.json', manifest)
            print(json.dumps({k: v for k, v in result.items() if k != 'outcomes'}), flush=True)

    native.EvalPlayer = Player
    sys.argv = [sys.argv[0], '--task', 'Dexplore_Inspire', '--cfg_env', str(cfg_path), '--cfg_train', str(train_cfg),
        '--motion_file', str(motions), '--checkpoint', str(OFFICIAL), '--headless', '--num_envs', '8', '--seed', '191',
        '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--pipeline', 'gpu', '--graphics_device_id', '0',
        '--num_threads', '1', '--disable-early-termination', '--output', str(out / 'unused.json'), '--output_path', str(out / 'native')]
    previous_cwd = Path.cwd()

    def deadline(signum, frame):
        raise TimeoutError('reference screening exceeded300s')

    previous_handler = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(300)
    try:
        os.chdir(ROOT / 'third_party/DExplore')
        native.main()
    except BaseException as error:
        manifest.update(status='FAILED', error=repr(error), elapsed_s=time.monotonic() - started)
        write(out / 'manifest.json', manifest)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        os.chdir(previous_cwd)


if __name__ == '__main__':
    main()
