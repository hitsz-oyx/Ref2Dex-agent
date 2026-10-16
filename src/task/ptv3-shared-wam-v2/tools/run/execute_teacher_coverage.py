"""Four paired native rows: official calibration and three scratch BC heads."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
BASE = ROOT / 'outputs' / TASK.name
PANEL = ('s10_mug_pass_1', 's2_cup_pass_1', 's3_mug_drink_1',
         's4_cup_pass_1', 's5_cup_drink_2', 's5_mug_pass_1')
ROLES = ('teacher', 'original', 'expanded', 'balanced')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sequence', choices=PANEL, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--component-attribution', action='store_true',
                   help='Expanded student plus diagnostic own-state teacher wrist/finger substitutions')
    a = p.parse_args()
    roles = ('teacher', 'expanded', 'teacher_wrist', 'teacher_fingers') if a.component_attribution else ROLES
    if a.component_attribution and (a.smoke or a.sequence not in ('s3_mug_drink_1', 's5_cup_drink_2')):
        raise ValueError('Fixed two-sequence attribution panel without engineering shortcut')
    out = a.output.resolve()
    runtime = Path('/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train')
    if Path(sys.prefix).resolve() != runtime or out.exists() or BASE not in out.parents:
        raise ValueError('Pinned native runtime and fresh Task output required')
    if subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('Assigned GPU occupied')
    if shutil.disk_usage(ROOT).free < 20 * 2**30:
        raise RuntimeError('Disk reserve')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), TMPDIR=str(ROOT / 'tmp'),
        TORCH_EXTENSIONS_DIR=str(ROOT / 'tmp/consequence-official-generator/torch201-extensions'),
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    old = ROOT / 'src/task/consequence-evaluator'
    trajectory = ROOT / 'src/task/trajectory-policy'
    native_root = ROOT / 'third_party/DExplore/dexplore'
    sys.path[:0] = ['/home2/wyy/isaac-gym/isaacgym/python', str(TASK / 'src'),
        str(TASK.parent / 'ptv3-shared-wam/src'), str(old / 'src'), str(old / 'tools/audit'),
        str(trajectory / 'src'), str(trajectory / 'tools/run'), str(trajectory / 'tools/audit'),
        str(ROOT), str(native_root), str(ROOT / 'src/task/CmResidual/tools')]
    sys.path.append('/home2/wyy/.local/lib/python3.8/site-packages')
    from isaacgym import gymtorch  # Isaac must precede torch in this runtime.
    import dexplore_ddp_rank_bootstrap
    import torch
    import evaluate as native
    from learning import common_player
    from rl_games.common import player as base_player
    from probe_official_generator import install_legacy_player_compat, model_state
    from consequence_evaluator.native_reset import install_reset_patch
    from consequence_evaluator.physical_geometry import PhysicalGeometry, poses
    from consequence_evaluator.value_geometry import TableSupport
    from env.tasks.base_dexplore_task import DexploreTask
    from trajectory_policy.pd_retargeter import pd_from_action, decode_target, command
    from collect_competent_pd import OFFICIAL, gpu_state
    from execute_wam_native_bridge import partial_grasp_outcome
    from ptv3_shared_wam.native_inputs import resolve_native_inputs, install_selected_motion_contract
    from ptv3_shared_wam_v2.teacher_native import TeacherInputStudent

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(292)
    cache = json.loads((BASE / 'full-teacher-cache-20261015-r1/manifest.json').read_text())
    selected = resolve_native_inputs(dict(provenance=dict(source=cache['source']),
        config=dict(native_controller=cache['controller'])), a.sequence)
    if selected['split'] != 'val':
        raise ValueError('Fixed held-out sequences required')
    controls = 4 if a.smoke else selected['frames'] - 25
    checkpoints = {r: BASE / 'teacher-state-coverage-20261016-r1' /
                  ('expanded' if a.component_attribution else r) / 'final.pt' for r in roles[1:]}
    students = {r: TeacherInputStudent(torch.load(path, map_location='cpu')).cuda()
                for r, path in checkpoints.items()}
    cfg = json.loads((ROOT / 'outputs/consequence-evaluator/baseline-transfer-s3-20261007-r2/airplane_base/config.json').read_text())
    out.mkdir(parents=True)
    motions = out / 'motions'
    motions.mkdir()
    (motions / a.sequence).symlink_to(selected['motion'], target_is_directory=True)
    cfg['cfg_env'] = str((ROOT / 'third_party/DExplore' / cfg['cfg_env']).resolve())
    cwd = out / 'gym_runtime'
    assets = cwd / 'dexplore/data/assets'
    assets.parent.mkdir(parents=True)
    measurement_assets = Path('/home2/wyy/oyx_ws/dexplore/dexplore/data/assets')
    assets.symlink_to(measurement_assets, target_is_directory=True)
    manifest = dict(schema='ref2dex.teacher-coverage-native.v1', status='RUNNING',
        experiment_id='P-20261016-teacher-coverage-native', run_id=out.name,
        work_version='git commit', git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        roles=roles, seed=292, native_input=selected, native_controller=cache['controller'],
        checkpoints={r: str(v) for r, v in checkpoints.items()}, teacher=str(OFFICIAL),
        controls=controls, engineering_only=a.smoke, physical_gpu=a.gpu, pid=os.getpid(),
        no_teacher_intervention=not a.component_attribution, component_attribution=a.component_attribution,
        reference='native full1442 with delta1 and delta16 hand/object references',
        budget=dict(seconds=600, bytes=256 * 2**20),
        resource_authorization='2026-10-16 user permits all idle GPUs; no occupied GPU touched')
    write(out / 'manifest.json', manifest)
    install_legacy_player_compat(native, common_player, base_player)
    install_reset_patch()
    install_selected_motion_contract(DexploreTask, selected)
    logs, executed = {}, 0
    started = time.monotonic()

    class Player(native.EvalPlayer):
        def restore(self, filename):
            state = native.torch_ext.load_checkpoint(filename)
            self.model.load_state_dict(model_state(state['model']), strict=True)
            if self.normalize_input:
                self.running_mean_std.load_state_dict(state['running_mean_std'], strict=True)
            if self._normalize_amp_input:
                self._amp_input_mean_std.load_state_dict(state['amp_input_mean_std'], strict=True)

        def run(self):
            nonlocal executed
            task = self.env.task
            if task.num_envs != 4 or self.is_rnn:
                raise ValueError('Four non-recurrent rows required')
            task._state_init = DexploreTask.StateInit.Start
            task._hybrid_init_prob = 1.
            task._adaptive_kappa_enabled = False
            task._enable_early_termination = False
            obs = self.env_reset(torch.arange(4, device=task.device))
            self.get_batch_size(obs['obs'], 1)
            names = [Path(task.motion_file[int(i)]).name for i in task.data_id.cpu().tolist()]
            if (task.start_times != 0).any() or names != [a.sequence] * 4:
                raise ValueError('Matched frame0 sequence reset required')
            if any(n != selected['object'] for n in task.object_name):
                raise ValueError('Object identity differs')
            geometry = PhysicalGeometry(task, measurement_assets, distance_device=task.device)
            support = TableSupport(measurement_assets, task.device)
            offset, scale = task._pd_action_offset.clone(), task._pd_action_scale.clone()
            if dict(offset=offset.tolist(), scale=scale.tolist()) != cache['controller']:
                raise ValueError('PD controller differs from supervised labels')
            captured, actual_pd = None, None
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

            def save(**values):
                for key, value in values.items():
                    logs.setdefault(key, []).append(value.detach().cpu().numpy().copy())

            def record():
                hand, gap = geometry.measure(task)
                sup, foot = support.measure(task, geometry)
                tick = task.progress_buf.clone()
                ref = task.hoi_data[task.data_id, tick]
                save(q=task._dof_pos, dq=task._dof_vel, hand=hand, obj=poses(task._target_states),
                    velocity=task._target_states[:, 7:13], surface_gap=gap, support_gap=sup,
                    table_footprint=foot, wrist=task._rigid_body_pos[:, 6],
                    wrist_quat=task._rigid_body_rot[:, 6], reference_wrist=ref[:, 4:7],
                    reference_wrist_quat=ref[:, :4], reference_object=ref[:, 106:109], progress=tick)

            record()
            for tick in range(controls):
                with torch.inference_mode():
                    if not (task.progress_buf == tick).all() or obs['obs'].shape != (4, 1442):
                        raise ValueError('Native clock/input drift')
                    raw = obs['obs'].clone()
                    q = task._dof_pos.clone()
                    teacher = self.get_action(dict(obs, obs=raw.repeat(16, 1)), True)[:4].clamp(-1, 1).clone()
                    applied, intended = teacher.clone(), teacher.clone()
                    predictions = []
                    for row, role in enumerate(roles[1:], 1):
                        prediction = students[role](raw[row:row + 1])[0]
                        predictions.append(prediction)
                        target = decode_target(prediction, q[row])
                        intended[row], applied[row], _ = command(target, q[row], offset, scale)
                    student_intended = intended.clone()
                    if a.component_attribution:
                        # Teacher acts on each hybrid's own state, never borrows row0 state/actions.
                        intended[2, :6] = teacher[2, :6]
                        intended[3, 6:] = teacher[3, 6:]
                        applied = intended.clamp(-1, 1)
                    if not torch.isfinite(intended).all():
                        raise ValueError('Nonfinite student action')
                    expected = pd_from_action(applied, q, offset, scale)
                    save(raw_observation=raw, student_prediction=torch.stack(predictions),
                        teacher_query_action=teacher, intended=intended, applied=applied,
                        student_intended=student_intended,
                        clipped=(intended - applied).abs().amax(-1) > 1e-6)
                    obs, _, done, _ = self.env_step(self.env, applied.clone())
                    executed += 1
                    if not isinstance(obs, dict):
                        obs = dict(obs=obs)
                    if captured is None or not torch.equal(captured, applied) or (actual_pd - expected).abs().max() > 1e-6:
                        raise ValueError('Native applied action/PD contract failed')
                    if done.any():
                        raise ValueError('Unexpected reset in bounded prefix')
                    save(pd_targets=actual_pd)
                    record()
                if tick == 0 or (tick + 1) % 64 == 0 or tick + 1 == controls:
                    gpu = gpu_state(a.gpu)
                    if set(gpu['pids']) - {os.getpid()}:
                        raise RuntimeError('Foreign process on assigned GPU')
                    elapsed = time.monotonic() - started
                    report = dict(control=tick + 1, elapsed_s=elapsed,
                        eta_s=elapsed / (tick + 1) * (controls - tick - 1), gpu=gpu)
                    print(json.dumps(report), flush=True)
                    with (out / 'monitor.jsonl').open('a') as stream:
                        stream.write(json.dumps(report) + '\n')
            arrays = {key: np.stack(value) for key, value in logs.items()}
            np.savez_compressed(out / 'trajectory.npz', **arrays)
            outcomes = []
            for row, role in enumerate(roles):
                outcome = partial_grasp_outcome(dict(object_pose=arrays['obj'][:, row],
                    object_velocity=arrays['velocity'][:, row], surface_gap=arrays['surface_gap'][:, row],
                    support_gap=arrays['support_gap'][:, row], table_footprint=arrays['table_footprint'][:, row]))
                outcome.update(role=role, clipped_fraction=float(arrays['clipped'][:, row].mean()))
                outcomes.append(outcome)
            write(out / 'result.json', dict(outcomes=outcomes, scope='prefix hold45, no complete-task success claim'))
            manifest.update(status='COMPLETED', executed_controls=executed, elapsed_s=time.monotonic() - started)
            write(out / 'manifest.json', manifest)

    native.EvalPlayer = Player
    sys.argv = [sys.argv[0], '--task', 'Dexplore_Inspire', '--cfg_env', cfg['cfg_env'],
        '--cfg_train', str(native_root / 'data/cfg/train/rlg/inspire.yaml'), '--motion_file', str(motions),
        '--checkpoint', str(OFFICIAL), '--headless', '--num_envs', '4', '--seed', '292',
        '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--pipeline', 'gpu', '--graphics_device_id', '0',
        '--num_threads', '1', '--disable-early-termination', '--output', str(out / 'unused.json'),
        '--output_path', str(out / 'native')]
    previous_cwd = Path.cwd()

    def deadline(*_):
        raise TimeoutError('600s native budget')

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(600)
    try:
        os.chdir(cwd)
        native.main()
    except BaseException as error:
        manifest.update(status='FAILED', error=repr(error), executed_controls=executed)
        write(out / 'manifest.json', manifest)
        if logs:
            np.savez_compressed(out / 'partial.npz', **{k: np.stack(v) for k, v in logs.items() if v})
        raise
    finally:
        signal.alarm(0)
        os.chdir(previous_cwd)


if __name__ == '__main__':
    main()
