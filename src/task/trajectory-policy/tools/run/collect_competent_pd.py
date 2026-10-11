"""Fresh official demonstrations or frozen actual-PD retargeter oracle execution."""
import argparse
import hashlib
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
RUNTIME = Path('/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train')
OFFICIAL = Path('/home2/wyy/oyx_ws/_external/dexplore_official_v120/checkpoint/inspire.pth')
OFFICIAL_SHA = '8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def gpu_state(gpu):
    row = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True)
    utilization, memory = map(int, row.strip().split(','))
    pids = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).split()
    return dict(utilization=utilization, used_mib=memory, pids=[int(v) for v in pids])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--seed', type=int, default=180)
    args = parser.parse_args()
    out = args.output.resolve()
    evaluating = args.source is not None
    if (out.exists() or ROOT / 'outputs/trajectory-policy' not in out.parents
            or Path(sys.prefix).resolve() != RUNTIME.resolve()
            or evaluating != (args.model is not None) or not 100 <= args.seed <= 299):
        raise ValueError('fresh task output, archived runtime and source/model pair required')
    before = gpu_state(args.gpu)
    if before['pids'] or before['utilization'] > 10 or before['used_mib'] > 512:
        raise RuntimeError('GPU occupied: ' + repr(before))
    scratch = ROOT / 'tmp/trajectory-policy/competent-pd'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(scratch),
                      TORCH_EXTENSIONS_DIR=str(ROOT / 'tmp/consequence-official-generator/torch201-extensions'),
                      OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode = True
    native_root = ROOT / 'third_party/DExplore/dexplore'
    sys.path[:0] = [str(Path('/home2/wyy/isaac-gym/isaacgym/python')), str(TASK / 'src'),
        str(OLD / 'src'), str(OLD / 'tools/audit'), str(ROOT), str(native_root),
        str(ROOT / 'src/task/CmResidual/tools')]
    # Optional joblib fallback only; archived Torch/NumPy/rl_games retain precedence.
    sys.path.append('/home2/wyy/.local/lib/python3.8/site-packages')
    from isaacgym import gymtorch  # noqa: F401 -- before Torch
    import dexplore_ddp_rank_bootstrap  # noqa: F401 -- legacy Horovod facade
    import torch
    import evaluate as native
    from learning import common_player
    from rl_games.common import player as base_player
    from probe_official_generator import install_legacy_player_compat, model_state
    from consequence_evaluator.native_reset import install_reset_patch
    from consequence_evaluator.physical_geometry import PhysicalGeometry, poses
    from consequence_evaluator.value_geometry import TableSupport
    from consequence_evaluator.gate1 import episode_outcome
    from env.tasks.base_dexplore_task import DexploreTask
    from trajectory_policy.pd_retargeter import (command, decode_target, load_model,
                                               pd_from_action)
    from trajectory_policy.retargeter_data import features
    if torch.__version__ != '2.0.1+cu118' or sha(OFFICIAL) != OFFICIAL_SHA:
        raise ValueError('pinned official runtime/checkpoint identity mismatch')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(args.seed)
    cfg_path = args.run_dir / 'config.json'
    cfg = json.loads(cfg_path.read_text())
    motion_root = Path(cfg['motion_root'])
    motions = list(motion_root.glob('*/interaction_hand_inspire.pt'))
    if len(motions) != 1:
        raise ValueError('exactly one frozen airplane reference required')
    train_cfg = native_root / 'data/cfg/train/rlg/inspire.yaml'
    files = [Path(__file__), cfg_path, Path(cfg['cfg_env']), train_cfg, OFFICIAL,
             TASK / 'src/trajectory_policy/pd_retargeter.py',
             TASK / 'src/trajectory_policy/retargeter_data.py', *motions,
             OLD / 'tools/audit/probe_official_generator.py',
             *list((OLD / 'src/consequence_evaluator').glob('*.py')),
             *list(native_root.rglob('*.py'))]
    source = model = None
    if evaluating:
        source_manifest = json.loads((args.source / 'manifest.json').read_text())
        if (source_manifest.get('status') != 'COMPLETED'
                or source_manifest.get('training_allowed') is not True
                or source_manifest['checkpoint_sha256'] != OFFICIAL_SHA):
            raise ValueError('fresh completed training-eligible official source required')
        for path, digest in source_manifest['input_sha256'].items():
            if sha(path) != digest:
                raise ValueError('teacher input drift: ' + path)
        if sha(args.source / 'trajectory.npz') != source_manifest['trajectory_sha256']:
            raise ValueError('teacher trajectory drift')
        with np.load(args.source / 'trajectory.npz', allow_pickle=False) as stream:
            source = {key: stream[key].copy() for key in ('hand', 'pd_targets', 'q', 'goal')}
        fit_result = json.loads((args.model.parent / 'result.json').read_text())
        if not fit_result.get('physical_screen_pass') or sha(args.model) != fit_result['checkpoint_sha256']['tau']:
            raise ValueError('offline precision screen and exact selected model required')
        packet = torch.load(args.model, map_location='cuda:0')
        model = load_model(packet, 'cuda:0')
        files += [args.source / 'manifest.json', args.source / 'trajectory.npz',
                  args.model, args.model.parent / 'manifest.json', args.model.parent / 'result.json']
    hashes = {str(path.resolve()): sha(path) for path in files}
    roles = (['teacher'] * 8 + ['source_pd'] * 8 + ['gt_tau_R'] * 8
             if evaluating else ['teacher'] * 32)
    out.mkdir(parents=True)
    manifest = dict(schema='ref2dex.competent-pd-rollout.v1', status='RUNNING',
        task='trajectory-policy', experiment_id='P-20261011-competent-pd-retargeter',
        run_id=out.name, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=ROOT, text=True).strip(), physical_gpu=args.gpu, seed=args.seed, roles=roles,
        controls=542, dt=1 / 30, checkpoint=str(OFFICIAL), checkpoint_sha256=OFFICIAL_SHA,
        training_allowed=not evaluating, source_origin='official DExplore demonstration',
        input_sha256=hashes, gpu_before=before,
        evaluation='full-reference oracle R mechanism, not independent high-level policy',
        inference='current measured state87 + oracle future actual hand24 -> actual PD target',
        teacher_inference_rows=64, source_envs=list(range(24, 32)) if evaluating else None,
        target_alignment='q[t],hand[t] -> PD[t] -> hand[t+1]; no future q/object input')
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
            n = task.num_envs
            if n != len(roles) or self.is_rnn or str(task.device) != 'cuda:0':
                raise ValueError('fixed GPU-pipeline nonrecurrent group required')
            task._state_init = DexploreTask.StateInit.Start
            task._hybrid_init_prob = 1.
            task._adaptive_kappa_enabled = False
            task._enable_early_termination = False
            obs = self.env_reset(torch.arange(n, device=task.device))
            self.get_batch_size(obs['obs'], 1)
            if (task.start_times != 0).any():
                raise ValueError('real frame0 reset required')
            geometry = PhysicalGeometry(task, native_root / 'data/assets', distance_device=task.device)
            support = TableSupport(native_root / 'data/assets', task.device)
            offset, scale = task._pd_action_offset.clone(), task._pd_action_scale.clone()
            manifest['native_controller'] = dict(offset=offset.tolist(), scale=scale.tolist())
            terminal_ref = task.hoi_data[task.data_id, task.max_episode_length[task.data_id] - 1]
            root = torch.zeros(n, 13, device=task.device)
            root[:, :3], root[:, 3:7] = terminal_ref[:, 106:109], terminal_ref[:, 109:113]
            goal = poses(root)
            goal_np = goal.cpu().numpy().copy()
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
            logs = {key: [] for key in ('q', 'dq', 'hand', 'obj', 'velocity', 'surface_gap',
                'support_gap', 'table_footprint', 'action', 'applied', 'pd_targets', 'intended',
                'clipped', 'requested_pd', 'state_features', 'tau_features', 'encoded_target')}

            def record():
                hand, gap = geometry.measure(task)
                sup, foot = support.measure(task, geometry)
                current = dict(q=task._dof_pos.clone(), dq=task._dof_vel.clone(), hand=hand,
                    obj=poses(task._target_states), velocity=task._target_states[:, 7:13].clone(),
                    surface_gap=gap, support_gap=sup, table_footprint=foot)
                for key, value in current.items():
                    logs[key].append(value.cpu().numpy().copy())
                return current

            current = record()
            work_started = time.monotonic()
            for tick in range(542):
                with torch.no_grad():
                    raw = obs['obs']
                    padded = raw.repeat((64 + n - 1) // n, 1)[:64]
                    native_command = self.get_action(dict(obs, obs=padded), True)[:n].clamp(-1, 1).clone()
                    intended = native_command.clone()
                    requested_pd = pd_from_action(native_command, current['q'], offset, scale)
                    if evaluating:
                        donor = np.arange(24, 32)
                        target = torch.as_tensor(source['pd_targets'][tick, donor], device=task.device)
                        first, native_command[8:16], _ = command(target, current['q'][8:16], offset, scale)
                        intended[8:16], requested_pd[8:16] = first, target
                        index = np.minimum(tick + np.arange(1, 25), 542)
                        future = np.moveaxis(source['hand'][index][:, donor], 0, 1)
                        s, tau = features(*(current[key][16:24].cpu().numpy() for key in
                                          ('q', 'dq', 'hand', 'obj', 'velocity')), future)
                        state_tensor = torch.as_tensor(s, device=task.device)
                        tau_tensor = torch.as_tensor(tau, device=task.device)
                        encoded = model(state_tensor, tau_tensor)
                        target = decode_target(encoded, current['q'][16:24])
                        first, native_command[16:24], _ = command(target, current['q'][16:24], offset, scale)
                        intended[16:24], requested_pd[16:24] = first, target
                        for key, value in dict(state_features=s, tau_features=tau,
                                              encoded_target=encoded.cpu().numpy()).items():
                            logs[key].append(value.copy())
                    if not torch.isfinite(native_command).all():
                        raise FloatingPointError('nonfinite native command')
                    expected_pd = pd_from_action(native_command, current['q'], offset, scale)
                    obs, _, done, _ = self.env_step(self.env, native_command.clone())
                    # Archived env_step returns a tensor; reset uses the repo's
                    # dict wrapper. Normalize the boundary before the next call.
                    if not isinstance(obs, dict):
                        obs = {'obs': obs}
                    if capture is None or not torch.equal(capture, native_command):
                        raise ValueError('requested/applied action mismatch')
                    if actual_pd is None or (expected_pd - actual_pd).abs().max() > 1e-6:
                        raise ValueError('native PD capture / inverse mismatch')
                    for key, value in dict(action=native_command, applied=capture,
                        pd_targets=actual_pd, intended=intended, requested_pd=requested_pd,
                        clipped=(intended - native_command).abs().amax(-1) > 1e-6).items():
                        logs[key].append(value.cpu().numpy().copy())
                    current = record()
                    if tick < 541 and done.any():
                        raise ValueError('early termination/reset invalidates windows')
                if tick == 0 or (tick + 1) % 128 == 0:
                    state = gpu_state(args.gpu)
                    if set(state['pids']) - {os.getpid()}:
                        raise RuntimeError('foreign GPU process: ' + repr(state))
                    monitor = dict(control=tick + 1, elapsed_s=time.monotonic() - started,
                        eta_s=(time.monotonic() - work_started) / (tick + 1) * (541 - tick), gpu=state)
                    print(json.dumps(monitor), flush=True)
                    with (out / 'monitor.jsonl').open('a') as stream:
                        stream.write(json.dumps(monitor) + '\n')
            packed = {key: np.stack(value) for key, value in logs.items() if value}
            packed['goal'] = goal_np
            np.savez_compressed(out / 'trajectory.npz', **packed)
            outcomes = []
            for env, role in enumerate(roles):
                outcome = episode_outcome(dict(object_pose=packed['obj'][:, env],
                    object_velocity=packed['velocity'][:, env], surface_gap=packed['surface_gap'][:, env],
                    support_gap=packed['support_gap'][:, env], table_footprint=packed['table_footprint'][:, env]))
                outcome.update(env=env, role=role)
                outcomes.append(outcome)
            summary = {}
            for role in sorted(set(roles)):
                ids = [i for i, r in enumerate(roles) if r == role]
                rows = [outcomes[i] for i in ids]
                summary[role] = dict(episodes=len(ids), hold45=sum(r['maximum_held_frames'] >= 45 for r in rows),
                    success=sum(r['success'] for r in rows), median_held=float(np.median([r['maximum_held_frames'] for r in rows])),
                    clipping_rate=float(packed['clipped'][:, ids].mean()),
                    endpoint_position_error_m=float(np.linalg.norm(packed['obj'][-1, ids, :3, 3] - goal_np[ids, :3, 3], axis=-1).mean()))
            readiness = summary['teacher']['hold45'] >= (6 if evaluating else 24) and summary['teacher']['success'] >= (6 if evaluating else 24)
            status = 'PROMISING' if readiness else 'UNCLEAR'
            if evaluating:
                calibration = readiness and summary['source_pd']['hold45'] >= 6 and summary['source_pd']['success'] >= 6
                r = summary['gt_tau_R']
                passed = (r['hold45'] >= 6 and r['success'] >= 6 and r['clipping_rate'] < .01
                          and r['median_held'] >= .8 * summary['source_pd']['median_held'])
                status = ('PROMISING' if passed else 'UNPROMISING') if calibration else 'UNCLEAR'
            write(out / 'result.json', dict(status=status, teacher_readiness=readiness,
                summary=summary, outcomes=outcomes, audit_required=True, scope=manifest['evaluation']))
            if any(sha(path) != value for path, value in hashes.items()):
                raise ValueError('input/source drift')
            manifest.update(status='COMPLETED', trajectory_sha256=sha(out / 'trajectory.npz'),
                elapsed_s=time.monotonic() - started, torch_peak_bytes=torch.cuda.max_memory_allocated())
            write(out / 'manifest.json', manifest)
            print(json.dumps(summary), flush=True)

    native.EvalPlayer = Player
    sys.argv = [sys.argv[0], '--task', 'Dexplore_Inspire', '--cfg_env', cfg['cfg_env'],
        '--cfg_train', str(train_cfg), '--motion_file', str(motion_root), '--checkpoint', str(OFFICIAL),
        '--headless', '--num_envs', str(len(roles)), '--seed', str(args.seed),
        '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--pipeline', 'gpu',
        '--graphics_device_id', '0', '--num_threads', '1', '--disable-early-termination',
        '--output', str(out / 'unused.json'), '--output_path', str(out / 'native')]
    cwd = Path.cwd()

    def deadline(signum, frame):
        raise TimeoutError('bounded competent-PD native run exceeded 300s')

    handler = signal.signal(signal.SIGALRM, deadline)
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
        signal.signal(signal.SIGALRM, handler)
        os.chdir(cwd)


if __name__ == '__main__':
    main()
