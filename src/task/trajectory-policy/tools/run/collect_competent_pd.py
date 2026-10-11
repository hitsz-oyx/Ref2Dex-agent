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
    rows = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True).splitlines()
    process_mib = {int(row.split(',')[0]): int(row.split(',')[1]) for row in rows if row.strip()}
    return dict(utilization=utilization, used_mib=memory, pids=[int(v) for v in pids], process_mib=process_mib)


def summarize(packed, roles, reference, manifest, autonomous, evaluating, dagger_rollin=False):
    """Summarize a complete captured trace; no simulator or model inference."""
    sys.path.insert(0, str(OLD / 'src'))
    from consequence_evaluator.gate1 import episode_outcome
    goal_np = packed['goal']
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
        if reference is not None:
            planned = reference['obj'][:, None]
            pos_error = packed['obj'][:, ids][:, :, :3, 3] - planned[..., :3, 3]
            relative_rotation = np.matmul(planned[..., :3, :3].swapaxes(-1, -2), packed['obj'][:, ids][:, :, :3, :3])
            angle = np.arccos(np.clip((np.trace(relative_rotation, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
            summary[role].update(reference_position_rmse_m=float(np.sqrt(np.mean(np.sum(pos_error ** 2, axis=-1)))),
                                 reference_rotation_mean_rad=float(angle.mean()))
    teacher_gate = 3 if autonomous else (6 if evaluating else 24)
    readiness = summary['teacher']['hold45'] >= teacher_gate and summary['teacher']['success'] >= teacher_gate
    status = 'PROMISING' if readiness else 'UNCLEAR'
    if autonomous:
        gt = summary['gt_tau_R']
        calibration = readiness and gt['hold45'] >= 3 and gt['success'] >= 3
        r = summary['joint']
        passed = (r['hold45'] >= 6 and r['success'] >= 6 and r['clipping_rate'] < .01
                  and r['median_held'] >= .8 * gt['median_held'])
        if reference is not None:
            passed = (passed and r['reference_position_rmse_m'] <= 1.5 * summary['teacher']['reference_position_rmse_m']
                      and r['reference_rotation_mean_rad'] <= 1.5 * summary['teacher']['reference_rotation_mean_rad'] + .1)
        status = ('PROMISING' if passed else 'UNPROMISING') if calibration else 'UNCLEAR'
        if dagger_rollin:
            status = 'UNCLEAR'  # collection with interventions is not autonomous evaluation
    elif evaluating:
        calibration = readiness and summary['source_pd']['hold45'] >= 6 and summary['source_pd']['success'] >= 6
        r = summary['gt_tau_R']
        passed = (r['hold45'] >= 6 and r['success'] >= 6 and r['clipping_rate'] < .01
                  and r['median_held'] >= .8 * summary['source_pd']['median_held'])
        status = ('PROMISING' if passed else 'UNPROMISING') if calibration else 'UNCLEAR'
    return dict(status=status, teacher_readiness=readiness, summary=summary,
                outcomes=outcomes, audit_required=True, scope=manifest['evaluation'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--actor-fit', type=Path, help='frozen matched H/G actor fit for autonomous execution')
    parser.add_argument('--actor-period', type=int, choices=(1, 8), default=8)
    parser.add_argument('--dagger-rollin', action='store_true', help='collect teacher PD queries with25% joint-arm expert interventions')
    parser.add_argument('--idle-context-pid', type=int, action='append', default=[],
                        help='pre-existing idle <=512MiB context; never authorize other active workloads')
    parser.add_argument('--physical-probe-card', type=Path,
                        help='explicit frozen physical Decision after a failed offline precision screen')
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--seed', type=int, default=180)
    args = parser.parse_args()
    if args.actor_fit is not None:
        # native.main changes cwd before Player.run loads the high-level actors.
        args.actor_fit = args.actor_fit.resolve()
    out = args.output.resolve()
    evaluating = args.source is not None
    autonomous = args.actor_fit is not None
    reference = None
    if (out.exists() or ROOT / 'outputs/trajectory-policy' not in out.parents
            or Path(sys.prefix).resolve() != RUNTIME.resolve()
            or evaluating != (args.model is not None) or (autonomous and not evaluating)
            or not 100 <= args.seed <= 299):
        raise ValueError('fresh task output, archived runtime and source/model pair required')
    before = gpu_state(args.gpu)
    allowed_contexts = set(args.idle_context_pid)
    if (allowed_contexts and not autonomous) or (set(before['pids']) - allowed_contexts
            or before['utilization'] > (5 if allowed_contexts else 10) or before['used_mib'] > 512):
        raise RuntimeError('GPU occupied: ' + repr(before))
    context_commands = {pid: subprocess.check_output(['ps', '-p', str(pid), '-o', 'args='], text=True).strip()
                        for pid in allowed_contexts if pid in before['pids']}
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
    producer_code_updates = []
    physical_decision = None
    if evaluating:
        source_manifest = json.loads((args.source / 'manifest.json').read_text())
        if (source_manifest.get('status') != 'COMPLETED'
                or source_manifest.get('training_allowed') is not True
                or source_manifest['checkpoint_sha256'] != OFFICIAL_SHA):
            raise ValueError('fresh completed training-eligible official source required')
        for path, digest in source_manifest['input_sha256'].items():
            if sha(path) != digest:
                # A new consumer protocol may update this producer entrypoint.
                # Authenticate the actual producer from Git; data/native/env
                # inputs must still match current immutable files exactly.
                if Path(path).resolve() != Path(__file__).resolve():
                    raise ValueError('teacher input drift: ' + path)
                relative = str(Path(path).resolve().relative_to(ROOT))
                original = subprocess.check_output(['git', 'show',
                    source_manifest['git_commit'] + ':' + relative], cwd=ROOT)
                if hashlib.sha256(original).hexdigest() != digest:
                    raise ValueError('producer Git blob differs from recorded source hash')
                producer_code_updates.append(dict(path=path, producer_sha256=digest,
                    consumer_sha256=sha(path), authenticated_git_commit=source_manifest['git_commit']))
        if sha(args.source / 'trajectory.npz') != source_manifest['trajectory_sha256']:
            raise ValueError('teacher trajectory drift')
        with np.load(args.source / 'trajectory.npz', allow_pickle=False) as stream:
            source = {key: stream[key].copy() for key in ('hand', 'pd_targets', 'q', 'goal')}
        fit_result = json.loads((args.model.parent / 'result.json').read_text())
        if sha(args.model) != fit_result['checkpoint_sha256']['tau']:
            raise ValueError('exact selected model required')
        if autonomous:
            card = TASK / ('docs/experiments/probes/P-20261011-competent-actor-cadence.md'
                           if args.actor_period == 1 else 'docs/experiments/probes/P-20261011-competent-joint-bc.md')
            actor_result = json.loads((args.actor_fit / 'result.json').read_text())
            actor_manifest = json.loads((args.actor_fit / 'manifest.json').read_text())
            if actor_manifest.get('reference'):
                card = TASK / 'docs/experiments/probes/P-20261011-reference-goal-dagger.md'
                ref_path = Path(actor_manifest['reference'])
                ref_manifest = json.loads((ref_path.parent / 'manifest.json').read_text())
                if sha(ref_path) != ref_manifest['reference_sha256']:
                    raise ValueError('planned reference identity changed')
                with np.load(ref_path, allow_pickle=False) as f:
                    reference = {k: f[k].copy() for k in f.files}
                files += [ref_path, ref_path.parent / 'manifest.json', TASK / 'src/trajectory_policy/reference_actor.py']
            if args.dagger_rollin and (reference is None or args.actor_period != 1):
                raise ValueError('DAgger roll-in requires reference actors queried each control')
            if (actor_manifest['status'] != 'COMPLETED'
                    or actor_manifest['retargeter_sha256'] != sha(args.model)
                    or not json.loads((args.actor_fit / 'audit.json').read_text())['pass']
                    or not all(actor_result['physical_screen_pass'].values())):
                raise ValueError('audited screened frozen H/G fit required')
            for arm in ('tau_only', 'joint'):
                if sha(args.actor_fit / (arm + '.pt')) != actor_result['checkpoint_sha256'][arm]:
                    raise ValueError('actor checkpoint drift')
            physical_decision = str(card)
            files += [card, TASK / 'src/trajectory_policy/competent_actor.py',
                TASK / 'src/trajectory_policy/history.py', Path(__file__).with_name('competent_actor_execution.py'),
                *[args.actor_fit / name for name in ('tau_only.pt', 'joint.pt', 'manifest.json', 'result.json', 'audit.json')]]
        if args.physical_probe_card is not None:
            card = args.physical_probe_card.resolve()
            expected = TASK / 'docs/experiments/probes/P-20261011-competent-pd-physical.md'
            if card != expected or 'model_sha256: ' + sha(args.model) not in card.read_text():
                raise ValueError('exact frozen physical Decision card/model required')
            if not json.loads((args.model.parent / 'audit.json').read_text()).get('pass'):
                raise ValueError('independent fit audit required for physical Decision')
            physical_decision = str(card)
            files.append(card)
        if not fit_result.get('physical_screen_pass') and physical_decision is None:
            raise ValueError('offline precision screen or separately specified physical Decision required')
        packet = torch.load(args.model, map_location='cuda:0')
        model = load_model(packet, 'cuda:0')
        files += [args.source / 'manifest.json', args.source / 'trajectory.npz',
                  args.model, args.model.parent / 'manifest.json', args.model.parent / 'result.json']
    hashes = {str(path.resolve()): sha(path) for path in files}
    roles = (['teacher'] * 4 + ['gt_tau_R'] * 4 + ['tau_only'] * 8 + ['joint'] * 8 if autonomous else
             ['teacher'] * 8 + ['source_pd'] * 8 + ['gt_tau_R'] * 8
             if evaluating else ['teacher'] * 32)
    out.mkdir(parents=True)
    manifest = dict(schema='ref2dex.competent-pd-rollout.v1', status='RUNNING',
        task='trajectory-policy', experiment_id=('P-20261011-competent-pd-physical'
            if physical_decision else 'P-20261011-competent-pd-retargeter'),
        run_id=out.name, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=ROOT, text=True).strip(), physical_gpu=args.gpu, seed=args.seed, roles=roles,
        controls=542, dt=1 / 30, checkpoint=str(OFFICIAL), checkpoint_sha256=OFFICIAL_SHA,
        training_allowed=not evaluating, source_origin='official DExplore demonstration',
        input_sha256=hashes, gpu_before=before,
        preexisting_idle_contexts=context_commands, idle_context_max_mib=512,
        evaluation='full-reference oracle R mechanism, not independent high-level policy',
        inference='current measured state87 + oracle future actual hand24 -> actual PD target',
        teacher_inference_rows=64, source_envs=list(range(24, 32)) if evaluating else None,
        producer_code_updates=producer_code_updates, physical_decision_card=physical_decision,
        original_offline_screen_pass=fit_result['physical_screen_pass'] if evaluating else None,
        target_alignment='q[t],hand[t] -> PD[t] -> hand[t+1]; no future measured q/object labels as input')
    write(out / 'manifest.json', manifest)
    if autonomous:
        manifest.update(experiment_id=('P-20261011-competent-actor-cadence' if args.actor_period == 1
                                       else 'P-20261011-competent-joint-bc'), actor_period=args.actor_period,
            evaluation='autonomous measured H/G trajectory -> frozen R full physical execution',
            inference='H328 + goal12 queried every%d; R uses live measured state every control' % args.actor_period,
            actor_fit=str(args.actor_fit.resolve()), retargeter=str(args.model.resolve()),
            source_envs=list(range(24, 28)))
        write(out / 'manifest.json', manifest)
        if reference is not None:
            manifest.update(experiment_id='P-20261011-reference-goal-dagger', reference=actor_manifest['reference'],
                reference_pose_anchors=[1, 16], inference='H328 + authorized reference human/object anchors408 -> trajectory -> frozen R',
                dagger_rollin=args.dagger_rollin, training_allowed=args.dagger_rollin,
                training_labels='expert native PD queries only; mixed future hand is not expert tau',
                intervention_beta=.25 if args.dagger_rollin else 0.,
                continuation_arms=actor_manifest.get('continuation_arms'))
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
            if reference is not None:
                from trajectory_policy.reference_actor import reference_from_processed
                measured_reference = reference_from_processed(task.hoi_data[task.data_id[0]].cpu().numpy())
                ref_error = {k: float(np.max(np.abs(measured_reference[k] - reference[k]))) for k in reference}
                if max(ref_error.values()) > 2e-5:
                    raise ValueError('exported reference disagrees with native processed reference')
                manifest['native_reference_max_errors'] = ref_error
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
            if args.dagger_rollin:
                logs.update(expert_action=[], expert_pd=[], intervention=[])

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
            actor_execution = None
            if autonomous:
                from competent_actor_execution import ActorExecution
                actor_execution = ActorExecution(args.actor_fit, task.device, args.actor_period)
            work_started = time.monotonic()
            for tick in range(542):
                with torch.no_grad():
                    raw = obs['obs']
                    padded = raw.repeat((64 + n - 1) // n, 1)[:64]
                    native_command = self.get_action(dict(obs, obs=padded), True)[:n].clamp(-1, 1).clone()
                    intended = native_command.clone()
                    requested_pd = pd_from_action(native_command, current['q'], offset, scale)
                    teacher_command = native_command.clone()
                    teacher_target = requested_pd.clone()
                    if args.dagger_rollin:
                        native_teacher_target = original_pd(teacher_command.clone())
                        if (teacher_target - native_teacher_target).abs().max() > 1e-6:
                            raise ValueError('teacher-query native PD target mismatch')
                    if evaluating:
                        donor = np.arange(24, 28) if autonomous else np.arange(24, 32)
                        low_ids = slice(4, 24) if autonomous else slice(16, 24)
                        if not autonomous:
                            target = torch.as_tensor(source['pd_targets'][tick, donor], device=task.device)
                            first, native_command[8:16], _ = command(target, current['q'][8:16], offset, scale)
                            intended[8:16], requested_pd[8:16] = first, target
                        index = np.minimum(tick + np.arange(1, 25), 542)
                        future = np.moveaxis(source['hand'][index][:, donor], 0, 1)
                        if autonomous:
                            future = np.concatenate((future, actor_execution.future(tick, logs, goal_np)))
                        s, tau = features(*(current[key][low_ids].cpu().numpy() for key in
                                          ('q', 'dq', 'hand', 'obj', 'velocity')), future)
                        state_tensor = torch.as_tensor(s, device=task.device)
                        tau_tensor = torch.as_tensor(tau, device=task.device)
                        encoded = model(state_tensor, tau_tensor)
                        target = decode_target(encoded, current['q'][low_ids])
                        first, native_command[low_ids], _ = command(target, current['q'][low_ids], offset, scale)
                        intended[low_ids], requested_pd[low_ids] = first, target
                        for key, value in dict(state_features=s, tau_features=tau,
                                              encoded_target=encoded.cpu().numpy()).items():
                            logs[key].append(value.copy())
                    if args.dagger_rollin:
                        intervention = torch.zeros(n, dtype=torch.bool, device=task.device)
                        intervention[16:24] = torch.rand(8, device=task.device) < .25
                        native_command[intervention] = teacher_command[intervention]
                        intended[intervention] = teacher_command[intervention]
                        requested_pd[intervention] = teacher_target[intervention]
                        for key, value in dict(expert_action=teacher_command, expert_pd=teacher_target,
                                               intervention=intervention).items():
                            logs[key].append(value.cpu().numpy().copy())
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
                    foreign = set(state['pids']) - {os.getpid()}
                    if (foreign - allowed_contexts or
                            any(state['process_mib'][pid] > 512 for pid in foreign)):
                        # Preserve incomplete measurements without labeling them
                        # a completed physical experiment or a training source.
                        partial = {key: np.stack(value) for key, value in logs.items() if value}
                        partial['goal'] = goal_np
                        if autonomous:
                            partial.update({key: np.stack(value) for key, value in actor_execution.records.items()})
                        np.savez_compressed(out / 'partial.npz', **partial)
                        manifest['partial_controls'] = tick + 1
                        raise RuntimeError('foreign GPU process: ' + repr(state))
                    monitor = dict(control=tick + 1, elapsed_s=time.monotonic() - started,
                        eta_s=(time.monotonic() - work_started) / (tick + 1) * (541 - tick), gpu=state)
                    print(json.dumps(monitor), flush=True)
                    with (out / 'monitor.jsonl').open('a') as stream:
                        stream.write(json.dumps(monitor) + '\n')
            packed = {key: np.stack(value) for key, value in logs.items() if value}
            if autonomous:
                packed.update({key: np.stack(value) for key, value in actor_execution.records.items()})
            packed['goal'] = goal_np
            np.savez_compressed(out / 'trajectory.npz', **packed)
            result = summarize(packed, roles, reference, manifest, autonomous, evaluating, args.dagger_rollin)
            write(out / 'result.json', result)
            summary = result['summary']
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
