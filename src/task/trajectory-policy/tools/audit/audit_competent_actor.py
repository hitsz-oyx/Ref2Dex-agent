"""Independent history/goal/plan/control replay of competent trajectory BC."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK / 'src'), str(ROOT / 'src/task/consequence-evaluator/src'),
               str(Path(__file__).parent)]
from audit_competent_pd import independent_features, independent_pd, ACTIVE, COUPLING


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_input(data, ticks, envs, reference=None):
    past = np.maximum(ticks[None] + np.arange(-3, 1)[:, None], 0)
    obj, hand, q, dq, vel = [data[k][past, envs[None]].swapaxes(0, 1)
                            for k in ('obj', 'hand', 'q', 'dq', 'velocity')]
    rotation, center = obj[:, -1, :3, :3], obj[:, -1, :3, 3]
    local = np.matmul(hand - center[:, None, None], rotation[:, None])
    local_obj = np.matmul(np.linalg.inv(obj[:, -1])[:, None], obj)
    fingers = local[:, :, 1:] - local[:, :, :1]
    velocity = np.concatenate((np.matmul(vel[..., :3], rotation), np.matmul(vel[..., 3:], rotation)), -1)
    wrist = np.concatenate((np.matmul(dq[..., :3], rotation), dq[..., 3:6]), -1)
    goal = data['goal'][envs]
    goal_position = np.einsum('nj,njk->nk', goal[:, :3, 3] - center, rotation)
    goal_rotation = np.matmul(rotation.swapaxes(-1, -2), goal[:, :3, :3])
    n = len(ticks)
    history = np.concatenate((local[:, :, 0].reshape(n, -1), fingers.reshape(n, -1),
        local_obj[:, :, :3].reshape(n, -1), q[..., 6:].reshape(n, -1), dq[..., 6:].reshape(n, -1),
        velocity.reshape(n, -1), wrist.reshape(n, -1), obj[:, -1, 2:3, 3], -rotation[:, 2],
        goal_position, goal_rotation.reshape(n, 9)), -1).astype(np.float32)
    if reference is None:
        return history
    columns = []
    for offset in (1, 16):
        index = np.minimum(ticks + offset, len(reference['obj']) - 1)
        position = np.matmul(reference['hand_pos'][index] - center[:, None], rotation)
        orientation = np.matmul(rotation.swapaxes(-1, -2)[:, None], reference['hand_rot'][index])
        obj_position = np.matmul((reference['obj'][index, :3, 3] - center)[:, None], rotation)[:, 0]
        obj_orientation = np.matmul(rotation.swapaxes(-1, -2), reference['obj'][index, :3, :3])
        columns.extend((position.reshape(n, 48), orientation.reshape(n, 144), obj_position, obj_orientation.reshape(n, 9)))
    return np.concatenate((history[:, :328], *columns), -1).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--execution', type=Path)
    args = p.parse_args()
    scratch = ROOT / 'tmp/trajectory-policy/competent-pd'
    os.environ.update(TMPDIR=str(scratch), MPLCONFIGDIR=str(scratch / 'matplotlib'))
    import torch
    from trajectory_policy.competent_actor import load_actor
    from trajectory_policy.pd_retargeter import load_model
    from consequence_evaluator.gate1 import episode_outcome
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    device = 'cuda:0'
    manifest = json.loads((args.fit / 'manifest.json').read_text())
    result = json.loads((args.fit / 'result.json').read_text())
    if manifest['status'] != 'COMPLETED':
        raise ValueError('completed fit required')
    source_dir = Path(manifest['source'])
    reference = None
    if manifest.get('reference'):
        from trajectory_policy.reference_actor import load_reference_actor
        load_actor = load_reference_actor
        ref_path = Path(manifest['reference'])
        rm = json.loads((ref_path.parent / 'manifest.json').read_text())
        if sha(ref_path) != rm['reference_sha256'] or sha(rm['motion']) != rm['motion_sha256']:
            raise ValueError('planned reference identity differs')
        with np.load(ref_path, allow_pickle=False) as f:
            reference = {k: f[k].copy() for k in f.files}
    with np.load(source_dir / 'trajectory.npz', allow_pickle=False) as f:
        source = {k: f[k].copy() for k in f.files}
    if sha(source_dir / 'trajectory.npz') != json.loads((source_dir / 'manifest.json').read_text())['trajectory_sha256']:
        raise ValueError('source identity mismatch')
    retargeter = Path(manifest['retargeter'])
    if sha(retargeter) != manifest['retargeter_sha256']:
        raise ValueError('R changed')
    r = load_model(torch.load(retargeter, map_location=device, weights_only=False), device)
    models = {}
    errors = {}

    def close(label, actual, expected, tolerance=2e-5):
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        errors[label] = max(error, errors.get(label, 0.))
        if error > tolerance:
            raise ValueError('independent replay differs: ' + label + ' ' + str(error))

    if reference is not None:
        from scipy.spatial.transform import Rotation
        raw = torch.load(rm['motion'], map_location='cpu', weights_only=False).numpy()
        close('reference_human_position', reference['hand_pos'], raw[:, 150:198].reshape(-1, 16, 3))
        close('reference_human_rotation', reference['hand_rot'], Rotation.from_quat(raw[:, 309:373].reshape(-1, 4)).as_matrix().reshape(-1, 16, 3, 3))
        close('reference_object_position', reference['obj'][:, :3, 3], raw[:, 198:201])
        close('reference_object_rotation', reference['obj'][:, :3, :3], Rotation.from_quat(raw[:, 201:205]).as_matrix())

    if not args.execution:
        with np.load(args.fit / 'predictions.npz', allow_pickle=False) as f:
            predictions = {k: f[k].copy() for k in f.files}
        arrays = {}
        for split, ids in manifest['groups'].items():
            ticks, envs = np.repeat(np.arange(519), len(ids)), np.tile(ids, 519)
            x = independent_input(source, ticks, envs, reference)
            future = np.stack([source['hand'][t + 1:t + 25, e] for t, e in zip(ticks, envs)])
            s, tau = independent_features(source, ticks, envs, future)
            pd = source['pd_targets'][ticks, envs][:, ACTIVE].copy()
            pd[:, :6] -= source['q'][ticks, envs, :6]
            arrays[split] = x, s, tau, pd, ticks
        for arm in ('tau_only', 'joint'):
            if sha(args.fit / (arm + '.pt')) != result['checkpoint_sha256'][arm]:
                raise ValueError('actor checkpoint differs')
            packet = torch.load(args.fit / (arm + '.pt'), map_location=device, weights_only=False)
            actor = load_actor(packet, device)
            for name, value in (('input', arrays['train'][0]), ('tau', arrays['train'][2])):
                close(arm + '_' + name + '_mean', value.astype(np.float64).mean(0), getattr(actor, name + '_mean').cpu().numpy())
                close(arm + '_' + name + '_scale', np.maximum(value.astype(np.float64).std(0), .001),
                      getattr(actor, name + '_scale').cpu().numpy())
            for split, (xx, ss, tt, yy, ticks) in arrays.items():
                tau_out, pd_out = [], []
                with torch.no_grad():
                    for i in range(0, len(xx), 512):
                        t = actor(torch.as_tensor(xx[i:i + 512], device=device))
                        pd_out.append(r(torch.as_tensor(ss[i:i + 512], device=device), t).cpu().numpy())
                        tau_out.append(t.cpu().numpy())
                tau_out, pd_out = np.concatenate(tau_out), np.concatenate(pd_out)
                close(arm + '_' + split + '_tau', tau_out, predictions[arm + '_' + split + '_tau'])
                close(arm + '_' + split + '_pd', pd_out, predictions[arm + '_' + split + '_pd'])
                close(arm + '_' + split + '_tau_mse', np.mean(((tau_out - tt) / actor.tau_scale.cpu().numpy()) ** 2), result['reports'][arm][split]['tau_mse'])
                close(arm + '_' + split + '_pd_mse', np.mean(((pd_out - yy) / r.target_scale.cpu().numpy()) ** 2), result['reports'][arm][split]['pd_mse'])
            monitor = [json.loads(row) for row in (args.fit / 'monitor.jsonl').read_text().splitlines()]
            if min(monitor, key=lambda row: row[arm]['objective'])['update'] != packet['selected_update']:
                raise ValueError('validation selection differs')
            models[arm] = actor
        for parameter in r.parameters():
            parameter.requires_grad_(False)
        x, s, _, y, _ = arrays['train']
        loss = ((r(torch.as_tensor(s[:32], device=device), models['joint'](torch.as_tensor(x[:32], device=device)))
                - torch.as_tensor(y[:32], device=device)) / r.target_scale).square().mean()
        loss.backward()
        gradient = sum(float(p.grad.abs().sum()) for p in models['joint'].parameters() if p.grad is not None)
        close('PD_actor_gradient', gradient, result['pd_actor_gradient_l1'], tolerance=.001)
        if gradient <= 0 or any(p.grad is not None for p in r.parameters()):
            raise ValueError('frozen-R gradient contract differs')
        folder = args.fit
    else:
        folder = args.execution
        em = json.loads((folder / 'manifest.json').read_text())
        er = json.loads((folder / 'result.json').read_text())
        if em['status'] != 'COMPLETED' or em['roles'] != ['teacher'] * 4 + ['gt_tau_R'] * 4 + ['tau_only'] * 8 + ['joint'] * 8:
            raise ValueError('completed fixed autonomous execution required')
        for arm in ('tau_only', 'joint'):
            if sha(args.fit / (arm + '.pt')) != result['checkpoint_sha256'][arm]:
                raise ValueError('actor identity changed')
            models[arm] = load_actor(torch.load(args.fit / (arm + '.pt'), map_location=device, weights_only=False), device)
        with np.load(folder / 'trajectory.npz', allow_pickle=False) as f:
            execution = {k: f[k].copy() for k in f.files}
        if sha(folder / 'trajectory.npz') != em['trajectory_sha256'] or not np.array_equal(execution['action'], execution['applied']):
            raise ValueError('execution identity or applied command mismatch')
        if (execution['action'].shape != (542, 24, 18)
                or any(len(execution[k]) != 543 for k in ('q', 'dq', 'hand', 'obj', 'velocity'))
                or any(not np.isfinite(v).all() for v in execution.values())):
            raise ValueError('full finite execution required')
        close('known_goal', execution['goal'], np.broadcast_to(source['goal'][0], (24, 4, 4)), 1e-6)
        close('actual_PD', independent_pd(execution['action'], execution['q'][:-1], em['native_controller']), execution['pd_targets'], 1e-6)
        if not np.array_equal(execution['clipped'], np.max(np.abs(execution['intended'] - execution['action']), axis=-1) > 1e-6):
            raise ValueError('clip flags differ')
        period = em.get('actor_period', 8)
        if period not in (1, 8) or not np.array_equal(execution['query_tick'], np.arange(0, 542, period)):
            raise ValueError('query period differs')
        plans = []
        for j, tick in enumerate(execution['query_tick']):
            ids = np.arange(8, 24)
            x = independent_input(execution, np.full(16, tick), ids, reference)
            close('history_goal_input', x, execution['high_input'][j])
            with torch.no_grad():
                prediction = np.concatenate([models[arm](torch.as_tensor(x[i * 8:(i + 1) * 8], device=device)).cpu().numpy()
                                             for i, arm in enumerate(('tau_only', 'joint'))])
            close('actor_output', prediction, execution['high_tau'][j])
            plan = np.matmul(prediction.reshape(16, 24, 11, 3), execution['obj'][tick, ids, :3, :3].swapaxes(-1, -2)[:, None])
            plan = plan + execution['hand'][tick, ids, None]
            close('query_world_plan', plan, execution['plan_world'][j])
            plans.append(plan)
        for tick in range(542):
            oracle = np.moveaxis(source['hand'][np.minimum(tick + np.arange(1, 25), 542)][:, 24:28], 0, 1)
            remaining = plans[tick // period][:, np.minimum(np.arange(24) + tick % period, 23)]
            s, tau = independent_features(execution, np.full(20, tick), np.arange(4, 24), np.concatenate((oracle, remaining)))
            close('live_R_state', s, execution['state_features'][tick])
            close('live_R_tau', tau, execution['tau_features'][tick])
            with torch.no_grad():
                prediction = r(torch.as_tensor(s, device=device), torch.as_tensor(tau, device=device)).cpu().numpy()
            close('R_prediction', prediction, execution['encoded_target'][tick])
            requested = np.zeros((20, 18), np.float32)
            requested[:, ACTIVE] = prediction
            requested[:, :6] += execution['q'][tick, 4:24, :6]
            for distal, parent, ratio in COUPLING:
                requested[:, distal] = requested[:, parent] * ratio
            if em.get('dagger_rollin'):
                intervention = execution['intervention'][tick, 4:24]
                requested[intervention] = execution['expert_pd'][tick, 4:24][intervention]
            close('requested_PD', requested, execution['requested_pd'][tick, 4:24])
        if em.get('dagger_rollin'):
            close('expert_PD_queries', independent_pd(execution['expert_action'], execution['q'][:-1], em['native_controller']), execution['expert_pd'], 1e-6)
            if execution['intervention'][:, :16].any() or not .15 <= execution['intervention'][:, 16:24].mean() <= .35:
                raise ValueError('intervention scope or rate differs')
            mask = execution['intervention']
            close('intervention_applied_action', execution['action'][mask], execution['expert_action'][mask], 1e-6)
        for env in range(24):
            outcome = episode_outcome(dict(object_pose=execution['obj'][:, env], object_velocity=execution['velocity'][:, env],
                surface_gap=execution['surface_gap'][:, env], support_gap=execution['support_gap'][:, env], table_footprint=execution['table_footprint'][:, env]))
            if any(value != er['outcomes'][env][key] for key, value in outcome.items()):
                raise ValueError('physical outcome differs')
        summary = {}
        for role in ('teacher', 'gt_tau_R', 'tau_only', 'joint'):
            ids = [i for i, name in enumerate(em['roles']) if name == role]
            rows = [er['outcomes'][i] for i in ids]
            summary[role] = dict(episodes=len(ids), hold45=sum(row['maximum_held_frames'] >= 45 for row in rows),
                success=sum(row['success'] for row in rows), median_held=float(np.median([row['maximum_held_frames'] for row in rows])),
                clipping_rate=float(execution['clipped'][:, ids].mean()),
                endpoint_position_error_m=float(np.linalg.norm(execution['obj'][-1, ids, :3, 3]
                    - execution['goal'][ids, :3, 3], axis=-1).mean()))
            for key, value in summary[role].items():
                close(role + '_' + key, value, er['summary'][role][key], 1e-6)
        calibration = all(summary[role]['hold45'] >= 3 and summary[role]['success'] >= 3 for role in ('teacher', 'gt_tau_R'))
        joint = summary['joint']
        passed = (joint['hold45'] >= 6 and joint['success'] >= 6 and joint['clipping_rate'] < .01
                  and joint['median_held'] >= .8 * summary['gt_tau_R']['median_held'])
        if reference is not None:
            for role in summary:
                ids = [i for i, name in enumerate(em['roles']) if name == role]
                pos_error = execution['obj'][:, ids, :3, 3] - reference['obj'][:, None, :3, 3]
                relative_rot = np.matmul(reference['obj'][:, None, :3, :3].swapaxes(-1, -2), execution['obj'][:, ids, :3, :3])
                angle = np.arccos(np.clip((np.trace(relative_rot, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
                summary[role].update(reference_position_rmse_m=float(np.sqrt(np.mean(np.sum(pos_error ** 2, axis=-1)))),
                                     reference_rotation_mean_rad=float(angle.mean()))
                for key in ('reference_position_rmse_m', 'reference_rotation_mean_rad'):
                    close(role + '_' + key, summary[role][key], er['summary'][role][key], 1e-6)
            passed = (passed and joint['reference_position_rmse_m'] <= 1.5 * summary['teacher']['reference_position_rmse_m']
                      and joint['reference_rotation_mean_rad'] <= 1.5 * summary['teacher']['reference_rotation_mean_rad'] + .1)
        expected_status = ('PROMISING' if passed else 'UNPROMISING') if calibration else 'UNCLEAR'
        if em.get('dagger_rollin'):
            expected_status = 'UNCLEAR'
        if er['status'] != expected_status:
            raise ValueError('frozen physical gate status differs')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        time = np.arange(543) / 30
        height = execution['obj'][..., 2, 3] - execution['obj'][0, :, 2, 3]
        held = (execution['surface_gap'] <= .01) & (height >= .03) & ~(execution['table_footprint'] & (np.abs(execution['support_gap']) <= .02))
        for role in ('teacher', 'gt_tau_R', 'tau_only', 'joint'):
            ids = [i for i, name in enumerate(em['roles']) if name == role]
            for ax, values in zip(axes, (height, execution['surface_gap'], held)):
                ax.plot(time, values[:, ids].mean(1), label=role)
        for ax, name in zip(axes, ('Object rise (m)', 'Surface gap (m)', 'Held fraction')):
            ax.set_ylabel(name)
            ax.grid(alpha=.2)
        axes[0].legend()
        axes[2].set_xlabel('Time (s)')
        fig.tight_layout()
        fig.savefig(folder / 'behavior.png', dpi=150)
        plt.close(fig)
    report = dict(**{'pass': True}, max_errors=errors, replay_device=device, audit_source_sha256=sha(__file__),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        scope='independent history/goal/trajectory/R/actual-PD replay; no formal validation')
    (folder / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
