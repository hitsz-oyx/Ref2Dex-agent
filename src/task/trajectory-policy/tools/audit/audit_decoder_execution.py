"""Independent NumPy replay of decoder labels, inputs, commands and outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(ROOT / 'src/task/consequence-evaluator/src'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or ROOT / 'outputs/trajectory-policy' not in args.output.parents:
        raise ValueError('fresh task-owned audit required')
    manifest = json.loads((args.evaluation / 'manifest.json').read_text())
    result = json.loads((args.evaluation / 'result.json').read_text())
    if manifest['status'] != 'COMPLETED':
        raise ValueError('completed native wave required')
    for path, digest in manifest['input_sha256'].items():
        if sha(path) != digest:
            raise ValueError('source/input drift: ' + path)
    def load(name):
        with np.load(args.evaluation / name, allow_pickle=False) as stream:
            return {key: stream[key].copy() for key in stream.files}
    data, plans = load('trajectory.npz'), load('plans.npz')
    if data['action'].shape != (542, 16, 18) or plans['tick'].tolist() != list(range(0, 542, 8)):
        raise ValueError('complete execution and all replans required')
    if not all(np.isfinite(value).all() for value in list(data.values()) + list(plans.values())):
        raise ValueError('nonfinite saved arrays')
    reference_path = [Path(path) for path in manifest['input_sha256'] if path.endswith('.pkl')]
    geometry_path = [Path(path) for path in manifest['input_sha256'] if path.endswith('/geometry.npz')]
    if len(reference_path) != 1 or len(geometry_path) != 1:
        raise ValueError('unique frozen hand reference and geometry required')
    with reference_path[0].open('rb') as stream:
        packet = pickle.load(stream)
    # Only initial measured state and future hand are extracted from the packet.
    for saved, source in (('q', 'dof_position'), ('dq', 'dof_velocity'), ('hand', 'hand_keypoints'), ('obj', 'object_pose')):
        if not np.array_equal(data[saved][0], np.broadcast_to(packet[source][0, 0], data[saved][0].shape)):
            raise ValueError('initial state mismatch: ' + saved)
    reference_hand = packet['hand_keypoints'][:, 0].copy()
    del packet
    with np.load(geometry_path[0], allow_pickle=False) as stream:
        geometry_q = stream['q'].copy()
        geometry_hand = stream['fitted_points'].copy()
    roles = np.asarray(manifest['roles'])
    if set(roles) != {'tau_gt', 'dense_fk', 'knots48', 'knots48_repeat'} or any(sum(roles == r) != 4 for r in set(roles)):
        raise ValueError('predeclared roles differ')
    gt, dense = roles == 'tau_gt', roles == 'dense_fk'
    compressed = ~(gt | dense)
    if not np.array_equal(plans['q'][:, :, 0], data['q'][plans['tick']]) or not np.array_equal(plans['query_obj'], data['obj'][plans['tick']]):
        raise ValueError('plan is not anchored in actual current state')
    dt = manifest['dt']
    def velocity(q):
        delta = np.diff(q, axis=-2).copy()
        delta[..., 3:6] = np.arctan2(np.sin(delta[..., 3:6]), np.cos(delta[..., 3:6]))
        return np.concatenate((delta[..., :1, :], (delta[..., :-1, :] + delta[..., 1:, :])/2,
                               delta[..., -1:, :]), axis=-2)/dt
    errors = dict(knots_translation=0., knots_rotation_matrix=0., knots_fingers=0.,
                  dense_q=0., dense_fk_hand=0., gt_hand=0., velocity=0.)
    global_v = velocity(geometry_q)
    for j, tick in enumerate(plans['tick']):
        indices = np.minimum(tick + np.arange(1, 25), 542)
        target = geometry_q[indices]
        q = plans['q'][j]
        def error(key, actual, expected):
            errors[key] = max(errors[key], float(np.max(np.abs(actual-expected))))
        error('dense_q', q[gt | dense, 1:], target[None])
        error('dense_fk_hand', plans['hand'][j, dense], geometry_hand[indices][None])
        error('gt_hand', plans['hand'][j, gt], reference_hand[indices][None])
        nodes = q[compressed][:, [1, 8, 16, 24]]
        labels = target[[0, 7, 15, 23]]
        error('knots_translation', nodes[:, :, :3], labels[None, :, :3])
        a = Rotation.from_euler('XYZ', nodes[:, :, 3:6].reshape(-1, 3)).as_matrix()
        b = Rotation.from_euler('XYZ', np.broadcast_to(labels[None, :, 3:6], nodes[:, :, 3:6].shape).reshape(-1, 3)).as_matrix()
        error('knots_rotation_matrix', a, b)
        error('knots_fingers', nodes[:, :, [6, 8, 10, 12, 14, 15]], labels[None, :, [6, 8, 10, 12, 14, 15]])
        expected_v = np.zeros_like(q)
        expected_v[:, 1:] = velocity(q[:, 1:])
        expected_v[gt, 1:] = global_v[indices]
        error('velocity', plans['velocity'][j], expected_v)
    if max(errors.values()) > 2e-5:
        raise ValueError('geometry/velocity label mismatch: ' + repr(errors))
    future = np.stack([plans['hand'][t//8][:, np.minimum(np.arange(24)+t%8, 23)] for t in range(542)])
    for tick in range(542):
        future[tick, gt] = reference_hand[np.minimum(tick+np.arange(1, 25), 542)]
    next_q = np.stack([plans['q'][t//8, :, t%8+1] for t in range(542)])
    next_v = np.stack([plans['velocity'][t//8, :, t%8+1] for t in range(542)])
    rot = data['obj'][:-1, :, :3, :3]
    center = data['obj'][:-1, :, :3, 3]
    hand = (data['hand'][:-1]-center[..., None, :]) @ rot
    local_future = (future-center[..., None, None, :]) @ rot[..., None, :, :]
    qe = next_q-data['q'][:-1]
    qe[..., 3:6] = np.arctan2(np.sin(qe[..., 3:6]), np.cos(qe[..., 3:6]))
    previous = np.concatenate((np.zeros((1, 16, 12), np.float32), np.tanh(data['latent'][:-1])))
    features = np.concatenate((data['q'][:-1], data['dq'][:-1]/8, hand.reshape(542, 16, 33)/.1,
        local_future.reshape(542, 16, 792)/.1, qe, previous, data['velocity'][:-1]/2), -1).clip(-20, 20)
    errors['features'] = float(np.max(np.abs(features-data['features'])))
    control = manifest['native_controller']
    offset, scale = np.asarray(control['offset'], np.float32), np.asarray(control['scale'], np.float32)
    target = next_q.copy()
    target[..., :6] += next_v[..., :6]*np.asarray(control['gain_ratio'], np.float32)
    active = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15]
    target[..., active] += np.tanh(data['latent'])*np.asarray([.06, .06, .06, .6, .6, .6, .5, .5, .5, .5, .35, .35], np.float32)
    intended = (target-offset)/scale
    intended[..., :6] = (target[..., :6]-data['q'][:-1, :, :6]-offset[:6])/scale[:6]
    intended[..., 6:] = intended[..., 6:]*2-1
    intended[..., [7, 9, 11, 13, 16, 17]] = 0
    errors['intended'] = float(np.max(np.abs(intended-data['intended'])))
    errors['command'] = float(np.max(np.abs(intended.clip(-1, 1)-data['action'])))
    if not np.array_equal(data['action'], data['applied']):
        raise ValueError('requested/applied command mismatch')
    pd = data['action'].copy()
    pd[..., 6:] = (pd[..., 6:]+1)/2
    pd = offset+scale*pd
    pd[..., :6] += data['q'][:-1, :, :6]
    for distal, parent, ratio in ((7, 6, 1.05), (9, 8, 1.05), (11, 10, 1.05), (13, 12, 1.05), (16, 15, .6), (17, 15, .8)):
        pd[..., distal] = pd[..., parent]*ratio
    errors['pd'] = float(np.max(np.abs(pd-data['pd_targets'])))
    if errors['features'] > 2e-5 or errors['intended'] > 2e-5 or errors['command'] > 2e-6 or errors['pd'] > 1e-6:
        raise ValueError('execution chain differs: ' + repr(errors))
    from consequence_evaluator.gate1 import episode_outcome
    height = data['obj'][:, :, 2, 3]-data['obj'][0, :, 2, 3]
    supported = data['table_footprint'] & (abs(data['support_gap']) <= .02)
    held = (data['surface_gap'] <= .01) & ~supported & (height >= .03)
    held[0] = False
    summary = {}
    for row in result['outcomes']:
        env = row['env']
        outcome = episode_outcome(dict(object_pose=data['obj'][:, env], object_velocity=data['velocity'][:, env],
            **{key: data[key][:, env] for key in ('surface_gap', 'support_gap', 'table_footprint')}))
        if any(outcome[key] != row[key] for key in outcome) or bool(held[-1, env]) != row['terminal_held']:
            raise ValueError('outcome differs')
    for role in sorted(set(roles)):
        rows = [row for row in result['outcomes'] if row['role'] == role]
        summary[role] = dict(episodes=4, long_held_terminal=int(sum(row['maximum_held_frames'] >= 433
            and held[-1, row['env']] for row in rows)), terminal=int(held[-1, roles == role].sum()),
            median_held=float(np.median([row['maximum_held_frames'] for row in rows])),
            clipping_rate=float(data['clipped'][:, roles == role].mean()))
    if summary != result['summary']:
        raise ValueError('summary differs')
    calibrated = summary['tau_gt']['long_held_terminal'] >= 3 and summary['dense_fk']['long_held_terminal'] >= 3
    passes = [summary[role]['long_held_terminal'] >= 3 and summary[role]['clipping_rate'] < .01
              for role in ('knots48', 'knots48_repeat')]
    status = 'PROMISING' if calibrated and all(passes) else ('UNPROMISING' if calibrated and not any(passes) else 'UNCLEAR')
    if status != result['status']:
        raise ValueError('screen differs')
    audit = dict(status=status, calibration_pass=calibrated, summary=summary, maximum_errors=errors,
        audited='Initial states, all68plans, hand-only oracle knots, future-only velocity, actual897inputs, nativecommands/PD/outcomes',
        claim=manifest['claim'])
    args.output.mkdir(parents=True)
    (args.output/'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    files = [args.evaluation/name for name in ('manifest.json', 'result.json', 'trajectory.npz', 'plans.npz')] + [Path(__file__)]
    (args.output/'manifest.json').write_text(json.dumps(dict(status='COMPLETED',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        input_sha256={str(path.resolve()): sha(path) for path in files}), indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for role in ('tau_gt', 'dense_fk', 'knots48', 'knots48_repeat'):
        mask = roles == role
        axes[0].plot(np.median(height[:, mask], 1), label=role)
        axes[1].plot(held[:, mask].mean(1), label=role)
    axes[0].set_ylabel('Object lift (m)')
    axes[0].legend(ncol=4)
    axes[1].set_ylabel('Held fraction')
    axes[1].set_xlabel('Control tick')
    fig.tight_layout()
    fig.savefig(args.output/'behavior.png', dpi=160)
    plt.close(fig)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
