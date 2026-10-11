"""Independent frozen warm-start/PPO policy and native execution replay."""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src'), str(ROOT/'src/task/consequence-evaluator/src')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
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
    n = manifest['envs']
    paired = manifest.get('action_mode', 'mean') == 'paired_gaussian'
    if n != (64 if paired else 16) or data['action'].shape != (542, n, 18) or plans['tick'].tolist() != list(range(0, 542, 8)):
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
    learned_roles = ('warm_start', 'ppo')
    expected_counts = dict(tau_gt=4, dense_fk=4, warm_start=28 if paired else 4, ppo=28 if paired else 4)
    if set(roles) != set(expected_counts) or any(sum(roles == role) != expected_counts[role] for role in set(roles)):
        raise ValueError('predeclared roles differ')
    if paired:
        if sha(args.evaluation/'noise.npy') != manifest['noise_sha256']:
            raise ValueError('frozen noise bank drift')
        noise_bank = np.load(args.evaluation/'noise.npy', allow_pickle=False)
        expected_noise = np.random.default_rng(295).standard_normal((68, 28, 288)).astype(np.float32)
        pair_ids = np.asarray(manifest['pair_ids'])
        if manifest['noise_seed'] != 295 or not np.array_equal(noise_bank, expected_noise):
            raise ValueError('noise bank differs from predeclared seed')
        for role in learned_roles:
            if pair_ids[roles == role].tolist() != list(range(28)):
                raise ValueError('pair assignment differs')
        if not np.all(pair_ids[np.isin(roles, ('tau_gt', 'dense_fk'))] == -1):
            raise ValueError('calibration rows have paired noise')
    gt, dense = roles == 'tau_gt', roles == 'dense_fk'
    compressed = ~(gt | dense)
    gpu_before = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    util, memory = map(int, gpu_before.split(','))
    if util > 10 or memory > 512:
        raise ValueError('GPU not idle: '+gpu_before)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from trajectory_policy.actor import load_actor
    from consequence_evaluator.reset_kinematics import ResetKinematics
    from consequence_evaluator.reference_motion import NATIVE_DOF_NAMES
    from consequence_evaluator.contracts import HAND_LINKS
    from consequence_evaluator.tau_tracking import FINGER_LIMITS
    torch.set_num_threads(2)
    if manifest['matmul_precision'] != 'highest':
        raise ValueError('declared PPO precision differs')
    torch.set_float32_matmul_precision('highest')
    actor_paths = [Path(path) for path in manifest['input_sha256'] if path.endswith('/best.pt')]
    urdf_paths = [Path(path) for path in manifest['input_sha256'] if path.endswith('inspire_hand_right.urdf')]
    if len(actor_paths) != 1 or len(urdf_paths) != 1:
        raise ValueError('unique frozen actor/URDF required')
    actor = load_actor(torch.load(actor_paths[0], map_location='cuda', weights_only=False), 'cuda').eval()
    ppo_paths = [Path(path) for path in manifest['input_sha256'] if path.endswith('/final.pt') and '/trajectory-ppo-' in path]
    if len(ppo_paths) != 1:
        raise ValueError('unique final PPO actor required')
    ppo_actor = load_actor(torch.load(ppo_paths[0], map_location='cuda', weights_only=False), 'cuda').eval()
    fk = ResetKinematics(urdf_paths[0], NATIVE_DOF_NAMES, HAND_LINKS, torch.device('cuda'))
    def history(tick):
        # Independently reconstruct the documented H328 from only measured t-3:t.
        past = np.maximum(tick+np.arange(-3, 1), 0)
        values = []
        for row in np.flatnonzero(compressed):
            obj, hand, q, dq, vel = (data[key][past, row] for key in ('obj', 'hand', 'q', 'dq', 'velocity'))
            rotation, translation = obj[-1, :3, :3], obj[-1, :3, 3]
            local_hand = (hand-translation) @ rotation
            local_obj = np.linalg.inv(obj[-1])[None] @ obj
            fingers = local_hand[:, 1:]-local_hand[:, :1]
            local_vel = np.concatenate((vel[:, :3] @ rotation, vel[:, 3:] @ rotation), -1)
            wrist_rate = np.concatenate((dq[:, :3] @ rotation, dq[:, 3:6]), -1)
            values.append(np.concatenate((local_hand[:, 0].reshape(-1), fingers.reshape(-1),
                local_obj[:, :3].reshape(-1), q[:, 6:].reshape(-1), dq[:, 6:].reshape(-1),
                local_vel.reshape(-1), wrist_rate.reshape(-1), obj[-1, 2:3, 3], -rotation[2])))
        return np.asarray(values, np.float32)
    if not np.array_equal(plans['q'][:, :, 0], data['q'][plans['tick']]) or not np.array_equal(plans['query_obj'], data['obj'][plans['tick']]):
        raise ValueError('plan is not anchored in actual current state')
    dt = manifest['dt']
    def velocity(q):
        delta = np.diff(q, axis=-2).copy()
        delta[..., 3:6] = np.arctan2(np.sin(delta[..., 3:6]), np.cos(delta[..., 3:6]))
        return np.concatenate((delta[..., :1, :], (delta[..., :-1, :] + delta[..., 1:, :])/2,
                               delta[..., -1:, :]), axis=-2)/dt
    errors = dict(dense_q=0., dense_fk_hand=0., gt_hand=0., velocity=0.)
    global_v = velocity(geometry_q)
    for j, tick in enumerate(plans['tick']):
        indices = np.minimum(tick + np.arange(1, 25), 542)
        target = geometry_q[indices]
        q = plans['q'][j]
        def error(key, actual, expected):
            errors[key] = max(errors.get(key, 0.), float(np.max(np.abs(actual-expected))))
        error('dense_q', q[gt | dense, 1:], target[None])
        error('dense_fk_hand', plans['hand'][j, dense], geometry_hand[indices][None])
        error('gt_hand', plans['hand'][j, gt], reference_hand[indices][None])
        measured_h = history(int(tick))
        error('actor_history', measured_h, plans['history'][j, compressed])
        with torch.no_grad():
            predicted = np.zeros((sum(compressed), 288), np.float32)
            for role, model in (('warm_start', actor), ('ppo', ppo_actor)):
                selected = roles[compressed] == role
                distribution = model.distribution(torch.as_tensor(measured_h[selected], device='cuda'))
                if paired:
                    rows = np.flatnonzero(compressed)[selected]
                    eps = noise_bank[j, pair_ids[rows]]
                    error('sample_mean', distribution.loc.cpu().numpy(), plans['mean'][j, rows])
                    error('sample_std', distribution.scale.cpu().numpy(), plans['std'][j, rows])
                    error('paired_noise', eps, plans['epsilon'][j, rows])
                    predicted[selected] = (distribution.loc+distribution.scale*torch.as_tensor(eps, device='cuda')).cpu().numpy()
                else:
                    predicted[selected] = distribution.loc.cpu().numpy()
        error('actor_c', predicted, plans['c'][j, compressed])
        value = plans['c'][j, compressed].reshape(-1, 24, 12)*np.asarray([.01]*3+[.1]*9, np.float32)
        rot = plans['query_obj'][j, compressed, :3, :3]
        cur = q[compressed, 0]
        xyz = cur[:, None, :3]+np.einsum('nij,nkj->nki', rot, value[:, :, :3].clip(-1, 1))
        relative = Rotation.from_rotvec(value[:, :, 3:6].reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
        current_rot = Rotation.from_euler('XYZ', cur[:, 3:6]).as_matrix()
        matrices = rot[:, None] @ relative @ rot[:, None].transpose(0, 1, 3, 2) @ current_rot[:, None]
        rebuilt = np.zeros((len(cur), 25, 18), np.float32)
        rebuilt[:, 1:, :3] = xyz
        rebuilt[:, 1:, [6, 8, 10, 12, 14, 15]] = value[:, :, 6:].clip(0, np.asarray(FINGER_LIMITS))
        for row in range(len(cur)):
            prior = cur[row, 3:6]
            for k, matrix in enumerate(matrices[row]):
                angles = Rotation.from_matrix(matrix).as_euler('XYZ')
                alternative = np.asarray([angles[0]+np.pi, np.pi-angles[1], angles[2]+np.pi])
                options = np.stack((angles, alternative))
                options += 2*np.pi*np.round((prior-options)/(2*np.pi))
                chosen = options[np.sum((options-prior)**2, -1).argmin()]
                rebuilt[row, k+1, 3:6] = chosen
                prior = rebuilt[row, k+1, 3:6]
        for distal, parent, ratio in ((7, 6, 1.05), (9, 8, 1.05), (11, 10, 1.05), (13, 12, 1.05), (16, 15, .6), (17, 15, .8)):
            rebuilt[:, :, distal] = rebuilt[:, :, parent]*ratio
        rebuilt[:, 0] = cur
        error('decoded_native_q', rebuilt, q[compressed])
        with torch.no_grad():
            tensor = torch.as_tensor(rebuilt[:, 1:].reshape(-1, 18), device='cuda')
            root = torch.zeros(len(tensor), 13, device='cuda')
            root[:, 6] = 1
            decoded_hand = fk.positions(tensor, root).reshape(-1, 24, 11, 3).cpu().numpy()
        error('decoded_hand', decoded_hand, plans['hand'][j, compressed])
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
    previous = np.concatenate((np.zeros((1, n, 12), np.float32), np.tanh(data['latent'][:-1])))
    features = np.concatenate((data['q'][:-1], data['dq'][:-1]/8, hand.reshape(542, n, 33)/.1,
        local_future.reshape(542, n, 792)/.1, qe, previous, data['velocity'][:-1]/2), -1).clip(-20, 20)
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
        summary[role] = dict(episodes=len(rows), long_held_terminal=int(sum(row['maximum_held_frames'] >= 433
            and held[-1, row['env']] for row in rows)), terminal=int(held[-1, roles == role].sum()),
            median_held=float(np.median([row['maximum_held_frames'] for row in rows])),
            clipping_rate=float(data['clipped'][:, roles == role].mean()))
    if summary != result['summary']:
        raise ValueError('summary differs')
    calibrated = summary['tau_gt']['long_held_terminal'] >= 3 and summary['dense_fk']['long_held_terminal'] >= 3
    ppo_success = summary['ppo']['long_held_terminal'] >= (21 if paired else 3) and summary['ppo']['clipping_rate'] < .01
    improvement = summary['ppo']['long_held_terminal'] - summary['warm_start']['long_held_terminal']
    status = 'PROMISING' if calibrated and ppo_success and improvement >= (7 if paired else 2) else (
        'UNPROMISING' if calibrated and all(summary[role]['long_held_terminal'] == 0 for role in learned_roles) else 'UNCLEAR')
    baseline_signal = 'PROMISING' if calibrated and ppo_success else 'UNCLEAR'
    if baseline_signal != result['baseline_signal']:
        raise ValueError('baseline signal differs')
    if status != result['status']:
        raise ValueError('screen differs')
    audit = dict(status=status, calibration_pass=calibrated, summary=summary, maximum_errors=errors,
        actor_replay_matmul_precision=torch.get_float32_matmul_precision(),
        audited='Initial states, all68 actor H from measured t-3:t, actual actor c, independent dense physical decode/native Euler/coupling/FK, control labels, future-only velocity, actual897inputs, nativecommands/PD/outcomes',
        claim=manifest['claim'])
    args.output.mkdir(parents=True)
    (args.output/'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    files = [args.evaluation/name for name in ('manifest.json', 'result.json', 'trajectory.npz', 'plans.npz')] + [Path(__file__)]
    if paired:
        files.append(args.evaluation/'noise.npy')
    (args.output/'manifest.json').write_text(json.dumps(dict(status='COMPLETED',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        physical_gpu=args.gpu, gpu_before=gpu_before, input_sha256={str(path.resolve()): sha(path) for path in files}), indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for role in ('tau_gt', 'dense_fk')+learned_roles:
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
    def deadline(signum, frame):
        raise TimeoutError('PPO execution audit exceeded60s')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(60)
    main()
