"""Independent NumPy command/clock replay and physical tracking summaries."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ACTIVE = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15]
COUPLED = [(7, 6, 1.05), (9, 8, 1.05), (11, 10, 1.05),
           (13, 12, 1.05), (16, 15, .6), (17, 15, .8)]


def couple(value):
    value = value.copy()
    for distal, parent, ratio in COUPLED:
        value[..., distal] = value[..., parent] * ratio
    return value


def nearest_euler(rotation, live):
    """Match the runtime's nearest equivalent XYZ Euler branch."""
    first = rotation.as_euler('XYZ').reshape(live.shape)
    second = np.stack((first[..., 0] + np.pi, np.pi-first[..., 1], first[..., 2] + np.pi), -1)
    choices = np.stack((first, second), -2)
    choices += 2*np.pi*np.round((live[..., None, :] - choices)/(2*np.pi))
    pick = np.argmin(np.square(choices-live[..., None, :]).sum(-1), axis=-1)
    result = np.take_along_axis(choices, pick[..., None, None], axis=-2)[..., 0, :]
    singular = np.abs(np.cos(first[..., 1])) < 1e-7
    sign = np.where(np.sin(first[..., 1]) >= 0, 1., -1.)
    phase = (first[..., 0]-live[..., 0]-sign*live[..., 2]+np.pi) % (2*np.pi)-np.pi
    result[..., 0] = np.where(singular, live[..., 0]+phase/2, result[..., 0])
    result[..., 2] = np.where(singular, live[..., 2]+sign*phase/2, result[..., 2])
    return result


def statistics(value):
    value = np.asarray(value).reshape(-1)
    return dict(mean=float(value.mean()), p95=float(np.percentile(value, 95)), maximum=float(value.max()))


def held(data, env, begin=0, end=None):
    """Same measured proximity/support/height held proxy, no placement claim."""
    end = len(data['q']) if end is None else end
    height = data['obj'][:, env, 2, 3] - data['obj'][0, env, 2, 3]
    supported = data['table_footprint'][:, env] & (np.abs(data['support_gap'][:, env]) <= .02)
    flag = (data['surface_gap'][:, env] <= .01) & ~supported & (height >= .03)
    flag[0] = False
    run = longest = 0
    for valid in flag[begin:end]:
        run = run + 1 if valid else 0
        longest = max(longest, run)
    return dict(maximum_held_frames=longest, held_fraction=float(flag[begin:end].mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--q-source', choices=('realized', 'reference'), default='realized')
    parser.add_argument('--reference-branch', choices=('raw', 'nearest-live'), default='raw')
    args = parser.parse_args()
    destination = args.run / 'independent_audit.json'
    if destination.exists():
        raise FileExistsError(destination)
    manifest = json.loads((args.run / 'manifest.json').read_text())
    if manifest['status'] != 'COMPLETED':
        raise ValueError('completed physics required')
    if manifest.get('q_source', 'realized') != args.q_source:
        raise ValueError('audit q-source disagrees with replay manifest')
    expected_branch = args.reference_branch if args.q_source == 'reference' else None
    if manifest.get('reference_branch') != expected_branch:
        raise ValueError('audit reference branch disagrees with replay manifest')
    with np.load(args.run / 'trajectory.npz') as stream:
        data = {key: stream[key].copy() for key in stream.files}
    source = Path(manifest['source']) / manifest['record']['path']
    with np.load(source / 'trajectory.npz') as stream:
        original = {key: stream[key].copy() for key in stream.files}
    if args.q_source == 'reference':
        import torch
        reference_raw = torch.load(manifest['reference_q_source'], map_location='cpu', weights_only=False).numpy()
        if reference_raw.ndim != 2 or reference_raw.shape[1] < 391 or reference_raw.shape[0] != len(original['q']):
            raise ValueError('geometric reference motion length/schema does not match RL trajectory')
        original['reference_q'] = reference_raw[:, 373:391].copy()
        if not np.isfinite(original['reference_q']).all():
            raise ValueError('reference q contains non-finite values')
    count = manifest['controls']; n = len(manifest['roles'])
    assert data['q'].shape == (count+1, n, 18)
    assert data['applied'].shape == (count, n, 18)
    assert all(np.isfinite(value).all() for value in data.values())
    controller = manifest['native_controller']
    offset, scale = (np.asarray(controller[key], np.float32) for key in ('offset', 'scale'))
    mask = np.zeros(18, np.float32); mask[ACTIVE] = 1
    errors = dict(intended=0., applied=0., requested_PD=0., actual_PD=0.)
    for tick in range(count):
        for env in range(n):
            live = data['q'][tick, env]
            begin = 0 if env < 3 else manifest['window_queries'][env-3]
            active = env in (1, 2) or (env >= 3 and begin <= tick < begin+8)
            if active:
                query = begin + ((tick-begin)//4)*4
                index = tick-query
                assert data['query_tick'][tick, env] == query
                assert data['chunk_index'][tick, env] == index
                target = original['pd_targets'][tick].copy() if env == 1 else (
                    original['reference_q'][tick+1].copy() if args.q_source == 'reference'
                    else original['q'][tick+1].copy())
                if env != 1 and args.q_source != 'reference' and manifest.get('q_velocity_feedforward'):
                    assert manifest['q_velocity_feedforward'] == 'finite-difference-0.1s.v1'
                    target += 3 * (original['q'][tick+1] - original['q'][tick])
                if env != 1 and args.q_source != 'reference' and manifest.get('oracle_finger_PD'):
                    target[6:] = original['pd_targets'][tick, 6:]
                if (args.q_source == 'reference' and args.reference_branch == 'nearest-live'
                        and env != 1):
                    target[3:6] = nearest_euler(
                        Rotation.from_euler('XYZ', target[3:6]), live[3:6][None])[0]
                target = couple(target)
                normalized = (target-offset)/scale
                normalized[:6] = (target[:6]-live[:6]-offset[:6])/scale[:6]
                normalized[6:] = normalized[6:]*2-1
                intended = normalized*mask
                requested = target
            else:
                assert data['query_tick'][tick, env] == -1
                assert data['chunk_index'][tick, env] == -1
                intended = original['applied'][tick].copy()
                requested = None
            applied = intended.clip(-1, 1)
            action = np.concatenate((applied[:6], (applied[6:]+1)/2))
            actual = offset+scale*action
            actual[:6] += live[:6]
            actual = couple(actual)
            if requested is None:
                requested = actual
            for key, expected in [('intended', intended), ('applied', applied),
                                  ('requested_PD', requested), ('actual_PD', actual)]:
                saved = data[{'requested_PD':'requested_pd', 'actual_PD':'pd_targets'}.get(key, key)][tick, env]
                errors[key] = max(errors[key], float(np.abs(saved-expected).max()))
            assert data['q_intervention'][tick, env] == (active and env != 1)
            assert data['clipped'][tick, env] == bool(np.abs(intended-applied).max() > 1e-6)
    assert max(errors.values()) < 5e-6, errors
    reset = {key: float(np.abs(data[key][0]-original[key][0]).max())
             for key in ('q', 'dq', 'hand', 'obj', 'velocity')}
    assert max(reset.values()) <= 1e-4, reset
    def metrics(env, begin=0, end=count):
        frames = slice(begin+1, end+1)
        hand = np.linalg.norm(data['hand'][frames, env]-original['hand'][frames], axis=-1)*1000
        obj = np.linalg.norm(data['obj'][frames, env, :3, 3]-original['obj'][frames, :3, 3], axis=-1)*1000
        hand_paired = np.linalg.norm(data['hand'][frames, env]-data['hand'][frames, 0], axis=-1)*1000
        obj_paired = np.linalg.norm(data['obj'][frames, env, :3, 3]-data['obj'][frames, 0, :3, 3], axis=-1)*1000
        return dict(hand_EPE_mm=statistics(hand), object_position_mm=statistics(obj),
            wrist_xyz_mm=statistics(np.linalg.norm(data['q'][frames, env, :3]-original['q'][frames, :3], axis=-1)*1000),
            finger_max_error_rad=statistics(np.abs(data['q'][frames, env, 6:]-original['q'][frames, 6:]).max(-1)),
            paired_hand_EPE_mm=statistics(hand_paired), paired_object_mm=statistics(obj_paired),
            clipping=float(data['clipped'][begin:end, env].mean()),
            contact_fraction=float((np.abs(data['contact_force'][frames, env]) > .1).any((-1, -2)).mean()),
            held=held(data, env, begin+1, end+1))
    full = {role: metrics(env) for env, role in enumerate(manifest['roles'][:3])}
    native = full['recorded_native']
    calibration = native['hand_EPE_mm']['mean'] <= 5 and native['object_position_mm']['mean'] <= 5
    q_role = 'reference_q_full' if args.q_source == 'reference' else 'realized_q_full'
    q = full[q_role]
    hand_pass = q['hand_EPE_mm']['mean'] <= 10 and q['hand_EPE_mm']['p95'] <= 20
    held_eligible = native['held']['maximum_held_frames'] >= 45
    object_pass = q['object_position_mm']['mean'] <= 10
    if held_eligible:
        object_pass &= q['held']['maximum_held_frames'] >= max(45, .8*native['held']['maximum_held_frames'])
    classification = 'UNCLEAR' if not calibration or manifest['engineering_only'] else ('A' if hand_pass and object_pass else 'B' if hand_pass else 'C')
    windows = {}
    for env, begin in enumerate(manifest['window_queries'], 3):
        if begin+8 > count:
            continue
        fork = {key: float(np.abs(data[key][begin, env]-data[key][begin, 0]).max())
                for key in ('q', 'dq', 'hand', 'obj', 'velocity')}
        baseline, treated = metrics(0, begin, begin+8), metrics(env, begin, begin+8)
        windows[manifest['roles'][env]] = dict(query=begin, fork_errors=fork,
            calibration_pass=max(fork.values()) <= 1e-4 and baseline['hand_EPE_mm']['mean'] <= 5 and baseline['object_position_mm']['mean'] <= 5,
            baseline=baseline, q_target=treated)
    pd = full['recorded_PD']
    pd_calibration = pd['hand_EPE_mm']['mean'] <= 5 and pd['object_position_mm']['mean'] <= 5
    report = dict(pass_=True, scope='independent command/clock audit; single-seed oracle physics Probe',
        q_source=args.q_source,
        reference_branch=expected_branch,
        errors=errors, reset_errors=reset, full_calibration_pass=calibration,
        classification=classification, supplemental_PD_calibration_pass=pd_calibration,
        held45_eligible=held_eligible, full=full, windows=windows)
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(pass_=True, classification=classification, calibration=calibration, full=full)))


if __name__ == '__main__':
    main()
