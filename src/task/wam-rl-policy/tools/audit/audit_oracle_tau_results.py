"""Recompute goal attainment from saved observations, without rerunning Gym."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from wam_rl_policy.hold_metrics import summarize_realtime_hold


def teacher_hold_endpoints(directory, observation, current_pose):
    """Inspect the saved teacher's8-step object endpoint at in-goal queries.

    Earlier oracle packets named the future pose ``object_pose``; corrected
    packets name it ``future_object_pose``. This is an endpoint check, not a
    claim of continuous holding during the teacher fork.
    """
    from scipy.spatial.transform import Rotation
    rotation = (Rotation.from_quat(current_pose[:, 3:7]).as_matrix()
                @ Rotation.from_quat(observation[:, 194:198]).as_matrix().transpose(0, 2, 1))
    reconstructed = np.einsum('bij,bj->bi', rotation, observation[:, 191:194])
    error = float(np.abs(reconstructed - current_pose[:, :3]).max())
    goal = current_pose[:, :3] + np.einsum('bij,bj->bi', rotation, observation[:, 204:207])
    drift = float(np.abs(goal - goal[0]).max())
    if error > 1e-5 or drift > 1e-5:
        raise ValueError('canonical/world reconstruction or fixed-goal check failed')
    near = np.linalg.norm(observation[:, 204:207], axis=-1) <= .05
    endpoints = []
    for tick in np.flatnonzero(near):
        with np.load(Path(directory) / f'oracle/tick-{tick:03d}/tau.npz', allow_pickle=False) as data:
            if 'future_object_pose' in data.files:
                future = data['future_object_pose']
            elif 'current_object_pose' not in data.files:
                future = data['object_pose']  # Pinned legacy t+8 field.
            else:
                raise ValueError('oracle packet omits its future object pose')
            endpoints.append(float(np.linalg.norm(future[:3] - goal[tick])))
    return dict(in_goal_queries=int(near.sum()),
                teacher_in_goal_after8=sum(d <= .05 for d in endpoints),
                teacher_endpoint_distances_m=endpoints,
                current_world_reconstruction_error=error, fixed_goal_drift=drift)


def audit_run(directory, *, teacher_hold=False):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    with np.load(directory / 'trajectory.npz', allow_pickle=False) as data:
        obs = data['obs']
        if obs.ndim != 2 or obs.shape[1] != 300 or not np.isfinite(obs).all():
            raise ValueError('finite official full-state observations required')
        # Pinned UniDex full_state: unpose_vec(goal_pos - object_pos).
        # Rotation preserves its norm. The upstream successes flag is latched.
        distance = np.linalg.norm(obs[:, 204:207], axis=-1)
        if 'goal_distance' in data.files:
            np.testing.assert_allclose(distance, data['goal_distance'], atol=1e-6, rtol=1e-5)
        prefix_equal = np.array_equal(data['action'], data['action_prefix'])
        if not prefix_equal:
            raise ValueError('recorded action and replay prefix differ: ' + str(directory))
        live = summarize_realtime_hold(distance)
        height = data['object_pose'][:, 2]
        lift = height - height[0]
        current_pose = data['object_pose'].copy()
        archived_replay = float(data['replay_error'].max())
        initial_obs = obs[0].tolist()
    query_errors = []
    for packet in sorted((directory / 'oracle').glob('tick-*/result.json')):
        query = json.loads(packet.read_text())
        if query.get('status') != 'COMPLETE' or query.get('fork_valid') is not True:
            raise ValueError('incomplete or invalid fork: ' + str(packet))
        query_errors.append(float(query['replay_error']))
    result = dict(run=str(directory.resolve()), seed=manifest['seed'], variant=manifest['variant'],
                policy=manifest['policy'], ticks=len(obs), max_lift_m=float(lift.max()),
                upstream_reported_success_frames=manifest.get('success_frames_ever',
                                                             manifest.get('success_frames')),
                last_record_hold=live, action_prefix_equal=prefix_equal,
                max_replay_error=archived_replay, oracle_queries=len(query_errors),
                max_query_replay_error=max(query_errors) if query_errors else None,
                terminal_after_last_action=(dict(distance_m=manifest['final_goal_distance_m'],
                                                 success=bool(manifest['final_success']))
                                           if 'final_goal_distance_m' in manifest else None),
                initial_observation=initial_obs,
                limitation='Saved observations precede each action; old runs omit the post-final-action state.')
    if teacher_hold and manifest['variant'] == 'p2_tau':
        result['teacher_hold_endpoints'] = teacher_hold_endpoints(directory, obs, current_pose)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--teacher-hold', action='store_true')
    args = parser.parse_args()
    report = dict(schema='ref2dex.oracle-tau-observed-goal-audit.v1',
                  runs=[audit_run(path, teacher_hold=args.teacher_hold) for path in args.runs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    for row in report['runs']:
        print(json.dumps({k: v for k, v in row.items() if k != 'initial_observation'}))


if __name__ == '__main__':
    main()
