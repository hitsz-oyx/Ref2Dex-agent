"""Recompute goal attainment from saved observations, without rerunning Gym."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from wam_rl_policy.hold_metrics import summarize_realtime_hold


def audit_run(directory):
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
        archived_replay = float(data['replay_error'].max())
        initial_obs = obs[0].tolist()
    query_errors = []
    for packet in sorted((directory / 'oracle').glob('tick-*/result.json')):
        query = json.loads(packet.read_text())
        if query.get('status') != 'COMPLETE' or query.get('fork_valid') is not True:
            raise ValueError('incomplete or invalid fork: ' + str(packet))
        query_errors.append(float(query['replay_error']))
    return dict(run=str(directory.resolve()), seed=manifest['seed'], variant=manifest['variant'],
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(schema='ref2dex.oracle-tau-observed-goal-audit.v1',
                  runs=[audit_run(path) for path in args.runs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    for row in report['runs']:
        print(json.dumps({k: v for k, v in row.items() if k != 'initial_observation'}))


if __name__ == '__main__':
    main()
