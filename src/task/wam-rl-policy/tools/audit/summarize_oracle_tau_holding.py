"""Compare the bounded holding adaptation with its archived development baseline."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_oracle_tau_results import audit_run, summarize_realtime_hold


def summarize(directory):
    audit = audit_run(directory)
    audit.pop('initial_observation')
    with np.load(directory / 'trajectory.npz', allow_pickle=False) as data:
        distance = np.linalg.norm(data['obs'][:, 204:207], axis=-1)
        height = data['object_pose'][:, 2]
        initial = {key: data[key][0].copy() for key in ('obs', 'object_pose')}
        terminal = audit['terminal_after_last_action']
        measured = (np.r_[distance, terminal['raw_observation_distance_m']]
                    if terminal and terminal['raw_observation_distance_m'] is not None else distance)
        audit['first150_preaction'] = summarize_realtime_hold(distance[:150])
        audit['all_available_measurements'] = summarize_realtime_hold(measured)
        audit['qualified'] = (audit['all_available_measurements']['longest_hold_frames'] >= 60
                              and terminal is not None and terminal['raw_observation_distance_m'] is not None
                              and terminal['raw_observation_distance_m'] <= .05)
        audit['terminal_available'] = len(measured) > len(distance)
        if 'terminal_object_pose' in data.files:
            height = np.r_[height, data['terminal_object_pose'][2]]
        audit['max_lift_including_available_terminal_m'] = float(height.max() - height[0])
    return audit, initial, measured, height - height[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adaptation', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # An absent post-action state cannot count as a qualified terminal hold.
    entries = [('old-tau', args.baseline), ('adapted-tau', args.adaptation / 'run-tau'),
               ('adapted-state', args.adaptation / 'run-state')]
    rows, curves, reference = {}, {}, None
    for label, directory in entries:
        row, initial, distance, lift = summarize(directory)
        if reference is None:
            reference = initial
        else:
            for key in reference:
                np.testing.assert_array_equal(reference[key], initial[key])
        rows[label], curves[label] = row, (distance, lift)
    if rows['adapted-tau']['oracle_queries'] != 200 or rows['adapted-tau']['max_query_replay_error'] > 1e-5:
        raise ValueError('requires all200 replay-verified oracle queries')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='ref2dex.oracle-tau-holding-comparison.v1', runs=rows,
                  initial_observation_and_object_pose_equal=True,
                  decision=('qualifies-on-one-development-initialization' if rows['adapted-tau']['qualified']
                            else 'stop-this-one-packet-adaptation'),
                  limitations=['Old baseline has150 pre-action states; compare first150 separately.',
                               'Added adaptation data and additional updates changed together.',
                               'One development initialization; no untouched TEST or WAM deployment claim.'])
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for label, (distance, lift) in curves.items():
        axes[0].plot(np.arange(len(distance)) / 60, distance, label=label)
        axes[1].plot(np.arange(len(lift)) / 60, lift, label=label)
    axes[0].axhline(.05, color='black', linestyle=':', label='5cm goal threshold')
    axes[0].set_ylabel('Goal distance (m)')
    axes[1].set_ylabel('Object lift from initial (m)')
    axes[1].set_xlabel('Simulation time (s), native60Hz')
    for axis in axes:
        axis.axvline(149 / 60, color='gray', linestyle='--', alpha=.5)
        axis.grid(alpha=.2)
    axes[0].legend()
    figure.suptitle('Seed482: archived baseline and bounded late-state adaptation')
    figure.tight_layout()
    figure.savefig(args.output / 'comparison.png', dpi=160)
    plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
