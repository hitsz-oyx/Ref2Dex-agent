"""Compare acquisition and startup curves from two residual curriculum runs."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    path = run / 'monitor.jsonl'
    rows = []
    if path.exists():
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
    recent = rows[-20:]
    phases = {}
    for phase in manifest['phases']:
        trials = [trial for row in recent for trial in row['grasp_trials'] if trial['phase'] == phase]
        acquire = [trial for trial in trials if not trial['initially_held']]
        phases[str(phase)] = dict(
            trials=len(trials), initially_held=sum(trial['initially_held'] for trial in trials),
            new_hold45_fraction=float(np.mean([trial['new_hold45'] for trial in acquire])) if acquire else None,
            median_maximum_held_frames=float(np.median([trial['maximum_held_frames'] for trial in trials])) if trials else None,
            mean_held_fraction=float(np.mean([trial['held_fraction'] for trial in trials])) if trials else None,
            mean_normalized_advantage=float(np.mean([row['phase_advantages'][str(phase)]['normalized_advantage_mean']
                                                    for row in recent])) if recent else None,
            mean_positive_advantage_fraction=float(np.mean([row['phase_advantages'][str(phase)]['positive_advantage_fraction']
                                                           for row in recent])) if recent else None,
        )
    evaluations = []
    for path in sorted(run.glob('eval[0-9]*.json')):
        value = json.loads(path.read_text())
        evaluations.append(dict(update=value['update'],
            hold45=sum(outcome['maximum_held_frames'] >= 45 for outcome in value['outcomes']),
            placement=sum(outcome['success'] for outcome in value['outcomes']),
            maximum_held_frames=[outcome['maximum_held_frames'] for outcome in value['outcomes']],
            residual_rms=value['eval_residual_rms'], status=value['status']))
    return dict(run_id=manifest['run_id'], status=manifest['status'], curriculum=manifest['curriculum'],
                parent_update=manifest['resume_updates'], completed_additional_updates=len(rows),
                last_update=rows[-1]['update'] if rows else manifest['resume_updates'],
                last20_phase_acquisition=phases, evaluations=evaluations,
                last_value_explained_variance=rows[-1]['value_explained_variance'] if rows else None,
                last_actor_lr=rows[-1]['actor_lr'] if rows else None,
                mean_held_fraction=float(np.mean([row['reward_terms']['held'] for row in rows])) if rows else None,
                mean_saturation=float(np.mean([row['saturation_by_dof'] for row in rows])) if rows else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', type=Path, required=True)
    parser.add_argument('--b', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifests = [json.loads((run / 'manifest.json').read_text()) for run in (args.a, args.b)]
    for key in ('resume', 'resume_updates', 'envs', 'rollout', 'alpha', 'gae_lambda',
                'retargeter_identity', 'effect_identity', 'control_identity'):
        assert manifests[0][key] == manifests[1][key], 'unmatched ' + key
    with np.load(args.a / 'reset-prefix.npz') as left, np.load(args.b / 'reset-prefix.npz') as right:
        assert set(left.files) == set(right.files)
        for key in left.files:
            np.testing.assert_array_equal(left[key], right[key])
    result = dict(common_parent_and_prefix=True, A=summarize(args.a), B=summarize(args.b),
                  claim='one matched-state exploratory pair; GPU physics variation and seed limits apply')
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
