#!/usr/bin/env python3
"""Build DAgger rows from a closed-loop oracle-tau run.

Each oracle query already replayed the student state and saved the teacher's
eight-step action sequence.  Its first row is the teacher action at exactly
that student state, so the closed-loop run is also its own DAgger packet.
"""
import argparse
import json
from pathlib import Path

import numpy as np

P2_SLICES = ((0, 44), (66, 131), (161, 191))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2,
                               default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run = args.run.resolve()
    with np.load(run / 'trajectory.npz', allow_pickle=False) as source:
        trajectory = {key: source[key] for key in source.files}
    obs = trajectory['obs']
    rows = dict(p2=[], tau=[], action=[], student_action=[], tick=[],
                success=[], object_pose=[], replay_error=[])
    for tick in range(obs.shape[0]):
        query = run / 'oracle' / f'tick-{tick:03d}' / 'tau.npz'
        if not query.exists():
            raise ValueError(f'missing oracle query for tick {tick}')
        with np.load(query, allow_pickle=False) as source:
            teacher_actions = source['teacher_actions']
            tau = source['tau']
        rows['p2'].append(np.concatenate([obs[tick, lo:hi] for lo, hi in P2_SLICES]))
        rows['tau'].append(tau)
        rows['action'].append(teacher_actions[0])
        rows['student_action'].append(trajectory['action'][tick])
        rows['tick'].append(tick)
        rows['success'].append(trajectory['success'][tick])
        rows['object_pose'].append(trajectory['object_pose'][tick])
        rows['replay_error'].append(trajectory['replay_error'][tick])
    data = {}
    for key, value in rows.items():
        data[key] = (np.stack(value) if key in ('p2', 'tau', 'action', 'student_action',
                                                'object_pose') else np.asarray(value))
    if not np.isfinite(data['p2']).all() or not np.isfinite(data['tau']).all():
        raise ValueError('nonfinite DAgger packet')
    correction = np.abs(data['action'] - data['student_action'])
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'dagger_data.npz', **data)
    write_json(args.output / 'dagger.json', dict(
        schema='ref2dex.unidex-dagger-packet.v1',
        source_run=str(run), rows=int(len(data['p2'])),
        student_action_l1_vs_teacher=float(correction.mean()),
        xyz=float(correction[:, 0:3].mean()),
        rot=float(correction[:, 3:6].mean()),
        fingers=float(correction[:, 6:24].mean()),
        max_replay_error=float(np.max(data['replay_error'])),
        note=('Labels are the teacher action at the replayed student state; '
              'the student actions are not replaced in the source run.')))
    print(json.dumps(dict(rows=int(len(data['p2'])),
                          student_action_l1_vs_teacher=float(correction.mean()),
                          xyz=float(correction[:, 0:3].mean()),
                          rot=float(correction[:, 3:6].mean()),
                          fingers=float(correction[:, 6:24].mean())), indent=2))


if __name__ == '__main__':
    main()
