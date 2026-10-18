#!/usr/bin/env python3
"""Build a partial-observation oracle-tau identification dataset from teacher rollouts.

Each input run is a single-environment closed-loop teacher rollout (the narrow
serial contract).  The oracle plan is the teacher's own future hand landmarks
within the same rollout, so no fork is needed: for teacher states the reference
trajectory *is* the live trajectory.

The executor observation has three variants:

* ``full``: the official 300-D policy observation.
* ``p1``: robot core only (DOF position/velocity + previous action), 68-D.
* ``p2``: robot core + fingertip rigid-body state + hand pose, 139-D.

``p1``/``p2`` remove the object pose/velocity, the contact-force channels and
the visual feature, so tau genuinely carries object/future information.
"""
import argparse
import json
from pathlib import Path

import numpy as np

K = 8
SLICES = {
    'p1': ((0, 44), (167, 191)),
    'p2': ((0, 44), (66, 131), (161, 191)),
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2,
                                 default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


def build(root, horizon=K):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('status') != 'COMPLETE':
        raise ValueError(f'incomplete teacher rollout: {root}')
    if manifest.get('envs') != 1:
        raise ValueError(f'single-environment teacher rollout required: {root}')
    with np.load(root / 'trajectory.npz', allow_pickle=False) as source:
        obs = source['obs'][:, 0]
        hand = source['hand'][:, 0]
        action = source['action'][:, 0]
        success = source['success'][:, 0]
        object_pose = source['object_pose'][:, 0]
    ticks = obs.shape[0]
    rows = ticks - horizon
    if rows <= 0:
        raise ValueError(f'rollout shorter than horizon: {root}')
    tau = np.stack([hand[t + 1:t + 1 + horizon] - hand[t] for t in range(rows)])
    partial = {name: np.concatenate([obs[:rows, lo:hi] for lo, hi in slices], axis=1)
               for name, slices in SLICES.items()}
    data = dict(full=obs[:rows], tau=tau, action=action[:rows],
                tick=np.arange(rows, dtype=np.int32),
                seed=np.full(rows, int(manifest['seed']), np.int32),
                success=success[:rows])
    for name, value in partial.items():
        data[name] = value
    data['object_pose'] = object_pose[:rows]
    return data, manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--horizon', type=int, default=K)
    args = p.parse_args()
    if args.output == Path('/tmp') or Path('/tmp') in args.output.parents:
        raise ValueError('dataset must remain in the project workspace')
    seeds = []
    blocks = []
    manifests = []
    for run in args.runs:
        data, manifest = build(run, args.horizon)
        blocks.append(data)
        seeds.append(int(manifest['seed']))
        manifests.append(dict(seed=manifest['seed'], run=str(Path(run).resolve()),
                              rows=int(data['full'].shape[0]),
                              final_success=float(data['success'][-1]),
                              ticks=int(manifest['ticks'])))
    keys = blocks[0].keys()
    merged = {key: np.concatenate([block[key] for block in blocks], axis=0) for key in keys}
    for key, value in merged.items():
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():
            raise ValueError(f'nonfinite dataset field {key}')
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'identification_data.npz', **merged)
    write_json(args.output / 'dataset.json', dict(
        schema='ref2dex.unidex-partial-obs-tau-dataset.v1',
        horizon=args.horizon, rows=int(merged['full'].shape[0]),
        seeds=seeds, runs=manifests,
        full_dim=int(merged['full'].shape[1]),
        p1_dim=int(merged['p1'].shape[1]), p2_dim=int(merged['p2'].shape[1]),
        tau_dim=int(np.prod(merged['tau'].shape[1:])),
        partial_slices={name: [list(s) for s in slices] for name, slices in SLICES.items()},
        tau_definition='teacher_future_hand_landmarks_minus_live_hand_landmarks',
        note=('Teacher states only. tau is the teacher future from the same rollout, '
              'so it is the live-state oracle plan without a fork.')))
    print(json.dumps(dict(rows=int(merged['full'].shape[0]), seeds=seeds,
                          p1_dim=merged['p1'].shape[1], p2_dim=merged['p2'].shape[1],
                          tau_dim=int(np.prod(merged['tau'].shape[1:]))), indent=2))


if __name__ == '__main__':
    main()
