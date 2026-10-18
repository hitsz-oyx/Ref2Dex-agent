#!/usr/bin/env python3
"""Audit UniDex serial single-environment replay across fresh processes.

Compares per-tick state, observation and teacher action packets bitwise.  The
contract is the narrow one validated earlier in consequence-evaluator
(P-20261010-serial-finger-pulse-prefix): one environment, fresh process, fixed
seed, no PhysX restore, no fork.
"""
import argparse
import json
from pathlib import Path

import numpy as np

FIELDS = ('obs', 'q', 'dq', 'hand', 'object_pose', 'forces', 'dof_force',
          'action', 'success')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def load(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('status') != 'COMPLETE':
        raise ValueError(f'incomplete serial run: {root}')
    with np.load(root / 'trajectory.npz', allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    missing = set(FIELDS).difference(data)
    if missing:
        raise ValueError(f'missing fields in {root}: {sorted(missing)}')
    return data, manifest, root


def first_diff(left, right, field):
    ticks = min(left.shape[0], right.shape[0])
    for tick in range(ticks):
        if left[tick].tobytes() != right[tick].tobytes():
            return tick
    if left.shape[0] != right.shape[0]:
        return ticks
    return None


def compare(reference, other):
    data_a, manifest_a, root_a = load(reference)
    data_b, manifest_b, root_b = load(other)
    if manifest_a.get('seed') != manifest_b.get('seed'):
        raise ValueError('serial runs disagree on seed')
    fields = {}
    for field in FIELDS:
        a, b = data_a[field], data_b[field]
        tick = first_diff(a, b, field)
        max_abs = None
        if a.shape == b.shape and a.dtype.kind in 'fc':
            max_abs = float(np.max(np.abs(a.astype('float64') - b.astype('float64'))))
        fields[field] = dict(
            bitwise_equal=bool(tick is None),
            first_diff_tick=tick,
            max_abs_diff=max_abs,
            shape_a=list(a.shape), shape_b=list(b.shape))
    divergence_ticks = [v['first_diff_tick'] for v in fields.values()
                        if v['first_diff_tick'] is not None]
    first = min(divergence_ticks) if divergence_ticks else None
    return dict(
        reference=str(root_a), other=str(root_b),
        ticks_a=int(data_a['obs'].shape[0]), ticks_b=int(data_b['obs'].shape[0]),
        bitwise_identical=bool(first is None),
        first_divergence_tick=first,
        diverging_fields=sorted(k for k, v in fields.items() if not v['bitwise_equal']),
        fields=fields)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if len(args.runs) < 2:
        raise ValueError('at least two fresh-process runs are required')
    reports = [compare(args.runs[0], run) for run in args.runs[1:]]
    all_identical = all(r['bitwise_identical'] for r in reports)
    first = min((r['first_divergence_tick'] for r in reports
                 if r['first_divergence_tick'] is not None), default=None)
    result = dict(
        schema='ref2dex.unidex-serial-replay-audit.v1',
        protocol='single_env_fresh_process_closed_loop_teacher',
        runs=[str(Path(p).resolve()) for p in args.runs],
        all_bitwise_identical=all_identical,
        first_divergence_tick=first,
        reports=reports)
    write(args.output, result)
    print(json.dumps(dict(all_bitwise_identical=all_identical,
                          first_divergence_tick=first,
                          runs=len(args.runs)), indent=2))
    for report in reports:
        print(json.dumps(dict(pair=[Path(report['reference']).name,
                                    Path(report['other']).name],
                              identical=report['bitwise_identical'],
                              first=report['first_divergence_tick'],
                              fields=report['diverging_fields']), indent=2))


if __name__ == '__main__':
    main()
