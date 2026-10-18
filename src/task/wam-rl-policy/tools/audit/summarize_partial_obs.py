#!/usr/bin/env python3
"""Summarize matched partial-observation identification runs.

Reads one or more ``results.json`` files produced by
``train_partial_obs_identification.py`` and reports:

* the per-variant validation table;
* paired differences by (fold, model seed), which removes the shared fold
  variance from the comparison.
"""
import argparse
import json
from pathlib import Path

import numpy as np

PAIRS = (
    ('p1_tau', 'p1'),
    ('p1_shuffle', 'p1'),
    ('p1_tau', 'p1_shuffle'),
    ('p2_tau', 'p2'),
    ('p2_shuffle', 'p2'),
    ('p2_tau', 'p2_shuffle'),
    ('p2_tau', 'full'),
)
METRICS = ('val_all', 'val_xyz', 'val_rot', 'val_fingers')


def load(path):
    value = json.loads(Path(path).read_text())
    rows = {}
    for name, items in value['per_run'].items():
        rows[name] = {(item['fold'], item['model_seed']): item for item in items}
    return value, rows


def paired(rows, left, right):
    keys = sorted(set(rows[left]) & set(rows[right]))
    out = {}
    for metric in METRICS:
        split, name = metric.split('_', 1)
        deltas = np.asarray([rows[left][key][split][name] - rows[right][key][split][name]
                             for key in keys])
        out[metric] = dict(mean=float(deltas.mean()), std=float(deltas.std()),
                           n=int(len(deltas)))
    return out


def summarize(label, path):
    value, rows = load(path)
    table = []
    for name in value['input_dims']:
        agg = value['aggregate'][name]
        table.append(dict(
            model=name, dim=value['input_dims'][name],
            train=agg['train_all']['mean'],
            val=agg['val_all']['mean'], val_std=agg['val_all']['std'],
            xyz=agg['val_xyz']['mean'], rot=agg['val_rot']['mean'],
            fingers=agg['val_fingers']['mean'],
            approach=agg['val_phase_all']['approach']['mean'],
            contact=agg['val_phase_all']['contact']['mean'],
            lift=agg['val_phase_all']['lift']['mean']))
    comparisons = {}
    for left, right in PAIRS:
        if left in rows and right in rows:
            comparisons[f'{left}_minus_{right}'] = paired(rows, left, right)
    return dict(label=label, path=str(Path(path).resolve()),
                rows=value['rows'], seeds=value['teacher_seeds'],
                successful_only=value['successful_only'],
                baseline=value['baseline'], table=table,
                paired=comparisons)


def markdown(report):
    lines = [f"### {report['label']}",
             f"rows={report['rows']}, teacher seeds={len(report['seeds'])}, "
             f"successful_only={report['successful_only']}", '',
             '| model | dim | train L1 | val L1 | xyz | rot | fingers | approach | contact | lift |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for row in report['table']:
        lines.append('| {model} | {dim} | {train:.4f} | {val:.4f} ± {val_std:.4f} | '
                     '{xyz:.4f} | {rot:.4f} | {fingers:.4f} | {approach:.4f} | '
                     '{contact:.4f} | {lift:.4f} |'.format(**row))
    lines += ['', '| comparison | Δall | Δxyz | Δrot | Δfingers |', '|---|---|---|---|---|']
    for name, values in report['paired'].items():
        lines.append('| {} | {:+.4f} ± {:.4f} | {:+.4f} ± {:.4f} | {:+.4f} ± {:.4f} | '
                     '{:+.4f} ± {:.4f} |'.format(
                         name,
                         values['val_all']['mean'], values['val_all']['std'],
                         values['val_xyz']['mean'], values['val_xyz']['std'],
                         values['val_rot']['mean'], values['val_rot']['std'],
                         values['val_fingers']['mean'], values['val_fingers']['std']))
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, nargs='+', required=True)
    p.add_argument('--labels', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if len(args.inputs) != len(args.labels):
        raise ValueError('one label per results file is required')
    reports = [summarize(label, path) for label, path in zip(args.labels, args.inputs)]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(reports, indent=2))
    (args.output / 'summary.md').write_text('\n'.join(markdown(r) for r in reports))
    for report in reports:
        print(markdown(report))


if __name__ == '__main__':
    main()
