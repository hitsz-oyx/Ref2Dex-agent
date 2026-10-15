"""Export existing WAM-v2 metrics without model inference or training changes."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path


DOMAINS = ('OakInk2', 'GRAB', 'ARCTIC')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_curve(curve, expected_steps):
    if [row['step'] for row in curve] != list(expected_steps):
        raise ValueError('Incomplete or duplicate dev evaluations')
    keys = set(curve[0]['metrics'])
    for row in curve:
        if set(row['metrics']) != keys or not math.isfinite(row['objective']):
            raise ValueError('Metric schema drift or nonfinite objective')
        for value in row['metrics'].values():
            mean, count = value['mean_mm'], value['count']
            if count < 0 or (count == 0) != (mean is None):
                raise ValueError('Empty categories must retain null mean and zero count')
            if mean is not None and not math.isfinite(mean):
                raise ValueError('Nonfinite recorded physical metric')
    for key in keys:
        if any(row['metrics'][key]['count'] != curve[0]['metrics'][key]['count'] for row in curve):
            raise ValueError('Fixed dev metric counts changed')
        if '/goal_only/' in key and any(row['metrics'][key] != curve[0]['metrics'][key] for row in curve):
            raise ValueError('Fixed goal-only baseline changed')


def export_curve(output, label, curve):
    write_json(output / (label + '-dev.json'), curve)
    long_rows, core = [], []
    for row in curve:
        compact = dict(step=row['step'], normalized_objective=row['objective'])
        for sid, domain in enumerate(DOMAINS):
            for metric in ('hand_all24', 'scene_all24'):
                compact[domain + '/' + metric + '_mm'] = row['metrics'][f'source{sid}/{metric}']['mean_mm']
        core.append(compact)
        for key, value in row['metrics'].items():
            source, metric = key.split('/', 1)
            long_rows.append(dict(step=row['step'], source=source, domain=DOMAINS[int(source[6:])],
                metric=metric, mean_mm=value['mean_mm'], count=value['count']))
    write_csv(output / (label + '-dev-core.csv'), core)
    write_csv(output / (label + '-dev-all-metrics.csv'), long_rows)
    return core


def md_table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |',
        '| ' + ' | '.join(['---'] * len(header)) + ' |']
        + ['| ' + ' | '.join(str(value) for value in row) + ' |' for row in rows])


def compact_table(curve):
    return md_table(['Step', 'Dev objective', 'OakInk2 hand', 'GRAB hand', 'ARCTIC hand',
        'OakInk2 scene', 'GRAB scene', 'ARCTIC scene'],
        [[row['step'], f'{row["objective"]:.6f}'] +
         [f'{row["metrics"][f"source{sid}/{metric}"]["mean_mm"]:.3f}'
          for metric in ('hand_all24', 'scene_all24') for sid in range(3)] for row in curve])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrain-snapshot', type=Path, required=True)
    parser.add_argument('--local-probe', type=Path, required=True)
    parser.add_argument('--robot-source', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    snapshot = json.loads(args.pretrain_snapshot.read_text())
    curve = snapshot['curve']
    validate_curve(curve, range(0, 80001, 1000))
    local_curve = json.loads((args.local_probe / 'curve.json').read_text())
    validate_curve(local_curve, range(0, 1001, 250))
    output = args.output.resolve()
    if Path(__file__).resolve().parents[5] / 'outputs' not in output.parents:
        raise ValueError('Local output directory required')
    output.mkdir(exist_ok=False)
    export_curve(output, 'pretrain-remote80k', curve)
    export_curve(output, 'pretrain-local1k', local_curve)
    write_json(output / 'pretrain-train.json', snapshot.get('train', []))
    training = [dict(step=row['step'], loss=row['loss'], hand=row['terms']['hand'],
        scene=row['terms']['scene'], elapsed_s=row['elapsed_s']) for row in snapshot.get('train', [])]
    write_csv(output / 'pretrain-train.csv', training)
    robot_curve, robot_train, panels = [], [], []
    import numpy as np
    for index, source in enumerate(args.robot_source):
        manifest = json.loads((source / 'manifest.json').read_text())
        if index + 1 < len(args.robot_source):
            next_manifest = json.loads((args.robot_source[index + 1] / 'manifest.json').read_text())
            cutoff = next_manifest['provenance']['resume_step']
        else:
            cutoff = math.inf
        rows = json.loads((source / 'curve.json').read_text())
        config = manifest['config']
        for row in rows:
            if row['step'] > cutoff:
                continue
            residual = row['objective'] - config['action_weight'] * row['action'] - config['anchor_weight'] * row['anchor']
            robot_curve.append(dict(step=row['step'], source_run=source.name, objective=row['objective'],
                action=row['action'], anchor=row['anchor'], forecast_residual=residual, count=row['count']))
        for line in (source / 'train.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['step'] <= cutoff:
                robot_train.append(dict(step=row['step'], source_run=source.name, loss=row['loss'], **row['terms']))
        panels.append(np.load(source / 'validation.npy', allow_pickle=False))
    if any(not np.array_equal(panel, panels[0]) for panel in panels):
        raise ValueError('Resumed robot run changed the fixed dev panel')
    if len({row['step'] for row in robot_curve}) != len(robot_curve):
        raise ValueError('Duplicate joint-training steps')
    write_json(output / 'robot-dev.json', robot_curve)
    write_csv(output / 'robot-dev.csv', robot_curve)
    write_csv(output / 'robot-train.csv', robot_train)
    best = min(curve, key=lambda row: row['objective'])
    last = curve[-1]
    breakdown = []
    for metric in sorted(key.split('/', 1)[1] for key in last['metrics'] if key.startswith('source0/')):
        values = []
        for sid in range(3):
            for row in (best, last):
                mean = row['metrics'][f'source{sid}/{metric}']['mean_mm']
                values.append('N/A' if mean is None else f'{mean:.3f}')
        breakdown.append([metric] + values)
    report = ('单位：全部物理误差为mm；objective为原训练归一化目标。\n\n'
        '### 远端80k完整dev主指标（固定384窗）\n\n' + compact_table(curve)
        + '\n\n### Best与final全部物理指标\n\n'
        + md_table(['Metric'] + [domain + ' step' + str(step) for domain in DOMAINS
                                 for step in (best['step'], last['step'])], breakdown)
        + '\n\n### 本地1k独立Probe（固定96窗，不与远端绝对值混比）\n\n'
        + compact_table(local_curve) + '\n')
    (output / 'pretrain-report.md').write_text(report)
    robot_report = md_table(['Step', 'Total', 'Pred τ action', 'GT τ anchor', 'Forecast residual'],
        [[row['step']] + [f'{row[key]:.6f}' for key in ('objective', 'action', 'anchor', 'forecast_residual')]
         for row in robot_curve])
    (output / 'robot-report.md').write_text(robot_report + '\n')
    metadata = dict(created_at=datetime.now(timezone.utc).isoformat(), inputs=dict(
        pretrain_snapshot=str(args.pretrain_snapshot.resolve()), local_probe=str(args.local_probe.resolve()),
        robot_sources=[str(p.resolve()) for p in args.robot_source]),
        pretrain_evaluations=len(curve), pretrain_metric_groups=len(last['metrics']),
        pretrain_metric_rows=len(curve) * len(last['metrics']), pretrain_train_logs=len(training),
        pretrain_best_objective_step=best['step'], pretrain_final_step=last['step'],
        pretrain_checkpoint_aliases=snapshot.get('aliases', {}),
        pretrain_checkpoint_files=snapshot.get('checkpoint_files', []),
        robot_evaluations=len(robot_curve), robot_last_dev_step=robot_curve[-1]['step'],
        robot_identical_resume_dev_panel=True,
        missing_metrics=['pretrain dev normalized hand/scene loss separately',
            'robot dev hand/scene loss separately; only aggregate residual recoverable',
            'robot physical PD errors/grasp success from these training logs'],
        scope='Existing recorded offline metrics; no new inference, simulation or causal/grasp validation')
    write_json(output / 'metadata.json', metadata)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    steps = [row['step'] for row in curve]
    for sid, domain in enumerate(DOMAINS):
        for ax, metric in zip(axes[:2], ('hand_all24', 'scene_all24')):
            ax.plot(steps, [row['metrics'][f'source{sid}/{metric}']['mean_mm'] for row in curve], label=domain)
    axes[2].plot(steps, [row['objective'] for row in curve], label='fixed dev objective')
    for ax, title in zip(axes, ('Hand all24 EPE (mm)', 'Scene all24 EPE (mm)', 'Normalized objective')):
        ax.set(xlabel='Pretrain updates', title=title)
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(output / 'pretrain-80k-curves.png', dpi=160)
    plt.close(fig)
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == '__main__':
    main()
