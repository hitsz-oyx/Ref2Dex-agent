"""Audit saved C panels and optimizer continuity; no model inference or training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous audit')
    args.output.mkdir(parents=True)
    train = args.run / 'train'
    result = json.loads((train / 'result.json').read_text())
    initial = json.loads((train / 'evaluation_step0.json').read_text())
    names = list(initial)
    rows = []
    maximum_metric_difference = 0.
    maximum_oracle_array_difference = 0.
    # The continued run preserves the original step0 oracle arrays in its
    # parent run. This is array-level replay auditing, not a fresh label audit.
    manifest = json.loads((train / 'manifest.json').read_text())
    source = Path(manifest['resumed_checkpoint']).parent
    with np.load(source / 'panel_step0.npz') as raw:
        oracle = {key: raw[key].copy() for key in raw.files if '_oracle_' in key}
    for path in sorted(train.glob('evaluation_step*.json'),
                       key=lambda x: int(x.stem.split('step')[-1])):
        step = int(path.stem.split('step')[-1])
        values = json.loads(path.read_text())
        if step:
            with np.load(train / ('panel_step%d.npz' % step)) as raw:
                for name in names:
                    for metric, value in values[name].items():
                        if metric.startswith('Delta_E_'):
                            suffix = metric[len('Delta_E_'):]
                            actual = values[name]['effect_' + suffix] - values[name]['oracle_' + suffix]
                        elif metric.startswith('PD_'):
                            key = name + '_' + metric.split('_p95_')[0]
                            actual = float(np.percentile(raw[key], 95))
                        else:
                            actual = float(raw[name + '_' + metric].mean())
                        difference = abs(actual - value)
                        maximum_metric_difference = max(maximum_metric_difference, difference)
                        if not np.isfinite(actual) or difference > 1e-8:
                            raise ValueError('saved window-array metric mismatch: ' + metric)
                for key, value in oracle.items():
                    difference = float(np.max(np.abs(raw[key] - value)))
                    maximum_oracle_array_difference = max(maximum_oracle_array_difference, difference)
                    if difference > 1e-5:
                        raise ValueError('oracle window arrays changed')
        rows.append(dict(step=step, metrics=values))
    checkpoint = torch.load(train / ('checkpoint_step%d.pt' % result['step']),
                            map_location='cpu', weights_only=False)
    optimizer_steps = sorted(set(float(v['step']) for v in checkpoint['optimizer']['state'].values()))
    if optimizer_steps != [float(result['step'])]:
        raise ValueError('optimizer update count differs from resumed global step')
    if not all((train / ('validation_context_rank%d.pt' % rank)).is_file() for rank in range(4)):
        raise ValueError('resume validation context missing')
    curve = json.loads((train / 'curve.json').read_text())['curve']
    if any(row['model_frozen_max_difference'] != 0. or row['R_max_difference'] != 0. for row in curve):
        raise ValueError('frozen state changed')
    final = rows[-1]['metrics']
    report = dict(status='PASS', run_id=args.run.name, recorded_manifest_run_id=manifest['run_id'],
                  scope='saved per-window arrays, optimizer continuity and recorded frozen-state checks; no fresh inference or label audit',
                  evaluations_audited=len(rows)-1, maximum_metric_difference=maximum_metric_difference,
                  maximum_oracle_array_difference_mm=maximum_oracle_array_difference,
                  optimizer_steps=optimizer_steps, total_windows=result['step']*400,
                  C1_windows=(result['step']-500)*400,
                  final_mean8_gap_mm=np.mean([final[name]['Delta_E_all24_mm'] for name in names]),
                  initial_mean8_gap_mm=np.mean([initial[name]['Delta_E_all24_mm'] for name in names]),
                  original_strict_final_failures=[key for key, value in curve[-1]['strict_checks'].items() if not value],
                  continuation_final_failures=[key for key, value in curve[-1]['checks'].items() if not value])
    (args.output / 'result.json').write_text(json.dumps(report, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    steps = np.asarray([row['step'] for row in rows])/1000
    for name in names[:3]:
        axes[0].plot(steps, [row['metrics'][name]['Delta_E_all24_mm'] for row in rows], label=name)
    axes[0].set(title='Human frozen-world gap', ylabel='Predicted minus GT point EPE (mm)')
    for name in names[3:]:
        axes[1].plot(steps, [row['metrics'][name]['Delta_E_all24_mm'] for row in rows], label=name)
        axes[2].plot(steps, [row['metrics'][name]['PD_xyz_p95_mm'] for row in rows], label=name)
    axes[1].axhline(0, color='gray', linewidth=.6)
    axes[1].set(title='Robot frozen-world gap', ylabel='Predicted minus GT point EPE (mm)')
    axes[2].set(title='Robot first-PD XYZ p95', ylabel='Error (mm)', yscale='log')
    for axis in axes:
        axis.set_xlabel('Global optimizer updates (thousands)')
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    figure.suptitle('C single-seed offline continuation; recovery panels are TRAIN; negative gap is model discrepancy')
    figure.savefig(args.output/'curves.png', dpi=160)
    plt.close(figure)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
