"""Merge the completed serial arms and parallel workers without reselection."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(first, parallel, grouped=False):
    sources = dict(baseline=first, current_reference=first,
                   current_geometry=parallel/'current_geometry', both=parallel/'both')
    if grouped:
        sources = dict(baseline=parallel/'baseline_geometry', current_reference=parallel/'reference_both',
                       current_geometry=parallel/'baseline_geometry', both=parallel/'reference_both')
        first = parallel/'baseline_geometry'
    expected = json.loads((first/'manifest.json').read_text())
    panels = np.load(first/'panels.npz')
    stats = np.load(first/'extra_train_stats.npz')
    retained = json.loads((first/'partial_result.json').read_text())
    results, provenance, curves = {}, {}, {}
    for arm, directory in sources.items():
        manifest = json.loads((directory/'manifest.json').read_text())
        for key in ['parent_step', 'parent_checkpoint', 'source', 'updates', 'batch', 'learning_rate', 'seeds']:
            assert manifest[key] == expected[key], (arm, key)
        for name, reference in [('panels.npz', panels), ('extra_train_stats.npz', stats)]:
            candidate = np.load(directory/name)
            assert set(candidate.files) == set(reference.files)
            for key in reference.files:
                np.testing.assert_array_equal(candidate[key], reference[key], err_msg=arm+':'+key)
        if directory == first:
            results[arm] = retained[arm]
        else:
            worker = json.loads((directory/'result.json').read_text())
            assert worker['status'] == 'COMPLETED'
            results[arm] = worker['arms'][arm]
        curve = json.loads((directory/(arm+'-curve.json')).read_text())
        key = lambda row: (row['panels']['val']['xyz_p95_mm'], row['panels']['val']['xyz_mean_mm'])
        assert results[arm]['selected'] == min(curve, key=key)
        assert results[arm]['initial'] == curve[0]
        assert results[arm]['final'] == curve[-1]
        assert [row['step'] for row in curve] == list(range(0, expected['updates']+1, 500))
        curves[arm] = curve
        provenance[arm] = dict(directory=str(directory), git_commit=manifest['git_commit'],
                               selected_step=results[arm]['selected']['step'])
    # Cross-device initial functions must agree before comparisons are interpreted.
    for arm, values in results.items():
        for split in ['train', 'val']:
            for metric in ['xyz_mean_mm', 'xyz_p95_mm', 'normalized_all8_L1', 'clipping']:
                np.testing.assert_allclose(values['initial']['panels'][split][metric],
                    results['baseline']['initial']['panels'][split][metric], rtol=1e-4, atol=1e-5)
    baseline = results['baseline']['selected']['panels']['val']
    signals = {}
    for arm in list(sources)[1:]:
        value = results[arm]['selected']['panels']['val']
        mean_ratio = value['xyz_mean_mm']/baseline['xyz_mean_mm']
        p95_ratio = value['xyz_p95_mm']/baseline['xyz_p95_mm']
        signals[arm] = dict(mean_ratio=mean_ratio, p95_ratio=p95_ratio,
                           promising=mean_ratio <= .85 and p95_ratio <= .85 and value['clipping'] < .01)
    paths = {p for directory in [first, parallel] for p in directory.rglob('*') if p.is_file()}
    artifacts = sum(p.stat().st_size for p in paths)
    assert artifacts < (300_000_000 if grouped else 150_000_000), artifacts
    result = dict(status='COMPLETED', conclusion='PROMISING' if any(v['promising'] for v in signals.values()) else 'UNCLEAR',
                  arms=results, signals=signals, provenance=provenance, artifacts_bytes=artifacts,
                  matched_checks='identical TRAIN stats, panel indices, initial metrics; original selection rule retained',
                  scope='GT-tau R-only current-information Probe; no grasping or deployment claim')
    path = parallel/'result.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, metric, title in zip(axes, ['xyz_mean_mm', 'xyz_p95_mm'], ['Mean', '95th percentile']):
        for arm, curve in curves.items():
            ax.plot([r['step'] for r in curve], [r['panels']['val'][metric] for r in curve], marker='o', label=arm)
        ax.set(xlabel='Matched R updates', ylabel='First PD wrist XYZ error (mm)', title=title)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('Current information: GT tau, same step-500 R, fixed dev panel (n=256)')
    fig.savefig(parallel/'current-information.png', dpi=160)
    plt.close(fig)
    for arm, values in results.items():
        m = values['selected']['panels']['val']
        print(arm, values['selected']['step'], 'mean/p95 mm', round(m['xyz_mean_mm'], 2),
              round(m['xyz_p95_mm'], 2), 'clipping', m['clipping'])
    print(result['conclusion'], json.dumps(signals))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first', type=Path, required=True)
    p.add_argument('--parallel', type=Path, required=True)
    p.add_argument('--grouped', action='store_true')
    a = p.parse_args()
    summarize(a.first, a.parallel, a.grouped)
