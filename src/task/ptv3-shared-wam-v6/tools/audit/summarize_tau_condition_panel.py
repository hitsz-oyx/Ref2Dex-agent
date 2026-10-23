#!/usr/bin/env python3
"""Summarize four fixed-panel cells without re-running or sampling model inputs."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'tools/run'))
from evaluate_no_goal_no_event_panel import cluster_ci


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for cell in ('init-a', 'trained-a', 'init-b', 'trained-b'):
        parser.add_argument('--' + cell, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports, arrays = {}, {}
    for key in ('init_a', 'trained_a', 'init_b', 'trained_b'):
        path = getattr(args, key)
        reports[key] = json.loads((path / 'metrics.json').read_text())
        with np.load(path / 'paired-panel.npz') as z:
            arrays[key] = {k: z[k] for k in ('sums', 'counts', 'mothers')}
    first = reports['init_a']
    counts, mothers = arrays['init_a']['counts'], arrays['init_a']['mothers']
    for key, report in reports.items():
        assert report['anchor_sha256'] == first['anchor_sha256']
        assert report['metrics_order'] == first['metrics_order']
        assert report['rows'] == 512 and report['mothers'] == 124
        assert report['repeated_prediction_bit_exact']
        assert np.array_equal(arrays[key]['counts'], counts)
        assert np.array_equal(arrays[key]['mothers'], mothers)
        assert np.isfinite(arrays[key]['sums']).all()
        np.testing.assert_allclose(arrays[key]['sums'].sum(0) / counts.sum(0), report['means_mm'], rtol=1e-12)
        if key.endswith('_b'):
            assert report['first_batch_A_bridge_bit_exact']
            assert report['noncondition_fields_bit_exact_to_A']
            assert not report['deployment_legal']
    contrasts = {}
    for label, new, old in (
        ('init_B_minus_A', 'init_b', 'init_a'),
        ('trained_B_minus_A', 'trained_b', 'trained_a'),
        ('trained_minus_init_at_A', 'trained_a', 'init_a'),
        ('trained_minus_init_at_B', 'trained_b', 'init_b'),
        ('trained_A_minus_init_B', 'trained_a', 'init_b'),
    ):
        sums = arrays[new]['sums'] - arrays[old]['sums']
        delta = sums.sum(0) / counts.sum(0)
        ci = np.array(cluster_ci(sums, counts, mothers))
        base = np.array(reports[old]['means_mm'])
        contrasts[label] = dict(delta_mm=delta.tolist(), ci95_mm=ci.tolist(),
            relative_improvement_percent=(-100 * delta / base).tolist(),
            clearly_better=((ci[:, 1] < 0) & (-delta / base >= .02)).tolist(),
            equivalent_within_two_percent=((ci[:, 0] / base >= -.02) & (ci[:, 1] / base <= .02)).tolist())
    interaction = ((arrays['trained_b']['sums'] - arrays['trained_a']['sums'])
                   - (arrays['init_b']['sums'] - arrays['init_a']['sums']))
    contrasts['condition_by_checkpoint_interaction'] = dict(
        delta_mm=(interaction.sum(0) / counts.sum(0)).tolist(),
        ci95_mm=cluster_ci(interaction, counts, mothers))
    summary = dict(schema='ref2dex.tau-condition-attribution.v1',
        experiment_id='P-20261023-tau-fixed-panel-metric', run_id='tau-condition-ab-20261023-r1',
        scope='Fixed TRAIN anchors only; B uses future observations and is diagnostic, never deployment legal',
        metrics_order=first['metrics_order'], rows=512, mothers=124, bootstrap_repeats=4000,
        bootstrap_seed=170, counts=counts.sum(0).tolist(), cells=reports, contrasts=contrasts,
        limitations=['B jointly changes endpoint, goal token validity, and legacy event conditions',
                     'Training jointly changed conditions, data selection, and optimizer updates',
                     'No endpoint-only causal attribution, heldout or execution claim'])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(contrasts, indent=2))


if __name__ == '__main__':
    main()
