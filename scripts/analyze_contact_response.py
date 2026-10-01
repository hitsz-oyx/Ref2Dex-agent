#!/usr/bin/env python3
"""Recompute fixed pulse/noise screen and export standalone scientific plots."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from src.task.CmResidual.contact_response import HORIZONS, sample_response
from scripts.run_contact_response_probe import ARMS, PANELS, sha


def norm_summary(value):
    norms = value.square().sum(-1).sqrt()
    return dict(rms_mm=float(norms.square().mean().sqrt() * 1000),
                median_mm=float(norms.median() * 1000),
                q95_mm=float(torch.quantile(norms, .95) * 1000))


def contrast_ratio(signal, noise):
    # An observed zero repeat difference gives no finite calibrated ratio.
    # Do not invent enormous ratios with an arbitrary denominator floor.
    return signal / noise if noise > 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if ROOT not in run.parents:
        raise ValueError('analysis outputs must remain in isolated worktree')
    output = args.output_dir.resolve() if args.output_dir else run
    if ROOT not in output.parents:
        raise ValueError('analysis destination outside isolated worktree')
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / 'run_manifest.json').read_text())
    if manifest['run_status'] != 'COMPLETED' or len(manifest['phases']) != 16:
        raise ValueError('requires complete fixed matrix')
    torch.set_num_threads(2)
    pooled = {h: dict(signal=[], noise=[]) for h in HORIZONS}
    rows, pre_rows, inputs = [], [], {}
    for t, seed in PANELS:
        data = {}
        for arm, amplitude in ARMS:
            directory = run / f't{t}_s{seed}_{arm}'
            result = json.loads((directory / 'results.json').read_text())
            path = directory / 'response.pt'
            if sha(path) != result['output_sha256']:
                raise ValueError('output hash drift')
            inputs[str(path)] = sha(path)
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if payload['amplitude'] != amplitude or len(payload['triggers']) != 96:
                raise ValueError('amplitude/window contract')
            data[arm] = payload
        reference = data['zero_a']
        for other in data.values():
            if other['initial_state_fingerprint'] != reference['initial_state_fingerprint']:
                raise ValueError('different cold initial conditions')
            if not torch.equal(other['triggers'], reference['triggers']) or not torch.equal(other['motion_id'], reference['motion_id']):
                raise ValueError('trigger/motion mismatch')
        triggers = reference['triggers']
        env = torch.arange(96)
        pre = {a: d['state_before'][triggers, env, 36:39] for a, d in data.items()}
        pre_rows.append(dict(panel=f't{t}_s{seed}',
                             pulse_pre_mismatch=norm_summary(pre['plus'] - pre['minus']),
                             repeat_pre_mismatch=norm_summary(pre['zero_a'] - pre['zero_b'])))
        for h in HORIZONS:
            response = {a: sample_response(d['state_before'], d['state_after'], triggers, h) for a, d in data.items()}
            signal = response['plus'][:, :3] - response['minus'][:, :3]
            noise = response['zero_a'][:, :3] - response['zero_b'][:, :3]
            pooled[h]['signal'].append(signal)
            pooled[h]['noise'].append(noise)
            signal_rms = norm_summary(signal)['rms_mm']
            noise_rms = norm_summary(noise)['rms_mm']
            row = dict(panel=f't{t}_s{seed}', horizon=h, seconds=h / 30,
                       signal_rms_mm=signal_rms, noise_rms_mm=noise_rms,
                       contrast_ratio=contrast_ratio(signal_rms, noise_rms),
                       signed_z_contrast_mm=float(signal[:, 2].mean() * 1000),
                       positive_z_fraction=float((signal[:, 2] > 0).float().mean()))
            row['by_motion'] = {str(m): dict(
                windows=int((reference['motion_id'] == m).sum()),
                signed_z_contrast_mm=float(signal[reference['motion_id'] == m, 2].mean() * 1000),
                signal_rms_mm=norm_summary(signal[reference['motion_id'] == m])['rms_mm'],
                noise_rms_mm=norm_summary(noise[reference['motion_id'] == m])['rms_mm']) for m in range(3)}
            rows.append(row)
    combined = []
    for h in HORIZONS:
        signal, noise = (torch.cat(pooled[h][key]) for key in ('signal', 'noise'))
        sr, nr = norm_summary(signal)['rms_mm'], norm_summary(noise)['rms_mm']
        combined.append(dict(horizon=h, seconds=h / 30, windows=384,
                             signal_rms_mm=sr, noise_rms_mm=nr, contrast_ratio=contrast_ratio(sr, nr),
                             signed_z_contrast_mm=float(signal[:, 2].mean() * 1000)))
    gates = {str(h): dict(
        contrast_resolved=(next(r for r in combined if r['horizon'] == h)['contrast_ratio'] or 0) >= 2,
        signed_z_positive_all_panels=all(r['signed_z_contrast_mm'] > 0 for r in rows if r['horizon'] == h)
    ) for h in (5, 10)}
    passed = all(v for gate in gates.values() for v in gate.values())
    report = dict(experiment_id=manifest['experiment_id'], run_id=manifest['run_id'],
                  run_status='COMPLETED', label='PROMISING' if passed else 'UNPROMISING',
                  scope='physical pulse/noise feasibility only; no learned-model or policy utility result',
                  gates=gates, pooled=combined, panels=rows, pre_intervention_mismatch=pre_rows,
                  inputs=inputs, analysis_source_sha256=sha(__file__),
                  limitations=['Contact force proxies are not pairwise contacts.',
                               'Schedule uses a frozen reference contact event; it is retrospective.',
                               'Warm hidden solver states cannot be cloned; actual pre-pulse mismatch is reported.',
                               'One positive and one negative run per panel; repeat variance is sparsely estimated.',
                               'Only three motions of one object and two existing actor seeds.',
                               'Fixed run order and shared simulator/GPU limit causal attribution.',
                               'No learned predictor, independent test set, task improvement or hardware evidence.',
                               'A zero observed repeat contrast is reported with a null ratio, not infinite reliability.'])
    target = output / 'analysis.json'
    if target.exists():
        raise FileExistsError('analysis already exists; do not overwrite evidence')
    target.write_text(json.dumps(report, indent=2) + '\n')
    with (output / 'horizon_summary.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    x = [r['horizon'] for r in combined]
    axes[0].plot(x, [r['signal_rms_mm'] for r in combined], 'o-', label='+1 cm vs -1 cm pulse')
    axes[0].plot(x, [r['noise_rms_mm'] for r in combined], 's-', label='Zero-pulse repeat contrast')
    axes[0].set(xlabel='Control steps after pulse', ylabel='Object response contrast RMS (mm)', xscale='log')
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(x, [r['contrast_ratio'] if r['contrast_ratio'] is not None else float('nan') for r in combined], 'o-', color='#2a7f62')
    axes[1].axhline(2, color='gray', linestyle='--', label='Predeclared screen')
    axes[1].set(xlabel='Control steps after pulse', ylabel='Pulse / repeat contrast ratio', xscale='log')
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([str(h) for h in x])
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    fig.savefig(output / 'contact_response.png', dpi=220)
    fig.savefig(output / 'contact_response.pdf')
    print(json.dumps({k: report[k] for k in ('label', 'gates', 'pooled')}, indent=2))


if __name__ == '__main__':
    main()
