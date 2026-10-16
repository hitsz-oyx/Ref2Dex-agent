"""Plot recorded physical tracking, without smoothing or excluding failed sequences."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    report = json.loads((a.run / 'audit/results.json').read_text())
    names = list(report['sequence_results'])
    roles = list(report['macro'])
    colors = ['#555555', '#d1495b', '#198754', '#2874b2']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for axis, name in zip(axes.flat, names):
        data = np.load(a.run / name / 'trajectory.npz')
        errors = np.linalg.norm(data['obj'][1:, :, :3, 3] - data['reference_object'][1:], axis=-1) * 1000
        for row, role in enumerate(roles):
            axis.plot(np.arange(1, len(errors) + 1) / 30, errors[:, row], color=colors[row], label=role, lw=1.5)
        axis.set(title=name, xlabel='Time (s)', yscale='log', ylim=(.5, 10000))
        axis.grid(alpha=.2)
    for axis in axes[:, 0]:
        axis.set_ylabel('Object position error (mm, log scale)')
    axes[0, 0].legend(fontsize=9)
    macro = report['macro']
    summary = 'Sequence-macro mean (mm): ' + ' | '.join('%s %.1f' % (r, macro[r]['object_mean_mm']) for r in roles)
    fig.suptitle('Scratch BC: state coverage improves native tracking; one drink sequence still fails', fontsize=13)
    fig.text(.5, .035, summary, ha='center', fontsize=10)
    fig.text(.5, .01, 'Six fixed VAL sequences, one reset each, full teacher reference; bounded prefixes, not full-task success.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .065, 1, .95])
    out = a.run / 'summary'
    out.mkdir(exist_ok=False)
    fig.savefig(out / 'tracking.png', dpi=170)
    fig.savefig(out / 'tracking.pdf')


if __name__ == '__main__':
    main()
