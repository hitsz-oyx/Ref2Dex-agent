"""Plot original-start evaluation and contact acquisition, excluding inherited hold."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def rows(run):
    path = run / 'monitor.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', type=Path, required=True)
    parser.add_argument('--b', type=Path, required=True)
    parser.add_argument('--b-continuation', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4), constrained_layout=True)
    for label, run, color in (('A: original', args.a, '#4c78a8'), ('B: contact', args.b, '#e45756')):
        runs = [run]
        if label.startswith('B') and args.b_continuation is not None:
            runs.append(args.b_continuation)
        evaluation = [json.loads(path.read_text()) for source in runs for path in sorted(source.glob('eval[0-9]*.json'))]
        if evaluation:
            axes[0].plot([v['update'] for v in evaluation],
                         [max(o['maximum_held_frames'] for o in v['outcomes']) for v in evaluation],
                         'o-', label=label, color=color)
            axes[1].plot([v['update'] for v in evaluation], [1000*v['close_rise_m'] for v in evaluation],
                         'o-', label=label, color=color)
        data = [row for source in runs for row in rows(source)]
        phases = [48] if label.startswith('A') else [48, 50, 52]
        for phase in phases:
            success = [next(t['new_hold45'] for t in row['grasp_trials'] if t['phase'] == phase) for row in data]
            smooth = [float(np.mean(success[max(0, index-19):index+1])) for index in range(len(success))]
            axes[2].plot([row['update'] for row in data], smooth,
                         label=f'{label}, phase {phase}', linestyle='-' if phase == 48 else '--',
                         alpha=1. if phase == 48 else .65)
    axes[0].axhline(45, color='#888888', linestyle='--', label='hold45 threshold')
    axes[0].set(title='Frame-0 full evaluation (best of 4 copies)', ylabel='Maximum consecutive held controls')
    axes[1].set(title='Close-range rise (support may remain)', ylabel='Object-center rise (mm)')
    axes[2].set(title='New hold45 (rolling 20 trials)', ylabel='Acquisition fraction', ylim=(-.02, 1.02))
    for axis in axes:
        axis.set_xlabel('Total PPO updates')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    title = ('A/B matched budget to60; only B continued to100' if args.b_continuation is not None else
             'One paired Probe; curriculum starts differ and physics varies')
    fig.suptitle(title, fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
