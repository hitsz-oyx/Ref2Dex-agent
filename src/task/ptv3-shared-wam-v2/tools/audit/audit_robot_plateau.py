"""Snapshot fixed-dev trends and rank0 training summaries without model inference."""
import argparse
import json
from pathlib import Path

import numpy as np


def run(directory, config):
    curve = json.loads((directory/'curve.json').read_text())
    training = [json.loads(line) for line in (directory/'train.jsonl').read_text().splitlines()]
    config = json.loads(config.read_text())
    dev = [dict(row, forecast=row['objective']-config['action_weight']*row['action']
                -config['anchor_weight']*row['anchor']) for row in curve]
    blocks = []
    for end in range(500, training[-1]['step']+1, 500):
        rows = [r for r in training if end-500 < r['step'] <= end]
        blocks.append(dict(end_step=end, logged_batches=len(rows),
            loss=float(np.mean([r['loss'] for r in rows])),
            **{k:float(np.mean([r['terms'][k] for r in rows])) for k in ['hand', 'scene', 'action', 'anchor']}))
    snapshot = dict(step=training[-1]['step'], dev=dev, rank0_train_500step_blocks=blocks,
        scope='fixed dev loss trend; not physical tracking/grasping validation',
        dev_evaluations=len(dev),
        last_interval_relative_change={k:dev[-1][k]/dev[-2][k]-1
            for k in ['objective', 'action', 'anchor', 'forecast']} if len(dev)>1 else {})
    stem = 'plateau-step%d'%snapshot['step']
    destination = directory/(stem+'.json')
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(snapshot, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    steps = [r['step'] for r in dev]
    for key in ['objective', 'forecast']:
        axes[0].plot(steps, [r[key] for r in dev], 'o-', label=key)
    for key in ['action', 'anchor']:
        axes[1].plot(steps, [r[key] for r in dev], 'o-', label=key)
    axes[2].plot([r['end_step'] for r in blocks], [r['loss'] for r in blocks], 'o-', label='rank0 train, block mean')
    for ax, title in zip(axes, ['Fixed dev: total / forecast', 'Fixed dev: predicted-tau / GT-tau R', 'Training samples (different panel)']):
        ax.set(xlabel='WAM joint updates', ylabel='Loss', title=title)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(directory/(stem+'.png'), dpi=160)
    plt.close(fig)
    print(json.dumps(snapshot, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    run(args.run, args.config)
