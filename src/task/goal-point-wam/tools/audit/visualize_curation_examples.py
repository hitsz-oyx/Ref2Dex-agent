#!/usr/bin/env python3
"""Representative raw TRAIN geometry for human review, not evidence selection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_native_curation import masks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--census', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.data / 'processed/manifest.json').read_text())
    examples = []
    for desc in manifest['sources']:
        name, root = desc['name'], Path(desc['root'])
        source = json.loads((root / 'processed/manifest.json').read_text())
        with np.load(args.census / (name + '.npz')) as f:
            rows, metric = f['rows'], f['metrics']
        flags = masks(metric)
        if name == 'oakink2':
            keys = rows[:, 0] * 1000000000 + rows[:, 2]
            active = (metric[:, 4] <= 50) & (metric[:, 1] >= 5)
            alternative = np.isin(keys, keys[active])
            groups = [('Static target; another object active', flags['static_goal_far_moving'] & alternative),
                      ('Static target; idle away', flags['static_goal_idle_away'])]
        elif name == 'grab':
            groups = [('Static target; departing hands', (metric[:, 0] < 2) & (metric[:, 4] <= 50) & (metric[:, 6] > 100))]
        else:
            groups = [('Static target; near at both ends', (metric[:, 0] < 2) & (metric[:, 4] <= 50) & (metric[:, 6] <= 50))]
        for title, selected in groups:
            ids = np.flatnonzero(selected)
            index = int(ids[np.argsort(metric[ids, 2], kind='stable')[len(ids) // 2]])
            seq, anchor, tick = map(int, rows[index, :3]); seqname = source['sequences'][seq]
            path = root / 'processed/sequences' / seqname
            meta = json.loads((path / 'meta.json').read_text())
            poses = np.load(path / 'poses.npy', mmap_mode='r')
            hand = np.load(path / 'hand.npy', mmap_mode='r')[tick:tick+25].copy()
            valid = np.load(path / 'hand_valid.npy', mmap_mode='r')[tick:tick+25].copy()
            R, center = poses[tick, anchor, :3, :3], poses[tick, anchor, :3, 3]
            hh = (hand - center) @ R * 1000
            clouds = {}
            for j, obj in enumerate(meta['objects']):
                points = np.load(root / 'processed/canonical' / (obj + '.npz'))['points']
                world = points @ poses[tick, j, :3, :3].T + poses[tick, j, :3, 3]
                clouds[j] = (world - center) @ R * 1000
            other = None
            if name == 'oakink2' and title.startswith('Static target; another'):
                candidates = np.flatnonzero((rows[:, 0] == seq) & (rows[:, 2] == tick) & active)
                other = int(rows[candidates[0], 1])
            examples.append(dict(source=name, title=title, index=index, sequence=seqname,
                                 anchor=anchor, tick=tick, other=other, metric=metric[index].tolist(),
                                 hand=hh, valid=valid, clouds=clouds))
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    for i, example in enumerate(examples):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        show = [example['anchor']]
        if example['other'] is not None: show.append(example['other'])
        extents = []
        for j in show:
            points = example['clouds'][j]
            color = '#377eb8' if j == example['anchor'] else '#4daf4a'
            ax.scatter(*points[::4].T, s=3, color=color, alpha=.5,
                       label='Target object now' if j == example['anchor'] else 'Other active object now')
            extents.append(points)
        for side, color in enumerate(['#e41a1c', '#ff7f00']):
            if not example['valid'][:, side].all(): continue
            hand = example['hand'][:, side]
            ax.plot(*hand[:, 0].T, color=color, label=('Right' if side == 0 else 'Left') + ' wrist path')
            ax.scatter(*hand[0].T, s=15, color=color, marker='o')
            ax.scatter(*hand[-1].T, s=18, color=color, marker='x')
            for point in (6, 7): ax.plot(*hand[:, point].T, color=color, alpha=.35, linewidth=.6)
            extents.append(hand.reshape(-1, 3))
        xyz = np.concatenate(extents); center = (xyz.min(0) + xyz.max(0)) / 2
        radius = max(float(np.ptp(xyz, axis=0).max()) * .55, 30)
        ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
               zlim=(center[2]-radius, center[2]+radius), xlabel='x (mm)', ylabel='y (mm)', zlabel='z (mm)')
        ax.set_box_aspect((1, 1, 1)); g, _, h, _, d0, df, de, *_ = example['metric']
        ax.set_title(f"{example['source']}: {example['title']}\ng={g:.2f} mm; hand peak={h:.1f}; target distance {d0:.0f}->{de:.0f} mm", fontsize=10)
        ax.legend(fontsize=6, loc='upper left')
    fig.suptitle('Raw TRAIN examples: object goal alone does not specify hand intent\nCircles=current hands; crosses=step24 hands; fixed current-target frame', fontsize=12)
    fig.savefig(args.output / 'examples.png', dpi=150); plt.close(fig)
    records = [{key: value for key, value in e.items() if key not in ('hand', 'valid', 'clouds')} for e in examples]
    (args.output / 'examples.json').write_text(json.dumps(dict(selection='median hand-peak within each declared geometric stratum; illustration only', examples=records), indent=2))


if __name__ == '__main__': main()
