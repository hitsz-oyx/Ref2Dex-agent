#!/usr/bin/env python3
"""Freeze unique val windows with no temporal frame overlap with prior Task panels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unseen_frames(rows, previous):
    """A window spans tick-3..tick+24; compare across all object anchors."""
    good = np.ones(len(rows), dtype=bool)
    for seq in np.unique(previous[:, 0]):
        ticks = np.sort(previous[previous[:, 0] == seq, 2])
        ids = np.flatnonzero(rows[:, 0] == seq)
        current = rows[ids, 2]
        pos = np.searchsorted(ticks, current)
        left = np.abs(current - ticks[np.clip(pos - 1, 0, len(ticks) - 1)])
        right = np.abs(current - ticks[np.clip(pos, 0, len(ticks) - 1)])
        good[ids] = np.minimum(left, right) >= 28
    return good


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    meta = json.loads((args.data / 'processed/manifest.json').read_text())
    names = [s['name'] for s in meta['sources']]
    if names != ['oakink2', 'grab', 'arctic']:
        raise ValueError('expected frozen main-three source order')
    source_rows = []
    for source in meta['sources']:
        index = source['indices']['val']
        path = Path(index['path'])
        if digest(path) != index['sha256']:
            raise ValueError('source index drift')
        source_rows.append(np.load(path))
    offsets = np.cumsum([0] + [len(a) for a in source_rows])
    history, seen, skipped = [], [], []
    for path in sorted(args.history_root.rglob('validation_balanced.npy')):
        manifest_path = path.parent / 'input_manifest.json'
        if not manifest_path.exists() and (path.parent / 'panel.json').exists():
            # A generated but unused panel is not an observed model evaluation.
            skipped.append(str(path))
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('sources') != names or manifest.get('stats_sha256') != digest(args.stats):
            skipped.append(str(path))
            continue
        values = np.load(path)
        if values.ndim != 1 or values.min() < 0 or values.max() >= offsets[-1]:
            raise ValueError('historical panel contract mismatch')
        seen.extend(values.tolist())
        history.append(dict(path=str(path.resolve()), sha256=digest(path)))
    seen = np.unique(seen).astype(np.int64)
    rng = np.random.default_rng(args.seed)
    chosen, counts = [], {}
    for i, (source, rows) in enumerate(zip(meta['sources'], source_rows)):
        local_seen = seen[(seen >= offsets[i]) & (seen < offsets[i + 1])] - offsets[i]
        eligible = np.flatnonzero(unseen_frames(rows, rows[local_seen]))
        # Match the established source-internal moving/static/program stratification.
        categories = rows[:, 3].copy()
        if source['kind'] == 'native':
            categories -= 1
        groups = [eligible[categories[eligible] == k] for k in range(3)]
        probabilities = np.array([.6, .2, .2]) * np.array([len(g) > 0 for g in groups])
        probabilities /= probabilities.sum()
        desired = rng.multinomial(args.per_source, probabilities)
        picked = []
        for n, group in zip(desired, groups):
            if n > len(group):
                raise ValueError('insufficient eligible stratum; do not adapt the frozen design')
            picked.extend(rng.choice(group, n, replace=False).tolist())
        picked = np.array(picked, dtype=np.int64)
        if not unseen_frames(rows[picked], rows[local_seen]).all():
            raise AssertionError('temporal exclusion failed')
        chosen.extend((picked + offsets[i]).tolist())
        counts[source['name']] = dict(eligible_windows=len(eligible), selected_windows=len(picked),
                                     selected_sequences=len(np.unique(rows[picked, 0])),
                                     selected_categories=desired.tolist())
    chosen = np.array(chosen, dtype=np.int64)
    rng.shuffle(chosen)
    if len(np.unique(chosen)) != len(chosen):
        raise AssertionError('duplicate held-out window')
    args.output.mkdir(parents=True)
    path = args.output / 'validation_balanced.npy'
    np.save(path, chosen)
    panel = dict(schema='goal-point-wam.frozen-val-panel.v1', seed=args.seed,
                 data_manifest_sha256=digest(args.data / 'processed/manifest.json'),
                 stats_sha256=digest(args.stats), validation_indices_sha256=digest(path),
                 history=history, incompatible_history_skipped=skipped, counts=counts,
                 exclusion='same source/sequence, any anchor, abs(tick difference)>=28',
                 scope='new temporal windows on existing val sequences; not unseen subjects or PW selection data',
                 within_panel_windows_may_overlap=True, no_test_access=True)
    (args.output / 'panel.json').write_text(json.dumps(panel, indent=2) + '\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'stats', 'history-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--per-source', type=int, default=512)
    run(parser.parse_args())
