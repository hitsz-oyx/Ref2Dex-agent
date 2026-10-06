#!/usr/bin/env python3
"""Sequence separation, object overlap and processed input integrity before training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    a = p.parse_args(); root = a.data / 'processed'
    m = json.loads((root / 'manifest.json').read_text())
    groups = {k: set(v) for k, v in m['split_sequences'].items()}
    assert not groups['train'] & groups['val'] and not groups['train'] & groups['test'] and not groups['val'] & groups['test']
    objects, scene_prefix, action_prefix, stats = {}, {}, {}, {}
    for split, sequences in groups.items():
        objects[split], scene_prefix[split], action_prefix[split] = set(), set(), set()
        rows = np.load(root / ('index_' + split + '.npy'), mmap_mode='r')
        assert set(m['sequences'][int(i)] for i in np.unique(rows[:, 0])).issubset(sequences)
        assert np.bincount(rows[:, 3], minlength=3).tolist() == m['statistics'][split]['strata']
        stats[split] = m['statistics'][split]
        for seq in sequences:
            meta = json.loads((root / 'sequences' / seq / 'meta.json').read_text())
            objects[split].update(meta['objects'])
            scene_prefix[split].add(seq.split('__')[0])
            action_prefix[split].add(seq.split('__')[1].split('++')[0])
    checksums = {str(f.relative_to(root)): sha(f) for f in sorted(root.rglob('*'))
                 if f.is_file() and f.suffix in ('.npy', '.npz', '.json') and f.name not in ('progress.json', 'integrity.json', 'split_audit.json')}
    report = dict(sequence_disjoint=True, statistics=stats,
                  overlapping_object_ids={k: sorted(objects['train'] & objects[k]) for k in ('val','test')},
                  overlapping_scene_prefixes={k: sorted(scene_prefix['train'] & scene_prefix[k]) for k in ('val','test')},
                  overlapping_action_prefixes={k: sorted(action_prefix['train'] & action_prefix[k]) for k in ('val','test')},
                  limitation='Sequence holdout only; shared object/scene/action prefixes are not unseen-object/subject generalization.',
                  processed_checksums=checksums)
    (root / 'split_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'processed_checksums'}))


if __name__ == '__main__': main()
