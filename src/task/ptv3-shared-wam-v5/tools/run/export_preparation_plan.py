"""Freeze admitted v4 window identities for a separate sequence-level v5 overlay."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-root', type=Path, required=True)
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.code_root/'src/task/ptv3-shared-wam-v4/src'))
    from ptv3_shared_wam_v4 import paths
    from ptv3_shared_wam_v4.data import NineWindows
    recipe = json.loads((a.recipe/'recipe.json').read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'windows').mkdir()
    jobs, groups = {}, {}
    for split in ('train', 'dev'):
        ds = NineWindows(recipe['main3_root'], recipe['native_root'], split, recipe)
        for sid, source in enumerate(ds.main.base.sources):
            for seq_id in np.unique(source.rows[:, 0]):
                seq_id = int(seq_id)
                rows = np.flatnonzero(source.rows[:, 0] == seq_id)
                rows = rows[~ds.blocked[ds.main.base.offsets[sid]+rows]]
                if not len(rows):
                    continue
                name = source.sequences[seq_id]
                folder = source.root/'processed/sequences'/name
                meta = json.loads((folder/'meta.json').read_text())
                key = f's{sid}-{seq_id:05d}'
                jobs[key] = dict(key=key, source_id=sid, sequence_id=seq_id,
                    source_sequence=name, sequence_folder=str(folder), source_root=str(source.root),
                    raw_path=meta.get('source_path'), objects=meta['objects'])
                groups[(key, split)] = source.rows[rows][:, [1, 2]].astype(np.int32)
        offset = len(ds.main)
        native_rows = np.asarray(ds.native.base.index, np.int64)
        admitted = ~ds.blocked[offset:]
        for row_id in np.unique(native_rows[admitted, 0]):
            row_id = int(row_id)
            row = ds.native.base.meta['sequences'][row_id]
            sid = row['source_id']
            key = f's{sid}-{row_id:05d}'
            payload = Path(row['path'])
            if not payload.is_absolute():
                payload = Path(recipe['native_root'])/payload
            jobs[key] = dict(key=key, source_id=sid, sequence_id=row_id,
                source_sequence=row['source_sequence'], payload=str(payload),
                source_row=row, objects=row['objects'])
            ticks = native_rows[admitted & (native_rows[:, 0] == row_id), 1]
            groups[(key, split)] = np.column_stack((np.full(len(ticks), row['target_slot']), ticks)).astype(np.int32)
    for (key, split), rows in groups.items():
        rel = f'windows/{key}-{split}.npy'
        np.save(a.output/rel, rows)
        jobs[key].setdefault('windows', {})[split] = rel
        jobs[key].setdefault('window_counts', {})[split] = len(rows)
    result = dict(schema='ref2dex.wam-v5-preparation-plan.v1', base_recipe=str(a.recipe),
        main3_root=recipe['main3_root'], native_root=recipe['native_root'],
        source_probabilities=recipe['source_probabilities'], jobs=list(jobs.values()),
        scope='Only frozen admitted TRAIN/DEV windows. No held-out TEST. Raw inputs read-only.')
    (a.output/'plan.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(jobs=len(jobs), windows=sum(len(x) for x in groups.values()),
        by_source={str(s):sum(j['source_id']==s for j in jobs.values()) for s in (0,1,2,4,5,6,8,9,10)})), flush=True)


if __name__ == '__main__':
    main()
