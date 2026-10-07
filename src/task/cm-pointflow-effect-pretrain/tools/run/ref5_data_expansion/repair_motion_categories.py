"""Reindex full-horizon motion categories without rerunning native reconstruction.

Create a fresh pack, linking immutable arrays/canonical geometry to the original
pack and writing new indices and sequence metadata. Keep the source pack intact.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

import numpy as np

from native_data import eligible_rows, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    source = a.input.resolve()/'processed'
    output = a.output.resolve()/'processed'
    if a.output.resolve().exists():
        raise FileExistsError('use a fresh destination; retain all previous evidence')
    meta = json.loads((source/'manifest.json').read_text())
    if meta['schema'] != 'ref2dex.native-wm30.v1' or meta['status'] != 'COMPLETED':
        raise ValueError('expected a completed native pack')
    old = {s:np.load(source/f'index_{s}.npy') for s in ['train','val','test']}
    new = {s:x.copy() for s,x in old.items()}
    records = {x['sequence']:dict(x) for x in meta['records']}
    changed = []
    # Compute and validate everything before creating the output pack.
    for seqid,name in enumerate(meta['sequences']):
        directory = source/'sequences'/name
        arrays = {k:np.load(directory/f'{k}.npy',mmap_mode='r') for k in
                  ['hand','hand_valid','poses','pose_valid','near','timestamps','centers']}
        recalculated = eligible_rows(**arrays)
        count = 0
        moving = 0
        for split,rows in old.items():
            positions = np.flatnonzero(rows[:,0]==seqid)
            current = rows[positions,1:]
            if not np.array_equal(current[:,:2],recalculated[:,:2]) and len(current):
                raise ValueError('window identity/eligibility drift: '+name)
            if not len(current):
                continue
            count += len(current)
            moving += int((recalculated[:,2]==1).sum())
            new[split][positions,3]=recalculated[:,2]
            for pos,before,after in zip(positions,current[:,2],recalculated[:,2]):
                if before != after:
                    changed.append(dict(split=split,row=int(pos),sequence=name,
                                        anchor=int(rows[pos,1]),tick=int(rows[pos,2]),
                                        old_category=int(before),new_category=int(after)))
        if count != len(recalculated):
            raise ValueError('sequence missing from split index: '+name)
        records[name]['motion_windows']=moving
    output.mkdir(parents=True)
    (output/'sequences').mkdir()
    (output/'canonical').symlink_to(source/'canonical',target_is_directory=True)
    linked = {}
    for name in meta['sequences']:
        destination=output/'sequences'/name
        destination.mkdir()
        for f in (source/'sequences'/name).glob('*.npy'):
            (destination/f.name).symlink_to(f)
            linked[str(f)]=sha(f)
        (destination/'meta.json').write_text(json.dumps(records[name],indent=2)+'\n')
    for split,rows in new.items():
        np.save(output/f'index_{split}.npy',rows)
    meta['records']=[records[name] for name in meta['sequences']]
    meta['motion_category_rule']=dict(future_frames=24,reference='current pose',
                                      translation_threshold_m=.002,rotation_threshold_rad=.02,
                                      native_category_ids={'moving_near_hand':1,'static_near_hand':2})
    meta['category_repair']=dict(source_pack=str(a.input.resolve()),
                                 source_manifest_sha256=sha(source/'manifest.json'),
                                 script_sha256=sha(Path(__file__)),changed_windows=len(changed))
    (output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    report=dict(status='COMPLETED',changed_windows=len(changed),changes=changed,
                input_index_sha256={s:sha(source/f'index_{s}.npy') for s in old},
                output_index_sha256={s:sha(output/f'index_{s}.npy') for s in new},
                linked_unchanged_array_sha256=linked,
                category_counts={s:np.bincount(x[:,3],minlength=3).tolist() for s,x in new.items()},
                window_identities_preserved=True,pose_hand_geometry_arrays_reconstructed=False)
    (a.output/'category_repair.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','changed_windows','category_counts']}),flush=True)


if __name__ == '__main__':
    main()
