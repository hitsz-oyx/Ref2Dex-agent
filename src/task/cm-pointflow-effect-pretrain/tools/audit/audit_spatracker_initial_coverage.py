#!/usr/bin/env python3
"""Initial object-query coverage, before treating original tracks as a corpus."""
import argparse
import hashlib
import json
import subprocess
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initial_queries(coords, w2c, intrinsic, mask):
    """Deduplicate initial observations only; do not use future tracks/masks."""
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.isfinite(coords).all():
        raise ValueError('initial coordinate schema differs')
    camera = coords @ w2c[:3, :3].T + w2c[:3, 3]
    if (camera[:, 2] <= 0).any():
        raise ValueError('initial query behind camera')
    p = camera @ intrinsic.T
    pixels = p[:, :2]/p[:, 2:]
    ui, vi = np.rint(pixels).astype('int64').T
    inside = (ui >= 0)&(ui < mask.shape[1])&(vi >= 0)&(vi < mask.shape[0])
    membership = np.zeros(len(coords), bool)
    membership[inside] = mask[vi[inside],ui[inside]]
    unique_indices = np.sort(np.unique(coords,axis=0,return_index=True)[1])
    return dict(original_queries=len(coords), unique_initial_queries=len(unique_indices),
        initial_exact_duplicates=len(coords)-len(unique_indices),
        object_query_columns=int(membership.sum()),
        unique_initial_object_queries=int(membership[unique_indices].sum()),
        unique_object_query_indices=unique_indices[membership[unique_indices]].tolist(),
        projected_outside_image=int((~inside).sum()),
        initial_mask_fraction=float(mask.mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('output collision')
    m = json.loads((args.data/'manifest.json').read_text())
    started, records = time.monotonic(), []
    for e in m['sequences']:
        track = args.raw/e['scene']/'spatracker.npz'
        mask_name = next(n for n in e['source_sha256'] if n.startswith('objects/'))
        mask_path = args.raw/e['scene']/mask_name
        if sha(track) != e['source_sha256']['spatracker.npz'] or sha(mask_path) != e['source_sha256'][mask_name]:
            raise ValueError('original track/mask source drift')
        with np.load(track,allow_pickle=False) as z:
            coords,ex,k = z['coords'],z['extrinsics'][0],z['intrinsics'][0]
        if not np.allclose(ex,np.eye(4),atol=1e-6):
            raise ValueError('initial camera differs: no unqualified later-frame coordinate assumption')
        with zipfile.ZipFile(track) as archive,archive.open('depths.npy') as source:
            version = np.lib.format.read_magic(source)
            shape,_,_ = (np.lib.format.read_array_header_1_0(source) if version==(1,0)
                         else np.lib.format.read_array_header_2_0(source))
        h,w=shape[1:]
        with np.load(mask_path,allow_pickle=False) as z:
            first_mask = np.asarray(z['0'])
        while first_mask.ndim > 2 and first_mask.shape[0]==1:
            first_mask=first_mask[0]
        if first_mask.ndim!=2:
            raise ValueError('object mask shape differs')
        mask=cv2.resize(first_mask.astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST)>0
        census=initial_queries(coords[0],ex,k,mask)
        # Future trajectory equality is diagnostic only, not the dedup selector.
        census['unique_full_trajectories_diagnostic_only']=len(np.unique(coords.transpose(1,0,2).reshape(coords.shape[1],-1),axis=0))
        records.append(dict(scene=e['scene'],split=e['split'],**census))
        if time.monotonic()-started>30:
            raise TimeoutError('30s CPU cap')
    coverage={split:sum(r['split']==split and r['unique_initial_object_queries']>=16 for r in records)
              for split in ('train','dev')}
    result=dict(schema='ref2dex.spattracker-initial-coverage.v1',records=records,
        initial_16_unique_object_supported_clips=coverage,
        decision_signal='INITIAL_QUERY_COVERAGE_PRESENT' if min(coverage.values())>=2 else 'INSUFFICIENT_INITIAL_QUERY_COVERAGE',
        data_manifest_sha256=sha(args.data/'manifest.json'),script_sha256=sha(__file__),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        elapsed_seconds=time.monotonic()-started,
        limitations=['initial-mask coverage is only an upper bound for initial-object trajectories, not H4/h8/h24 visibility qualification',
            'actual coordinates projected, no assumption of100 unique uniform-grid identities',
            'initial exact duplicates removed using ONLY initial coordinates; future equality is diagnostic',
            'does not qualify raw visibs, future-derived tracking inputs, physical world units or object GT',
            'no later-born tracks, new masks, conversion, training or native corpus registration'])
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
