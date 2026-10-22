"""Bind the frozen curated tuple set without changing sampling or goal intent."""
from pathlib import Path

import numpy as np

from .curated_sampling import CuratedTauSampler, load_manifest


def bind_curated_index(dataset, path):
    manifest=load_manifest(path)
    rows=manifest['windows']
    if dataset.split!='train' or len(rows)!=6000:
        raise ValueError('curated tau binding requires all 6000 TRAIN rows')
    by_key={row['key']:i for i,row in enumerate(dataset.meta['sequences'])}
    original={tuple(map(int,row)):i for i,row in enumerate(dataset.index)}
    positions=[];keys=[]
    for row in rows:
        if row['split']!='train' or not row['admitted']:
            raise ValueError('curated tuple set must contain admitted TRAIN only')
        sequence_id=by_key[row['sequence_key']]
        if int(dataset.meta['sequences'][sequence_id]['source_id'])!=int(row['source_id']):
            raise ValueError('curated source binding changed')
        # Existing sample_id is part of the frozen identity, not a new ordinal.
        if sequence_id!=int(row['sequence_id']):
            raise ValueError('parent sequence order changed frozen sample_id')
        key=(sequence_id,int(row['slot']),int(row['tick']))
        positions.append(original[key]);keys.append(key)
    if len(set(keys))!=6000:
        raise ValueError('duplicate curated sample identity')
    dataset.index=np.asarray(dataset.index[positions]).copy()
    sources=np.asarray([row['source_id'] for row in rows])
    dataset.pools={str(source):np.flatnonzero(sources==source) for source in sorted(set(sources))}
    dataset.stratum_pools={}
    dataset.curated_manifest=Path(path).resolve()
    dataset.curated_sampler=CuratedTauSampler(manifest,seed=170,max_per_sequence=8)
    dataset.curated_positions={key:i for i,key in enumerate(keys)}


def draw_curated(dataset,count,seed,step,rank):
    if int(seed)!=170:
        raise ValueError('frozen curated sampler seed is 170')
    rows=dataset.curated_sampler.draw(count,step=step,rank=rank)
    return [(dataset.curated_positions[(int(row['sequence_id']),int(row['slot']),int(row['tick']))],False)
            for row in rows]
