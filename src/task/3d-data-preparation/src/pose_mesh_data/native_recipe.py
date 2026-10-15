"""Explicit source -> uniform-window recipe bound to frozen native TRAIN moments."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .cleaning import CLEANING_SCHEMA
from .statistics import mix_moments

RECIPE_SCHEMA='ref2dex.native-pretrain-recipe.v1'
NATIVE_STATS_SCHEMA='ref2dex.native-horizon-stats.v1'
SOURCE_IDS=(4,5,6,8,9,10)


def source_probabilities(counts,temperature=.5,caps=None):
    if not counts or not np.isfinite(temperature) or not 0<=temperature<=1:
        raise ValueError('bounded source temperature and nonempty counts required')
    if any(type(n) is not int or n<=0 for n in counts.values()):
        raise ValueError('positive integer source window counts required')
    caps=caps or {}
    if set(caps)-set(counts) or any(not np.isfinite(v) or not 0<v<=1 for v in caps.values()):
        raise ValueError('known sources and positive probability caps required')
    raw={sid:n**temperature for sid,n in counts.items()}
    fixed={}
    while True:
        free=sorted(set(counts)-set(fixed),key=int)
        remainder=1-sum(fixed.values())
        if not free:
            if abs(remainder)>1e-10:
                raise ValueError('probability caps cannot allocate unit mass')
            return fixed
        proposed={sid:remainder*raw[sid]/sum(raw[k] for k in free) for sid in free}
        excess={sid:caps[sid] for sid,p in proposed.items() if sid in caps and p>caps[sid]+1e-14}
        if not excess:
            return dict(sorted({**fixed,**proposed}.items(),key=lambda x:int(x[0])))
        fixed.update(excess)


def recipe_for_snapshot(root,policy):
    root=Path(root).resolve()
    meta=json.loads((root/'manifest.json').read_text())
    moments=json.loads((root/'source-moments.json').read_text())
    if (meta.get('cleaning_schema')!=CLEANING_SCHEMA or meta.get('status')!='COMPLETED'
            or meta.get('training_allowed') is not False
            or meta.get('encoder_admission',{}).get('schema')!='observed-fixed-grid.v1'
            or json.loads((root/'progress.json').read_text()).get('status')!='COMPLETED'
            or moments.get('schema')!='ref2dex.native-train-moments.v1'
            or moments.get('status')!='COMPLETED' or moments.get('split')!='train'
            or moments.get('training_allowed') is not False
            or moments.get('input_manifest')!=str(root/'manifest.json')):
        raise ValueError('matching completed encoder view and TRAIN-only moments required')
    path=root/meta['window_index'];stamp=path.stat()
    if (stamp.st_size,stamp.st_mtime_ns)!=(meta['window_index_size_bytes'],meta['window_index_mtime_ns']):
        raise ValueError('frozen index changed')
    with np.load(path,allow_pickle=False) as z:
        index=z['verified_train']
        counts=Counter()
        for row,tick in index:
            r=meta['sequences'][row]
            if r['split']!='train' or r.get('clock_verified') is not True:
                raise ValueError('unverified/DEV row in TRAIN recipe')
            counts[str(r['source_id'])]+=1
    if tuple(sorted(map(int,counts)))!=SOURCE_IDS or moments['windows']!=sum(counts.values()):
        raise ValueError('six verified native TRAIN sources required')
    if set(moments['source_moments'])!=set(counts) or any(
            moments['source_moments'][sid]['windows']!=n for sid,n in counts.items()):
        raise ValueError('source moments and frozen TRAIN counts differ')
    if (policy.get('schema')!='ref2dex.native-sampling-policy.v1'
            or policy.get('permitted_use')!='engineering_only'
            or policy.get('interpolated_targets')!='include_when_valid'):
        raise ValueError('explicit bounded engineering policy required')
    weights=source_probabilities(dict(counts),policy['temperature'],policy['probability_caps'])
    binding=dict(data_manifest=str(root/'manifest.json'),window_index_size_bytes=stamp.st_size,
        window_index_mtime_ns=stamp.st_mtime_ns,source_counts=dict(counts),source_probabilities=weights)
    recipe=dict(schema=RECIPE_SCHEMA,training_allowed=False,permitted_use='engineering_only',
        sampling='source_probability_then_uniform_frozen_window.v1',**binding,policy=policy,
        source_profiles=meta['source_profiles'],encoder_workspace=meta['encoder_admission']['workspace'],
        loss='Existing global masked-point mean, hand_weight=scene_weight=1; no source loss multipliers.',
        limitations=['Native six-source engineering recipe; original main3 data is separate.',
            'Source probability is not an equal contribution guarantee under masked-point loss.',
            'No temporal/motion strata; changing within-source sampling requires matching moments.'])
    combined=mix_moments(moments['source_moments'],weights,policy['std_floor_m'])
    stats=dict(schema=NATIVE_STATS_SCHEMA,split='train',training_allowed=False,units='m',
        fps=30,history=4,horizon=24,source_ids=list(SOURCE_IDS),**binding,
        std_floor_m=policy['std_floor_m'],moment_source=str(root/'source-moments.json'),
        aggregation='Expected masked-point moments under recipe source -> uniform-window draws.',**combined)
    return recipe,stats


def source_pools(dataset):
    ids=np.asarray([dataset.meta['sequences'][row]['source_id'] for row,tick in dataset.index])
    return {str(sid):np.flatnonzero(ids==sid) for sid in sorted(np.unique(ids))}


class NativeSourceSampler:
    def __init__(self,dataset,recipe):
        self.pools=source_pools(dataset)
        counts={sid:len(pool) for sid,pool in self.pools.items()}
        if (recipe.get('schema')!=RECIPE_SCHEMA or recipe.get('training_allowed') is not False
                or recipe.get('permitted_use')!='engineering_only'
                or recipe.get('sampling')!='source_probability_then_uniform_frozen_window.v1'
                or recipe['data_manifest']!=str(dataset.root/'manifest.json')
                or counts!=recipe['source_counts'] or not dataset.verified_clock_only
                or dataset._index_stamp!=(recipe['window_index_size_bytes'],recipe['window_index_mtime_ns'])):
            raise ValueError('recipe does not match frozen verified TRAIN reader')
        if any(dataset.meta['sequences'][row]['split']!='train' for row,tick in dataset.index):
            raise ValueError('TRAIN sampler cannot consume DEV')
        self.sources=list(self.pools)
        if set(recipe['source_probabilities'])!=set(self.sources):
            raise ValueError('recipe source probability identities differ')
        self.probabilities=np.asarray([recipe['source_probabilities'][sid] for sid in self.sources],float)
        if not np.isfinite(self.probabilities).all() or (self.probabilities<=0).any() or not np.isclose(self.probabilities.sum(),1,atol=1e-12,rtol=0):
            raise ValueError('positive unit source probability mass required')

    def draw(self,count,seed,step=0,rank=0):
        if not 1<=count<=65536 or min(seed,step,rank)<0:
            raise ValueError('bounded deterministic draw required')
        rng=np.random.default_rng(np.random.SeedSequence([seed,step,rank]))
        chosen=rng.choice(len(self.sources),count,p=self.probabilities)
        result=np.empty(count,np.int64)
        for i,sid in enumerate(self.sources):
            mask=chosen==i
            result[mask]=rng.choice(self.pools[sid],int(mask.sum()))
        return result


def validate_stats_binding(recipe,stats):
    for field in ('data_manifest','window_index_size_bytes','window_index_mtime_ns','source_counts','source_probabilities'):
        if recipe.get(field)!=stats.get(field):
            raise ValueError('native statistics/recipe binding mismatch: '+field)
    if stats.get('schema')!=NATIVE_STATS_SCHEMA or stats.get('training_allowed') is not False or stats.get('split')!='train':
        raise ValueError('native engineering TRAIN statistics required')
