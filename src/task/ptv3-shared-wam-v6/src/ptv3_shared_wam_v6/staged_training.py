"""Explicit, separately admitted NAS stages; existing recipes remain unchanged."""
from collections import OrderedDict, Counter
from pathlib import Path
import json

import numpy as np

from .prepared_reader import PreparedWindows,sha256


class StagedWindows(PreparedWindows):
    def __init__(self, manifest, split, geometry_manifest, expected_sha256,
                 geometry_gt_object=True, cache_size=2):
        super().__init__(manifest,split,geometry_manifest,expected_sha256,geometry_gt_object,cache_size)
        self.phase=np.load(self.path.parent/(split+'-phase.npy'),mmap_mode='r')
        expected=self.manifest['files'][split+'-phase.npy']
        if sha256(self.path.parent/(split+'-phase.npy'))!=expected['sha256']:
            raise ValueError('staged phase index changed')
        self.source_mothers={s:sorted(r['sequence_id'] for r in self.rows.values()
            if r['split']==split and r['source_id']==s) for s in sorted({r['source_id'] for r in self.rows.values()})}
        self.source_mothers={s:v for s,v in self.source_mothers.items() if v}
        self.phase_cache=OrderedDict()

    def draw(self,count,seed,step=0,rank=0):
        if self.split!='train':raise ValueError('TRAIN sampling required')
        rng=np.random.default_rng(np.random.SeedSequence([seed,step,rank,20261023]))
        sources=sorted(self.source_mothers);chosen=rng.choice(sources,count)
        counts=Counter();result=[]
        for source in chosen:
            mothers=self.source_mothers[int(source)]
            available=[s for s in mothers if counts[s]<8]
            if not available:raise ValueError('source exhausted per-mother batch cap')
            mother=int(rng.choice(available));counts[mother]+=1
            row=self.rows[mother];start=row['normalized_offset'];n=row['admitted_windows']
            if mother not in self.phase_cache:
                phase=self.phase[start:start+n]
                pools=[np.flatnonzero(phase==p) for p in range(4)]
                pools=[p for p in pools if len(p)] or [np.arange(n)]
                self.phase_cache[mother]=pools
                while len(self.phase_cache)>32:self.phase_cache.popitem(last=False)
            self.phase_cache.move_to_end(mother);pools=self.phase_cache[mother]
            pool=pools[int(rng.integers(len(pools)))];result.append(start+int(rng.choice(pool)))
        return result


def validate_admission(recipe, smoke=False):
    if recipe.get('schema')!='ref2dex.remote-staged-training.v1':
        raise ValueError('explicit NAS staged recipe required')
    for field in ('manifest','stats','panel_manifest','config'):
        if sha256(recipe[field])!=recipe[field+'_sha256']:
            raise ValueError('staged binding changed: '+field)
    gates=json.loads(Path(recipe['data_checks']).read_text())
    if (gates['manifest_sha256']!=recipe['manifest_sha256']
            or gates['status']!='DATA_CHECKS_PASS_E2E_REQUIRED'):
        raise ValueError('independent data gates required')
    if not smoke:
        admission=json.loads(Path(recipe['admission']).read_text())
        if (admission.get('status')!='EIGHT_GATES_PASS_RESTRICTED_SCOPE'
                or admission.get('recipe_sha256')!=sha256(recipe['recipe_path'])
                or admission.get('training_allowed') is not True):
            raise ValueError('eight gates have not admitted this recipe')
    if (recipe['source_probabilities']!={str(s):1/9 for s in (0,1,2,4,5,6,8,9,10)}
            or recipe['world_size']!=4 or recipe['local_batch']!=96):
        raise ValueError('frozen world4/local96/global384 and equal-source policy required')
    return recipe
