"""Explicit diagnostic source weights and curated support; production defaults untouched."""
from collections import Counter,OrderedDict
import numpy as np
from ptv3_shared_wam_v6.staged_training import StagedWindows

class DiagnosticWindows(StagedWindows):
 def __init__(self,manifest,split,geometry_manifest,expected_sha256,geometry_gt_object=True,cache_size=2,probabilities=None,curated=None):
  super().__init__(manifest,split,geometry_manifest,expected_sha256,geometry_gt_object,cache_size)
  self.probabilities=probabilities;self.curated=curated
  if curated:
   import json
   meta=json.load(open(curated)); self.curated_pools={}
   by_key={r['key']:r for r in self.rows.values()};pools={}
   for w in meta.get('windows',meta.get('selection',{}).get('train_window_rows',[])):
    r=by_key[w['sequence_key']]; sid=r['sequence_id'];start=r['normalized_offset'];n=r['admitted_windows']
    matches=np.flatnonzero((self.index[start:start+n,1]==w['slot'])&(self.index[start:start+n,2]==w['tick']))
    if len(matches)!=1:raise ValueError('curated tuple absent or duplicate')
    pools.setdefault(int(w['source_id']),{}).setdefault(sid,{}).setdefault(str(w['phase']),[]).append(start+int(matches[0]))
   assert sum(len(v) for m in pools.values() for ps in m.values() for v in ps.values())==6000
   self.curated_pools=pools
 def __getstate__(self):
  return dict(super().__getstate__(),probabilities=self.probabilities,curated=self.curated)
 def draw(self,count,seed,step=0,rank=0):
  rng=np.random.default_rng(np.random.SeedSequence([seed,step,rank,20261023]));sources=sorted(map(int,self.probabilities));p=np.array([self.probabilities[str(s)] for s in sources]);p=p/p.sum()
  chosen=rng.choice(sources,count,p=p);counts=Counter();result=[]
  for source in chosen:
   mothers=sorted(self.curated_pools[int(source)]) if self.curated else self.source_mothers[int(source)]
   available=[s for s in mothers if counts[s]<8]
   if not available:raise ValueError('per-mother cap exhausted')
   mother=int(rng.choice(available));counts[mother]+=1
   if self.curated:
    phases=self.curated_pools[int(source)][mother];keys=[p for p in ('approach','close','contact','hold') if p in phases] or ['unknown'];pool=phases[keys[int(rng.integers(len(keys)))]];result.append(int(rng.choice(pool)));continue
   row=self.rows[mother];start=row['normalized_offset'];n=row['admitted_windows']
   if mother not in self.phase_cache:
    phase=self.phase[start:start+n];pools=[np.flatnonzero(phase==p) for p in range(4)];self.phase_cache[mother]=[p for p in pools if len(p)] or [np.arange(n)]
    while len(self.phase_cache)>32:self.phase_cache.popitem(last=False)
   self.phase_cache.move_to_end(mother);pools=self.phase_cache[mother];pool=pools[int(rng.integers(len(pools)))];result.append(start+int(rng.choice(pool)))
  return result

def validate_diagnosis(recipe,smoke=False):
 from pathlib import Path
 from remote_staged_data import sha,read
 assert recipe['schema']=='ref2dex.diagnostic-training.v1'
 for field in ('manifest','stats','panel_manifest','config'):
  assert sha(recipe[field])==recipe[field+'_sha256'],field
 assert recipe['initialization_sha256']=='3f903dea22402c3f8ac7e13131ae322793bcb8f03331b82373f676a126ff092d'
 assert recipe['last_step']==500 and recipe['max_seconds']==2400 and recipe['local_batch']==96
 assert recipe['world_size'] in (1,4) and recipe['automatic_restart'] is False
 c=read(recipe['config']);assert c['goal_input_policy']=='v6.no-goal.no-event.unknown-invalid.frame30.v1';assert c['event_input_policy']=='v6.event-free.unknown-invalid.frame30.v1'
 assert c['flow_lr']==1e-4 and c['shared_lr']==1e-5 and c['warmup_updates']==50 and c['weight_decay']==.01 and c['clip_grad']==1.
 assert all(float(v)>=0 for v in recipe['source_probabilities'].values());assert abs(sum(recipe['source_probabilities'].values())-1)<1e-10
 if recipe.get('curated_manifest'):assert sha(recipe['curated_manifest'])==recipe['curated_sha256']
 return recipe
