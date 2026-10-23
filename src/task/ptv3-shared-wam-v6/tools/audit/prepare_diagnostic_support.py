"""Rebind brief31 tuples through brief32 sequence identities; audit C5 whole-window validity."""
import sys,json,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/ptv3-shared-wam-v6';sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/run')]
from remote_staged_data import read,write,sha
D=ROOT/'tmp/rh';OUT=D/'diagnosis';CENSUS=OUT/'full-census'
def main():
 before=time.monotonic();rows=[json.loads(l) for l in (CENSUS/'sequences.jsonl').open()];bykey={r['key']:r for r in rows};index=np.load(CENSUS/'train-all.npy',mmap_mode='r');phase=np.load(CENSUS/'train-phase.npy',mmap_mode='r')
 original=ROOT/'src/task/ptv3-shared-wam-v6/docs/audit/brief31-deployability-census-20261023.json';inventory=ROOT/'src/task/ptv3-shared-wam-v6/docs/audit/brief32-raw-signal-inventory-20261023.json';inv={r['sequence_id']:r for r in read(inventory)['sequence_inventory']};tuples=read(original)['unproven_windows'];matched=[];missing=[];mapping={};by_source=Counter()
 for w in tuples:
  s,old,slot,tick=map(int,w['window_tuple']);identity=inv.get(old);r=bykey.get(identity['key']) if identity else None
  if r is None or r['source_id']!=s or r['split']!='train':missing.append(dict(tuple=w['window_tuple'],reason='sequence/source binding absent'));continue
  start=r['normalized_offset'];end=start+r['admitted_windows'];ids=np.flatnonzero((index[start:end,1]==slot)&(index[start:end,2]==tick))
  if len(ids)!=1:missing.append(dict(tuple=w['window_tuple'],key=r['key'],reason='tuple absent or duplicate'));continue
  pos=start+int(ids[0]);matched.append(dict(source_id=s,sequence_id=r['sequence_id'],original_sequence_id=old,sequence_key=r['key'],slot=slot,tick=tick,phase=('approach','close','contact','hold','unknown')[int(phase[pos])],split='train',admitted=True,diagnostic_index=pos));by_source[s]+=1;mapping[old]=dict(key=r['key'],remote_sequence_id=r['sequence_id'],source_id=s)
 rate=len(matched)/len(tuples);result=dict(status='MATCHED' if rate>=.95 else 'NOT_REPRODUCIBLE',matched=len(matched),requested=len(tuples),match_fraction=rate,by_source=dict(by_source),mapping=mapping,missing=missing,identity_method='brief31 original sequence_id -> brief32 sequence_inventory.key and source_id -> converted census stable key; never equate unrelated numeric sequence IDs',brief31_sha256=sha(original),brief32_sha256=sha(inventory),data_sha256=sha(CENSUS/'manifest.json'),elapsed_s=time.monotonic()-before)
 write(OUT/'C1-support-match.json',result,immutable=True)
 if rate>=.95:write(OUT/'curated-rebound.json',dict(schema='ref2dex.diagnostic-curated-support.v1',windows=matched,match_fraction=rate,source_counts=dict(by_source)),immutable=True)
 print(json.dumps({k:v for k,v in result.items() if k not in ('mapping','missing')},indent=2),flush=True)
 # All-source exact whole-window hand validity gate, preserving full index positions.
 # Count all24-step actual trajectory masks; missing other-hand slots never reject a window.
 valid=np.ones(len(index),bool);counts=defaultdict(Counter)
 for number,r in enumerate(x for x in rows if x['split']=='train'):
  start=r['normalized_offset'];end=start+r['admitted_windows'];ticks=np.asarray(index[start:end,2],np.int64);j=np.load(Path(r['path'])/'joint_valid.npy',mmap_mode='r');hv=j[ticks].copy()
  for lag in (1,2,3):hv &=j[ticks-lag]
  any24=np.zeros(len(ticks),bool);any8=np.zeros(len(ticks),bool);points=0
  for future in range(1,25):
   mask=hv&j[ticks+future];any24|=mask.any((1,2));points+=int(mask.sum())
   if future<=8:any8|=mask.any((1,2))
  valid[start:end]=any24;s=r['source_id'];counts[s].update(windows=len(ticks),valid24=int(any24.sum()),zero24=int((~any24).sum()),zero8=int((~any8).sum()),valid_point_frames24=points,total_point_frames24=len(ticks)*24*42)
  if number%3000==0:write(OUT/'validity-audit-progress.json',dict(mothers=number,elapsed_s=time.monotonic()-before))
 np.save(OUT/'c5-valid-windows.npy',valid)
 result=dict(status='PASS',gate='any((history_H4_valid & future_hand_joint_valid)[future1:24,hand2,joint21])',gate_excludes='whole windows with zero valid hand target point-frames; does NOT require two hands or 100% point coverage',counts={str(s):dict(v) for s,v in counts.items()},total_windows=len(index),kept_windows=int(valid.sum()),rejected_windows=int((~valid).sum()),gate_sha256=sha(OUT/'c5-valid-windows.npy'),parent_manifest_sha256=sha(CENSUS/'manifest.json'),elapsed_s=time.monotonic()-before)
 write(OUT/'C5-validity-audit.json',result,immutable=True);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
