"""Read-only CPU scale audit over existing converted halves; bounded sampling."""
import os,sys,json,time,hashlib,subprocess
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[5]; TASK=ROOT/'src/task/ptv3-shared-wam-v6'
sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/run')]
from remote_staged_data import read,write,sha
from ptv3_shared_wam_v6.prepared_reader import PreparedWindows
D=Path('/mnt/ugreen_nas/storage/oyx_storage/wam-nine-source-full-20261023-pending'); OUT=D/'diagnosis'

def summary(a):
 a=np.concatenate(a).astype(np.float64); return dict(n=len(a),mean=a.mean(0).tolist(),std=a.std(0).tolist(),p1=np.percentile(a,1,axis=0).tolist(),p50=np.percentile(a,50,axis=0).tolist(),p99=np.percentile(a,99,axis=0).tolist(),rms=float(np.sqrt(np.mean(a*a))),norm_mean=float(np.linalg.norm(a,axis=-1).mean()) if a.ndim==2 else None)

def receipt(p):
 r=read(p); z=np.load(r['window_shard'],allow_pickle=False)
 return r,z['tuples'],z['phase']

def main():
 OUT.mkdir(exist_ok=True); start=time.monotonic()
 # Publish a diagnostic complete census from existing immutable conversion receipts.
 dest=OUT/'full-census';dest.mkdir(exist_ok=False)
 paths=sorted((D/'half/receipts').glob('*.json'))+sorted((D/'remaining/receipts').glob('*.json'))
 counts=Counter();phases=defaultdict(Counter);rows=[];arrays=defaultdict(list);pas=defaultdict(list)
 with ThreadPoolExecutor(12) as pool:
  for i,(r,t,p) in enumerate(pool.map(receipt,paths)):
   assert r['status']=='ADMITTED' and len(t)==r['admitted_windows']; split=r['split'];s=r['source_id']
   r['normalized_offset']=counts[split];counts[split]+=len(t);rows.append(r);arrays[split].append(t);pas[split].append(p)
   phases[(split,s)].update(map(int,p))
   if i%2000==0:write(OUT/'audit-progress.json',dict(stage='census',mothers=i,total=len(paths),elapsed_s=time.monotonic()-start))
 for split in ('train','dev'):
  np.save(dest/(split+'-all.npy'),np.concatenate(arrays[split]));np.save(dest/(split+'-phase.npy'),np.concatenate(pas[split]));np.save(dest/(split+'-clock-valid.npy'),np.ones(counts[split],bool))
 with (dest/'sequences.jsonl').open('w') as f:
  for r in rows:f.write(json.dumps(r)+'\n')
 files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in dest.iterdir()}
 sources=sorted({r['source_id'] for r in rows})
 manifest=dict(schema='ref2dex.nine-source-readiness.v1',expected_mothers=len(rows),files=files,sources=[dict(source_id=s,**{split:dict(admitted_windows=sum(r['admitted_windows'] for r in rows if r['source_id']==s and r['split']==split),mothers=sum(r['source_id']==s and r['split']==split for r in rows)) for split in ('train','dev')}) for s in sources],training_allowed=False,scope='diagnostic reader census; admission inherited from converted receipts; native clocks not re-audited',plan_sha256=sha(D/'plan.json'))
 write(dest/'manifest.json',manifest)
 ds=PreparedWindows(dest/'manifest.json','train',D.parent/'ptv3-shared-wam-v5-20261019-r1/data/manifest.json',sha(dest/'manifest.json'),geometry_gt_object=False)
 config=read(D/'half-recipe/config.json');stats=read(D/'half-recipe/stats.json');rng=np.random.default_rng(20261023);results={}
 # Uniform windows within each source, 256 per source; sort by mother for NAS locality.
 for s in sources:
  pool=[r for r in rows if r['split']=='train' and r['source_id']==s];sizes=np.array([r['admitted_windows'] for r in pool]); ends=np.cumsum(sizes)
  draws=rng.choice(int(ends[-1]),min(256,int(ends[-1])),replace=False); selected=[]
  for n in draws:
   j=int(np.searchsorted(ends,n,side='right'));r=pool[j];selected.append(r['normalized_offset']+int(n-(ends[j-1] if j else 0)))
  store=defaultdict(list);valid=total=0
  for index in sorted(selected):
   a=ds[index];sid,slot,tick=map(int,ds.index[index]);seq,_,_=ds.sequence(sid)
   mask=a['trajectory_valid'][:8]&a['start_hand_valid'][None];v=a['trajectory_target'][:8];world=seq['joints'][tick+1:tick+9]-seq['joints'][tick]
   valid+=int(mask.sum());total+=mask.size
   store['tau_training_m'].append(v[mask]);store['tau_world_m'].append(world[mask])
   z=(v-np.array(stats['hand']['mean'])[:8,None,None])/np.array(stats['hand']['std'])[:8,None,None]
   store['tau_inherited_z'].append(z[mask])
   for name,key,scale,m in [('scene','xyz',config['position_scale_m'],a['point_valid']),('hand','start_hand',config['position_scale_m'],a['start_hand_valid']),('object','scene_points',config['position_scale_m'],np.ones(a['scene_points'].shape[:-1],bool))]:
    store[name+'_input_scaled'].append((a[key]/scale)[m])
  entry={k:summary(v) for k,v in store.items()};entry.update(sampled_windows=len(selected),sampled_mothers=len({int(ds.index[i,0]) for i in selected}),effective_supervision_fraction=valid/total,train_windows=int(sizes.sum()),phase_counts=dict(phases[('train',s)]),dev_windows=sum(r['admitted_windows'] for r in rows if r['source_id']==s and r['split']=='dev'))
  results[str(s)]=entry;write(OUT/'audit-progress.json',dict(stage='sample',completed_source=s,elapsed_s=time.monotonic()-start));print(json.dumps(dict(source=s,stats=entry)),flush=True)
 result=dict(schema='ref2dex.source-supervision-diagnosis.v1',head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),environment=dict(python=sys.version,executable=sys.executable,device='CPU',conda='graspenv'),sampling='256 uniform TRAIN windows/source without replacement seed20261023; exact receipt census window/phase counts; no DEV labels used in moments',contract='first8 x 2hands x21 joints x3; actual training is start-object axes, also world-axis displacement; no invented 11-joint mapping',input_normalization='fixed position_scale_m; no input mean/std moments exist',manifest_sha256=sha(dest/'manifest.json'),sources=results,elapsed_s=time.monotonic()-start)
 write(OUT/'source-supervision-stats.json',result)
 lines=['|source|TRAIN windows|sample windows|valid fraction|tau world RMS mm|tau train RMS mm|tau norm mean mm|inherited z p1 / p99 xyz|phase counts 0/1/2/3/4|','|---|---:|---:|---:|---:|---:|---:|---|---|']
 for s,v in results.items():
  z=v['tau_inherited_z'];lines.append('|%s|%d|%d|%.4f|%.3f|%.3f|%.3f|%s / %s|%s|'%(s,v['train_windows'],v['sampled_windows'],v['effective_supervision_fraction'],v['tau_world_m']['rms']*1000,v['tau_training_m']['rms']*1000,v['tau_training_m']['norm_mean']*1000,str(np.round(z['p1'],2)),str(np.round(z['p99'],2)),str(v['phase_counts'])))
 (OUT/'source-supervision-table.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
