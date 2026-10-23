"""Append a completed arm's measured aggregate and source/phase paired results."""
import json,sys
from pathlib import Path
import numpy as np
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2];sys.path.insert(0,str(TASK/'tools/run'))
from remote_staged_data import read,write
D=Path('/mnt/ugreen_nas/storage/oyx_storage/wam-nine-source-full-20261023-pending');OUT=D/'diagnosis'

def groups(s,c,m,ref,rows,key):
 result={}
 for val in sorted({str(r[key]) for r in rows}):
  mask=np.array([str(r[key])==val for r in rows]);ss=s[mask];cc=c[mask];rr=ref[mask];mm=m[mask];den=cc.sum(0)
  mean=np.divide(ss.sum(0),den,out=np.full(3,np.nan),where=den>0);baseline=np.divide(rr.sum(0),den,out=np.full(3,np.nan),where=den>0)
  ids=np.unique(mm);ns=np.array([ss[mm==v].sum(0)-rr[mm==v].sum(0) for v in ids]);nc=np.array([cc[mm==v].sum(0) for v in ids]);rng=np.random.default_rng(170);ci=[]
  for _ in range(4000):
   draw=rng.integers(0,len(ids),len(ids));d=nc[draw].sum(0);ci.append(np.divide(ns[draw].sum(0),d,out=np.full(3,np.nan),where=d>0))
  with np.errstate(all='ignore'):bounds=np.nanpercentile(ci,[2.5,97.5],axis=0).T
  clean=lambda a:[None if not np.isfinite(v) else float(v) for v in a]
  result[val]=dict(rows=int(mask.sum()),mothers=len(ids),means_mm=clean(mean),baseline_mm=clean(baseline),delta_mm=clean(mean-baseline),relative_change_percent=clean((mean/baseline-1)*100),paired_cluster_ci95_mm=[clean(v) for v in bounds],counts=den.tolist())
 return result

def append(arm):
 folder=OUT/arm; panels=sorted(folder.glob('panel-step*/summary.json'));final=read(panels[-1]);step=final['step']; assert step>0
 with np.load(panels[-1].parent/'paired-panel.npz') as z: s,c,m=[z[k] for k in ('sums','counts','mothers')]
 with np.load(D/'half-training/panel-step000000/paired-panel.npz') as z:ref=z['sums'];assert np.array_equal(c,z['counts']) and np.array_equal(m,z['mothers'])
 rows=read(D/'fixed-panel/manifest.json')['rows'];report=dict(summary=final,by_source=groups(s,c,m,ref,rows,'source_id'),by_phase=groups(s,c,m,ref,rows,'phase'))
 write(folder/'grouped-panel.json',report,immutable=True)
 lines=['\n## '+arm+' completed (measured)','\nProgress: `'+json.dumps(read(folder/'progress.json'),sort_keys=True)+'`','\nPanel summary:\n```json\n'+json.dumps(final,indent=2)+'\n```']
 for key in ('by_source','by_phase'):
  lines+=['\n'+key,'\n|group|rows/mothers|first8 hand mm (delta)|contact hand mm (delta)|first8 object mm (delta)|','|---|---:|---:|---:|---:|']
  for k,v in report[key].items():
   fmt=lambda i:'NA' if v['means_mm'][i] is None else '%.4f (%+.4f)'%(v['means_mm'][i],v['delta_mm'][i]);lines.append('|%s|%d/%d|%s|%s|%s|'%(k,v['rows'],v['mothers'],fmt(0),fmt(1),fmt(2)))
 lines+=['\nFull reproducible paired group CIs and relative changes: `'+str(folder/'grouped-panel.json')+'`.']
 with (OUT/'README.md').open('a') as f:f.write('\n'.join(lines)+'\n')
if __name__=='__main__':append(sys.argv[1])
