"""Append a completed arm's measured aggregate and source/phase paired results."""
import json,sys,subprocess,os
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
 if arm.split('-r')[0]=='C3':
  import torch
  initial=torch.load(D/'local3000-weights.pt',map_location='cpu',weights_only=False)['model']
  checkpoint=torch.load(folder/('step%06d.pt'%step),map_location='cpu',weights_only=False)['model']
  names=read(folder/'run-manifest.json')['trainable_names'];assert names and all(n.startswith('flow_future.') for n in names)
  frozen=[n for n in initial if not n.startswith('flow_future.')];changed=[n for n in frozen if not torch.equal(initial[n],checkpoint[n])]
  flow_changed=[n for n in initial if n.startswith('flow_future.') and not torch.equal(initial[n],checkpoint[n])]
  proof=dict(trainable_names_all_flow=True,frozen_state_tensors=len(frozen),changed_nonflow_tensors=changed,changed_flow_tensors=len(flow_changed),frozen_exactly_equal=not changed)
  write(folder/'frozen-state-audit.json',proof,immutable=True);report['frozen_state_audit']=proof
  assert not changed,'C3 non-flow state changed despite freeze'
 write(folder/'grouped-panel.json',report,immutable=True)
 lines=['\n## '+arm+' completed (measured)','\nProgress: `'+json.dumps(read(folder/'progress.json'),sort_keys=True)+'`','\nPanel summary:\n```json\n'+json.dumps(final,indent=2)+'\n```']
 for key in ('by_source','by_phase'):
  lines+=['\n'+key,'\n|group|rows/mothers|first8 hand mm (delta)|contact hand mm (delta)|first8 object mm (delta)|','|---|---:|---:|---:|---:|']
  for k,v in report[key].items():
   fmt=lambda i:'NA' if v['means_mm'][i] is None else '%.4f (%+.4f)'%(v['means_mm'][i],v['delta_mm'][i]);lines.append('|%s|%d/%d|%s|%s|%s|'%(k,v['rows'],v['mothers'],fmt(0),fmt(1),fmt(2)))
 lines+=['\nFull reproducible paired group CIs and relative changes: `'+str(folder/'grouped-panel.json')+'`.']
 delta=final['versus_local3000']['delta_mm'];stable=all(v<=0 for v in delta)
 if arm=='C1':verdict='All three stable/improve: continuation cleared under matched C1; inspect distribution via C2.' if stable else 'C1 does not recover all three; compare magnitude with half before assigning H3 (large batch/fresh optimizer).'
 elif arm=='C2':verdict='All three stable/improve: source sampling weights implicated by preregistered rule.' if stable else 'C2 does not recover all three: new-source supervision remains implicated by owner rubric; inspect C3/C5 and scale evidence.'
 elif arm.split('-r')[0]=='C3':verdict='All three stable/improve: freezing shared/backbone protects panel; shared-parameter damage implicated.' if stable else 'C3 does not recover all three: flow-only update can still shift predictions; inspect target semantics/scales.'
 elif arm=='C5':
  audit=read(OUT/'C5-validity-audit.json')
  verdict=('All three stable/improve: invalid-window attribution under owner rule.' if stable else 'C5 does not recover all three: return to weights/scales under owner rule.')
  if audit['rejected_windows']==0:verdict+=' IDENTIFIABILITY LIMIT: gate removed0 windows; this outcome cannot be caused by invalid-window removal. C5 measures full-data uniform continuation with a no-op validity gate.'
 else:
  others=[v for key,v in report['by_source'].items() if key!=arm.split('source')[1]]
  large=any(v['relative_change_percent'][i] is not None and v['relative_change_percent'][i]>=50 for v in others for i in range(3))
  verdict='Other-source EPE worsening>=50% observed; strong interference under owner threshold.' if large else 'No other-source EPE worsening>=50% in this arm; inspect full matrix and sparse-source CIs.'
 lines+=['\nPreregistered interpretation: '+verdict]

 with (OUT/'README.md').open('a') as f:f.write('\n'.join(lines)+'\n')
 (OUT/'README.md').chmod(0o664)
def append_matrix_if_ready():
 arms=['C4-source1','C4-source10','C4-source0']
 if (OUT/'C4-interference-matrix.json').exists() or not all((OUT/arm/'grouped-panel.json').exists() for arm in arms):return
 matrix={arm:read(OUT/arm/'grouped-panel.json')['by_source'] for arm in arms}
 write(OUT/'C4-interference-matrix.json',matrix,immutable=True)
 sources=sorted({source for values in matrix.values() for source in values},key=int);lines=['\n## C4 interference matrix (measured)\n','Cells are EPE delta mm (relative %), trained-source rows versus evaluated-source columns. Positive means worse; negative means improved. Diagonal is within-source transfer, off-diagonal is interference/transfer. Owner50% interference threshold is evaluated as EPE worsening>=50%; raw signed changes are retained.']
 for i,name in enumerate(['first8_hand','fixed_contact_hand','first8_object']):
  lines+=['\n'+name,'\n|trained source|'+ '|'.join(sources)+'|','|---|'+ '|'.join(['---:']*len(sources))+'|']
  for arm in arms:
   cells=[]
   for source in sources:
    v=matrix[arm][source];delta=v['delta_mm'][i];relative=v['relative_change_percent'][i];cells.append('NA' if delta is None or relative is None else '%+.3f (%+.1f%%)'%(delta,relative))
   lines.append('|'+arm.split('source')[1]+'|'+'|'.join(cells)+'|')
 with (OUT/'README.md').open('a') as f:f.write('\n'.join(lines)+'\n')
 (OUT/'README.md').chmod(0o664)

def commit_snapshot(arm):
 for artifact in (OUT/arm).rglob('*'):
  if artifact.is_file() and not artifact.is_symlink():artifact.chmod(0o664)
 report=ROOT/'src/task/ptv3-shared-wam-v6/docs/audit/nine-source-degradation-20261023.md'
 report.write_text((OUT/'README.md').read_text())
 r=read(OUT/arm/'recipe-v2.json');result=read(OUT/arm/'result.json')
 subprocess.run(['git','add',str(report)],cwd=ROOT,check=True)
 message='Record '+arm+' '+str(result['updates'])+' updates; data '+r.get('data_identity_sha256',r['manifest_sha256'])
 subprocess.run(['git','commit','-m',message],cwd=ROOT,check=True)
 write(OUT/arm/'report-commit.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),pushed=False),immutable=True)
if __name__=='__main__':
 append(sys.argv[1]);append_matrix_if_ready();commit_snapshot(sys.argv[1])
