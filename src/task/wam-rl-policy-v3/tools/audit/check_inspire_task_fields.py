"""Check full admission and offline/online task7 equivalence; no neural inference."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
TASK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TASK/'src'))
from wam_rl_policy_v3.inspire_shadow import BatchedInspireInputs,TASK_SLICE,CONTRACT
from wam_rl_policy_v3.wam_v4_runtime import load_geometry


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('fresh audit output required')
    torch.set_num_threads(2);m=json.loads((a.cache/'manifest.json').read_text())
    assert m['task_status']=='COMPLETED' and m['conditioning']==CONTRACT and len(m['sequences'])==660
    geometry=load_geometry(a.cache,m['records'],a.cache/'surface-manifest.json')
    context={s:np.load(a.cache/f'{s}-task.npy',mmap_mode='r') for s in ('train','dev')}
    physical={s:np.load(a.cache/f'{s}-x.npy',mmap_mode='r') for s in context}
    for split,values in context.items():assert values.shape==(m['splits'][split]['commands'],7) and np.isfinite(values).all()
    rows=[]
    for obj in sorted({r['object'] for r in m['sequences']}):
        r=next(r for r in m['sequences'] if r['object']==obj)
        with np.load(Path(r['source'])/'trajectory.npz') as z:poses=z['obj'][:-1].copy();hands=z['hand'][:-1].copy();reference=z['obj'].copy()
        runtime=type('Runtime',(),dict(device='cpu',geometry=[geometry[r['geometry']]],config=dict(voxel_origin_m=[-2]*3,voxel_m=.01)))()
        online=BatchedInspireInputs(runtime,[reference]);online.predict_trajectory=False
        error=0.
        for t in range(min(20,len(poses))):
            out=online.observation(physical[r['cache_split']][r['lo']+t:r['lo']+t+1],poses[t:t+1],hands[t:t+1],np.array([t]))
            error=max(error,float(np.abs(out['observation'][0,TASK_SLICE]-context[r['cache_split']][r['lo']+t]).max()))
        if error>1e-6:raise ValueError('offline/online task mismatch '+obj)
        rows.append(dict(object=obj,max_error=error))
    report=dict(status='PASS',objects=len(rows),sequences=660,commands=sum(len(x) for x in context.values()),
        max_offline_online_error=max(r['max_error'] for r in rows),rows=rows,neural_inference=False,optimizer_steps=0)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':main()
