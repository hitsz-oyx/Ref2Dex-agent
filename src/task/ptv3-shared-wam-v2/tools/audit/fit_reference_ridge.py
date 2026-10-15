"""Closed-form wrist predictability controls with and without current reference error."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[5];out=a.output.resolve()
    if out.exists() or root/'outputs/ptv3-shared-wam-v2' not in out.parents:
        raise ValueError('Fresh Task output required')
    os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    import numpy as np
    import torch
    torch.set_num_threads(2)
    out.mkdir(parents=True)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('300s ridge diagnostic')))
    signal.alarm(300);started=time.monotonic()
    source=root/'outputs/ptv3-shared-wam-v2/dexplore-current-rollout-20261015-r1'
    corpus=json.loads((source/'manifest.json').read_text())
    membership=json.loads((Path(corpus['source_membership'])/'manifest.json').read_text())
    geometry=Path(membership['source_geometric_root'])
    snapshot=json.loads((root/'outputs/ptv3-shared-wam-v2/shared-actor-shared-20261015-r1/encoder_workspace_admission.json').read_text())
    lookup={r['sequence']:r for r in corpus['records']}
    data={}
    for split in ('train','val'):
        xs,refs,ys,ts,ss=[],[],[],[],[]
        names=[]
        for record in snapshot[split]['records']:
            ticks=np.asarray(record['admitted_ticks'],dtype=np.int64)
            if not len(ticks):continue
            name=record['sequence'];item=lookup[name];assert item['split']==split
            with np.load(source/item['path']/'trajectory.npz') as z:
                d={k:z[k].copy() for k in ('q','dq','hand','obj','velocity','reference_obj','applied')}
            raw=torch.load(geometry/name/'interaction_hand_inspire.pt',map_location='cpu',weights_only=False)
            raw=raw.numpy() if isinstance(raw,torch.Tensor) else np.asarray(raw)
            assert len(raw)==len(d['q'])==item['frames'] and ticks.max()+24<len(raw)
            q_ref=raw[ticks,373:391].astype(np.float32)
            rotation=d['obj'][ticks,:3,:3];center=d['obj'][ticks,:3,3]
            local=np.einsum('npj,njk->npk',d['hand'][ticks]-center[:,None],rotation)
            state=np.concatenate((d['q'][ticks],d['dq'][ticks],local.reshape(-1,33),
                d['obj'][ticks,:3,:].reshape(-1,12),d['velocity'][ticks]),-1)
            goal=d['reference_obj'][ticks+24,:3,3]-center
            xs.append(np.concatenate((state,goal),-1));refs.append(q_ref-d['q'][ticks])
            ys.append(d['applied'][ticks,:6]);ts.append(ticks);ss.append(np.full(len(ticks),len(names),np.int32));names.append(name)
        data[split]=dict(x=np.concatenate(xs).astype(np.float64),ref=np.concatenate(refs).astype(np.float64),
            y=np.concatenate(ys).astype(np.float64),tick=np.concatenate(ts),sequence=np.concatenate(ss),names=names)
    assert [len(data[s]['y']) for s in ('train','val')]==[93680,10376]
    scale=np.asarray(corpus['native_controller']['scale'][:6],dtype=np.float64)
    result=dict(scope='Matched linear ridge predictability only; no learned WAM update or closed-loop claim',
        lambda_=1e-3,cpu_reason='Small normal-equation sufficient statistics, no neural network or simulation',
        feature_contract='current measured87 + existing object_goal24 delta; optionally current q_ref[t]-q_real[t]18 only',arms={})
    coefficients={}
    for arm in ('measured_and_object_goal','plus_current_reference_error'):
        x={s:(d['x'] if arm=='measured_and_object_goal' else np.concatenate((d['x'],d['ref']),-1)) for s,d in data.items()}
        mean=x['train'].mean(0);std=x['train'].std(0).clip(1e-3)
        ymean=data['train']['y'].mean(0);ystd=data['train']['y'].std(0).clip(.01)
        z=(x['train']-mean)/std;y=(data['train']['y']-ymean)/ystd
        coef=np.linalg.solve(z.T@z/len(z)+np.eye(z.shape[1])*1e-3,z.T@y/len(z))
        coefficients.update({arm+'_'+key:value for key,value in dict(mean=mean,std=std,ymean=ymean,ystd=ystd,coef=coef).items()})
        panels={}
        for split,d in data.items():
            pred=((x[split]-mean)/std)@coef*ystd+ymean
            delta=(pred-d['y'])*scale;xyz=np.linalg.norm(delta[:,:3],axis=-1)*1000
            masks={'all':np.ones(len(xyz),bool),'tick0':d['tick']==0,'startup_1_15':(d['tick']>0)&(d['tick']<16),'later':d['tick']>=16}
            panels[split]={name:dict(count=int(mask.sum()),xyz_mean_mm=float(xyz[mask].mean()) if mask.any() else None,
                xyz_p95_mm=float(np.percentile(xyz[mask],95)) if mask.any() else None,
                wrist_Euler_MAE_rad=float(np.abs(delta[mask,3:]).mean()) if mask.any() else None) for name,mask in masks.items()}
            np.savez_compressed(out/(arm+'-'+split+'.npz'),prediction=pred.astype(np.float32),target=d['y'].astype(np.float32),tick=d['tick'],sequence=d['sequence'])
        result['arms'][arm]=dict(features=x['train'].shape[1],panels=panels)
        print(arm,json.dumps(panels),flush=True)
    np.savez_compressed(out/'coefficients.npz',**coefficients)
    (out/'sequences.json').write_text(json.dumps({s:d['names'] for s,d in data.items()},indent=2)+'\n')
    result.update(status='COMPLETED',elapsed_s=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    signal.alarm(0)


if __name__=='__main__':main()
