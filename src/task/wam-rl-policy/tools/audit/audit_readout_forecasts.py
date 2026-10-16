"""Compare already-cached WAM priors to actual native point motion, without inference."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

TASK=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(TASK/'src'),str(TASK.parent/'shadow-finetune/src')]
from shadow_finetune import paths
from video_data_preparation.unidexgrasp_data import object_poses


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    a=p.parse_args()
    out=a.run/'forecast-audit.json'
    if out.exists():raise FileExistsError(out)
    packet=torch.load(a.run/'features/features.pt',map_location='cpu',weights_only=False)
    forecast=packet['forecast'].numpy();fit=packet['fit'].numpy()
    predicted_hand=forecast[:,:66].reshape(-1,2,11,3)
    predicted_scene=forecast[:,66:].reshape(-1,2,32,3)
    hand=np.zeros_like(predicted_hand);scene=np.zeros_like(predicted_scene)
    cv_hand=np.zeros_like(hand);cv_scene=np.zeros_like(scene)
    meta=json.loads((a.cache/'manifest.json').read_text())
    geometry={r['object_code']:r['geometry'] for r in meta['records']}
    queries=[]
    for row in packet['objects']:
        with np.load(geometry[row['object_code']]) as g:
            points=g['surface'][np.linalg.norm(g['normals'],axis=-1)>1e-6]
            queries.append(points[np.linspace(0,len(points)-1,32,dtype=int)])
    queries=np.stack(queries)
    source=Path(packet['manifest']['source'])
    for episode,name in enumerate(['first-episode.npz','last-episode.npz']):
        with np.load(source/name) as d:
            objects=d['object'];hands=d['hand']
        poses=object_poses(objects)
        for tick in packet['tick'].unique().tolist():
            selected=((packet['episode']==episode)&(packet['tick']==tick)).numpy()
            envs=packet['object_id'][selected].numpy()
            now=poses[tick,envs];r=now[:,:3,:3];inverse=np.linalg.inv(now)
            query=queries[envs]
            prev=inverse@poses[tick-2,envs]
            prev_delta=np.einsum('bij,bpj->bpi',prev[:,:3,:3],query)+prev[:,None,:3,3]-query
            for j,horizon in enumerate([16,32]):
                relative=inverse@poses[tick+horizon,envs]
                scene[selected,j]=np.einsum('bij,bpj->bpi',relative[:,:3,:3],query)+relative[:,None,:3,3]-query
                hand[selected,j]=np.einsum('bki,bij->bkj',hands[tick+horizon,envs]-hands[tick,envs],r)
                cv_scene[selected,j]=-prev_delta*(horizon/2)
                cv_hand[selected,j]=np.einsum('bki,bij->bkj',hands[tick,envs]-hands[tick-2,envs],r)*(horizon/2)
    result={}
    for split,mask in [('fit_objects',fit),('heldout_objects',~fit)]:
        rows={}
        for name,target,pred,cv in [('hand',hand,predicted_hand,cv_hand),('scene',scene,predicted_scene,cv_scene)]:
            for j,horizon in enumerate([16,32]):
                truth=target[mask,j]
                moving=np.linalg.norm(truth,axis=-1).mean(-1)>.005
                for subset,chosen in [('all',np.ones(len(truth),bool)),('moving5mm',moving)]:
                    if not chosen.any():continue
                    values={}
                    for method,guess in [('wam',pred[mask,j]),('zero',np.zeros_like(truth)),('constant_velocity',cv[mask,j])]:
                        error=np.linalg.norm(guess[chosen]-truth[chosen],axis=-1).mean(-1)*1000
                        values[method]=dict(mean_point_error_mm=float(error.mean()),p95_mm=float(np.percentile(error,95)))
                    rows[f'{name}-{horizon}-{subset}']=dict(states=int(chosen.sum()),metrics=values)
        result[split]=rows
    result['scope']='Cached frozen goal-conditioned predictions under actual mixed robot behavior; no action-conditioned causal claim'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['heldout_objects'],indent=2))


if __name__=='__main__':main()
