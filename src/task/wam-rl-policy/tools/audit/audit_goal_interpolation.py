"""Same-oracle-endpoint geometric baseline; no extra model evaluation."""
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
    output=a.run/'goal-contract/interpolation-audit.json'
    if output.exists():raise FileExistsError(output)
    packet=torch.load(a.run/'features/features.pt',map_location='cpu',weights_only=False)
    meta=json.loads((a.cache/'manifest.json').read_text())
    lookup={r['object_code']:r['geometry'] for r in meta['records']}
    queries=[]
    for row in packet['objects']:
        with np.load(lookup[row['object_code']]) as g:
            points=g['surface'][np.linalg.norm(g['normals'],axis=-1)>1e-6]
            queries.append(points[np.linspace(0,len(points)-1,32,dtype=int)])
    queries=np.stack(queries);interpolated=[];checks=[];truth_checks=[]
    for ep,name in enumerate(['first-episode.npz','last-episode.npz']):
        with np.load(Path(packet['manifest']['source'])/name) as data:
            poses=object_poses(data['object']);hand=data['hand']
        for tick in packet['tick'].unique().tolist():
            if tick+48>=len(poses):continue
            now=poses[tick];inv=np.linalg.inv(now);r=now[:,:3,:3]
            current_hand=np.einsum('bki,bij->bkj',hand[tick]-now[:,None,:3,3],r)
            endpoint=inv@poses[tick+48]
            object_delta=np.einsum('bij,bpj->bpi',endpoint[:,:3,:3],queries)+endpoint[:,None,:3,3]-queries
            hand_delta=np.einsum('bij,bpj->bpi',endpoint[:,:3,:3],current_hand)+endpoint[:,None,:3,3]-current_hand
            actual_hand=[];actual_scene=[]
            for horizon in [16,32]:
                rel=inv@poses[tick+horizon]
                actual_hand.append(np.einsum('bki,bij->bkj',hand[tick+horizon]-hand[tick],r))
                actual_scene.append(np.einsum('bij,bpj->bpi',rel[:,:3,:3],queries)+rel[:,None,:3,3]-queries)
            truth_checks.append(np.concatenate((np.stack(actual_hand,1).reshape(len(poses[0]),-1),
                                               np.stack(actual_scene,1).reshape(len(poses[0]),-1)),1))
            factor=np.array([16/48,32/48])[None,:,None,None]
            interpolated.append(np.concatenate(((hand_delta[:,None]*factor).reshape(len(queries),-1),
                                                 (object_delta[:,None]*factor).reshape(len(queries),-1)),1))
            chosen=(packet['episode']==ep)&(packet['tick']==tick)
            assert packet['object_id'][chosen].tolist()==list(range(len(queries)))
            checks.append(packet['fit'][chosen].numpy())
    pred=np.concatenate(interpolated);fit=np.concatenate(checks)
    with np.load(a.run/'goal-contract/diagnostic-only-predictions.npz') as d:
        np.testing.assert_array_equal(fit,d['fit'])
        np.testing.assert_allclose(np.concatenate(truth_checks),d['target'],rtol=1e-6,atol=1e-7)
        target=d['target'];wam=d['oracle']
    result={}
    for name,part,points in [('hand',slice(0,66),11),('scene',slice(66,None),32)]:
        truth=target[~fit,part].reshape(-1,2,points,3)
        simple=pred[~fit,part].reshape(-1,2,points,3)
        learned=wam[~fit,part].reshape(-1,2,points,3)
        for j,horizon in enumerate([16,32]):
            base=float(np.linalg.norm(simple[:,j]-truth[:,j],axis=-1).mean()*1000)
            error=float(np.linalg.norm(learned[:,j]-truth[:,j],axis=-1).mean()*1000)
            result[f'{name}-{horizon}']=dict(geometric_interpolation_mm=base,wam_mm=error,relative_reduction=1-error/base)
    output.write_text(json.dumps(dict(status='PASS',scope='same future endpoint diagnostic, not online evidence',
                                     heldout_states=int((~fit).sum()),metrics=result),indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
