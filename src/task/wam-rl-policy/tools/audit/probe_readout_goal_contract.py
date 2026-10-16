"""Frozen goal-condition intervention: task vs causal short goal vs diagnostic oracle.

The oracle uses future endpoint labels ONLY for this separate diagnostic. It
must never be fed to policy training, a deployed controller, or value fitting.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

TASK=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(TASK/'src'),str(TASK.parent/'shadow-finetune/src')]
from shadow_finetune import paths
from wam_rl_policy.fast_bc import batched_inputs
from wam_rl_policy.wam_features import FeatureController
from wam_rl_policy.task_readout import observed_batch,forecast_from_encoded
from video_data_preparation.unidexgrasp_data import object_poses


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args()
    output=a.run/'goal-contract'
    if output.exists():raise FileExistsError(output)
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    torch.set_num_threads(2);torch.manual_seed(246)
    started=time.monotonic();output.mkdir()
    packet=torch.load(a.run/'features/features.pt',map_location='cpu',weights_only=False)
    manifest=packet['manifest'];objects=packet['objects']
    controller=FeatureController(manifest['checkpoint'],objects,a.cache)
    model=controller.model.forecaster
    point_budget=max(len(g['surface']) for g in controller.geometry)
    subsets=[];targets=[];predictions={name:[] for name in ['task','scheduled','zero','oracle']}
    endpoints={name:[] for name in predictions}
    jobs=0
    with torch.inference_mode():
        for ep,name in enumerate(['first-episode.npz','last-episode.npz']):
            with np.load(Path(manifest['source'])/name) as data:
                obj=data['object'];hand=data['hand'];goal=data['goal'];reset=data['reset']
            poses=object_poses(obj)
            for tick in packet['tick'].unique().tolist():
                if tick+48>=len(obj):continue
                assert not reset[tick+1:tick+49].any()
                for start in range(0,len(objects),16):
                    stop=min(start+16,len(objects));past=tick+np.array([-6,-4,-2,0])
                    selected=((packet['episode']==ep)&(packet['tick']==tick)&
                              (packet['object_id']>=start)&(packet['object_id']<stop))
                    assert packet['object_id'][selected].tolist()==list(range(start,stop))
                    samples=batched_inputs(obj[past,start:stop].swapaxes(0,1),hand[past,start:stop].swapaxes(0,1),
                                           goal[tick,start:stop],controller.geometry[start:stop])
                    batch=observed_batch(samples,point_budget)
                    original=batch['goal_object_delta'].clone()
                    now=poses[tick,start:stop];inverse=np.linalg.inv(now);r=now[:,:3,:3]
                    query=np.stack([s['scene_points'][0] for s in samples])
                    rel=inverse@poses[tick+48,start:stop]
                    oracle=np.einsum('bij,bpj->bpi',rel[:,:3,:3],query)+rel[:,None,:3,3]-query
                    actual_hand=[];actual_scene=[]
                    for h in [16,32]:
                        rel=inverse@poses[tick+h,start:stop]
                        actual_scene.append(np.einsum('bij,bpj->bpi',rel[:,:3,:3],query)+rel[:,None,:3,3]-query)
                        actual_hand.append(np.einsum('bki,bij->bkj',hand[tick+h,start:stop]-hand[tick,start:stop],r))
                    targets.append(np.concatenate((np.stack(actual_hand,1).reshape(stop-start,-1),
                                                   np.stack(actual_scene,1).reshape(stop-start,-1)),1))
                    subsets.append(packet['fit'][selected].numpy())
                    predictions['task'].append(packet['forecast'][selected].numpy())
                    endpoints['task'].append(original.cpu().numpy())
                    goals=dict(scheduled=original*min(48/(199-tick),1),zero=torch.zeros_like(original),
                               oracle=torch.as_tensor(oracle,dtype=original.dtype,device='cuda'))
                    for condition,value in goals.items():
                        batch['goal_object_delta']=value
                        shared,valid,geom=model.encode(batch)
                        hf,sf=forecast_from_encoded(model,batch,shared,valid,geom['tokens'].shape[1])
                        pred=torch.cat((hf[:,[7,15],0].flatten(1),sf[:,:,[7,15]].flatten(1)),-1)
                        if not torch.isfinite(pred).all():raise FloatingPointError(condition)
                        predictions[condition].append(pred.cpu().numpy());endpoints[condition].append(value.cpu().numpy())
                    jobs+=1
                    if jobs==1 or jobs%28==0:
                        elapsed=time.monotonic()-started
                        row=dict(batches=jobs,total=112,seconds=elapsed,eta_seconds=elapsed/jobs*(112-jobs),
                            resource=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
                            '--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip())
                        print(json.dumps(row),flush=True)
                        with (output/'resources.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    if time.monotonic()-started>300:raise TimeoutError('goal diagnostic budget')
    target=np.concatenate(targets);fit=np.concatenate(subsets)
    predictions={k:np.concatenate(v) for k,v in predictions.items()}
    endpoints={k:np.concatenate(v) for k,v in endpoints.items()}
    result={}
    for split,mask in [('fit',fit),('heldout',~fit)]:
        result[split]={}
        for domain,part,points in [('hand',slice(0,66),11),('scene',slice(66,None),32)]:
            truth=target[mask,part].reshape(-1,2,points,3)
            for j,h in enumerate([16,32]):
                motion=np.linalg.norm(truth[:,j],axis=-1).mean(-1)
                for subset,chosen in [('all',np.ones(len(truth),bool)),('moving5mm',motion>.005)]:
                    if not chosen.any():continue
                    metrics={}
                    for name,pred in predictions.items():
                        value=pred[mask,part].reshape(-1,2,points,3)
                        error=np.linalg.norm(value[chosen,j]-truth[chosen,j],axis=-1).mean(-1)*1000
                        metrics[name]=float(error.mean())
                    metrics['stationary_prediction']=float(motion[chosen].mean()*1000)
                    result[split][f'{domain}-{h}-{subset}']=dict(states=int(chosen.sum()),error_mm=metrics)
        result[split]['goal_distance_mm']={k:float(np.linalg.norm(v[mask],axis=-1).mean()*1000) for k,v in endpoints.items()}
    result.update(status='COMPLETED',seconds=time.monotonic()-started,samples=len(fit),
        scope='Frozen condition diagnostic; oracle contains future and cannot enter any policy or value fit',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_checkpoint=manifest['checkpoint'],peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(output/'diagnostic-only-predictions.npz',target=target,fit=fit,**predictions)
    print(json.dumps(result['heldout'],indent=2),flush=True)


if __name__=='__main__':main()
