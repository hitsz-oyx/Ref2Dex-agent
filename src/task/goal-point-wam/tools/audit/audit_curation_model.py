#!/usr/bin/env python3
"""Frozen actor errors and TRAIN-only cross-sequence condition neighbors."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'tools/run'))
from train_three_domain import GoalPointWAM, GoalWindows, loader, move_batch, gpu_snapshot
from oakink_wm.multisource import MixedWindows, source_draw
from audit_native_curation import sequence_metrics, masks, FIELDS, STRATA


def geometry(base, indices):
    result=np.empty((len(indices),len(FIELDS)),np.float32)
    jobs=[]; destinations=[]
    for sid,source in enumerate(base.sources):
        pos=np.flatnonzero((indices>=base.offsets[sid])&(indices<base.offsets[sid+1]))
        rows=source.rows[indices[pos]-base.offsets[sid]]
        for seq in np.unique(rows[:,0]):
            selected=rows[:,0]==seq
            jobs.append((str(source.root),source.sequences[int(seq)],rows[selected]));destinations.append(pos[selected])
    with ProcessPoolExecutor(max_workers=4) as pool:
        for dest,(values,_) in zip(destinations,pool.map(sequence_metrics,jobs)):
            result[dest]=values
    return result


@torch.no_grad()
def infer(model, dataset, indices, batch_size, label):
    records={k:[] for k in ('index','source','sequence','presence','real_mm','static_mm','zero_mm','first8_mm',
        'h1_mm','goal_response_mm','target','valid','scene_embedding','goal_embedding','hand_embedding')}
    cursor=0;started=time.time()
    for cpu in loader(dataset,indices,batch_size,0):
        batch=move_batch(cpu,torch.device('cuda'))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            scene=model.scene(batch);raw=model.observed_hand_condition(batch)
            goal,gm=model.goal(batch['start_object_points'],batch['goal_object_points'],batch['object_goal_valid'])
            pred=model.trajectory(scene['scene_tokens'],scene['scene_mask'],goal,gm,raw)
            sg,sm=model.goal(batch['start_object_points'],batch['start_object_points'],batch['object_goal_valid'])
            static=model.trajectory(scene['scene_tokens'],scene['scene_mask'],sg,sm,raw)
        valid=batch['trajectory_valid'];target=batch['trajectory_target'];n=len(pred)
        errors={k:torch.linalg.vector_norm(p.float()-target,dim=-1) for k,p in
            [('real',pred),('static',static),('zero',torch.zeros_like(pred))]}
        for key,error in errors.items():
            records[key+'_mm'].append(((error*valid).sum((1,2,3))/valid.sum((1,2,3))*1000).cpu().numpy())
        for name,sl in [('first8',slice(0,8)),('h1',slice(0,1))]:
            e=errors['real'][:,sl];v=valid[:,sl]
            records[name+'_mm'].append(((e*v).sum((1,2,3))/v.sum((1,2,3))*1000).cpu().numpy())
        response=torch.linalg.vector_norm(pred.float()-static.float(),dim=-1)
        records['goal_response_mm'].append(((response*valid).sum((1,2,3))/valid.sum((1,2,3))*1000).cpu().numpy())
        records['index'].append(indices[cursor:cursor+n]);records['source'].append(batch['source_id'].cpu().numpy())
        records['sequence'].append(batch['sample_id'][:,0].cpu().numpy())
        records['presence'].append(batch['hand_presence'].cpu().numpy())
        records['target'].append(target.cpu().numpy());records['valid'].append(valid.cpu().numpy())
        for name,tokens,mask in [('scene',scene['scene_tokens'],scene['scene_mask']),('goal',goal,gm)]:
            avg=(tokens.float()*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
            records[name+'_embedding'].append(avg.cpu().numpy())
        if raw is None:raise ValueError('expected observed-hand model')
        records['hand_embedding'].append(raw.flatten(1).float().cpu().numpy())
        cursor+=n
        if cursor%256==0 or cursor==len(indices):
            print(json.dumps(dict(phase=label,done=cursor,total=len(indices),elapsed_s=time.time()-started,
                eta_s=(time.time()-started)/cursor*(len(indices)-cursor))),flush=True)
    return {k:np.concatenate(v) for k,v in records.items()}


def report_strata(arr,metrics):
    return {name:dict(n=int(mask.sum()),**{k:float(arr[k][mask].mean()) for k in
        ('real_mm','static_mm','zero_mm','first8_mm','h1_mm','goal_response_mm')})
        for name,mask in masks(metrics).items() if mask.any()}


@torch.no_grad()
def neighbors(arr, metrics, seed):
    reports={};rng=np.random.default_rng(seed)
    for sid,name in enumerate(('oakink2','grab','arctic')):
        ids=np.flatnonzero(arr['source']==sid);features=[]
        for key in ('scene_embedding','goal_embedding','hand_embedding'):
            x=arr[key][ids].astype(np.float64)
            scale=np.maximum(x.std(0),1e-3)
            features.append(((x-x.mean(0))/scale/np.sqrt(x.shape[1])).astype(np.float32))
        z=torch.as_tensor(np.concatenate(features,-1),device='cuda')
        distance=torch.cdist(z,z)
        allowed=(arr['sequence'][ids,None]!=arr['sequence'][ids][None])
        allowed &= (arr['presence'][ids,None]==arr['presence'][ids][None]).all(-1)
        distance.masked_fill_(~torch.as_tensor(allowed,device='cuda'),float('inf'))
        dd,nn=distance.topk(5,largest=False)
        if not torch.isfinite(dd).all():raise ValueError('insufficient cross-sequence hand-matched neighbors')
        nn=nn.cpu().numpy();dd=dd.cpu().numpy()
        random=np.stack([rng.choice(np.flatnonzero(row),5,replace=False) for row in allowed])
        target=arr['target'][ids];valid=arr['valid'][ids]
        def spread(other):
            v=valid[:,None]&valid[other]
            error=np.linalg.norm(target[:,None]-target[other],axis=-1)
            return ((error*v).sum((1,2,3,4))/v.sum((1,2,3,4))*1000)
        local,baseline=spread(nn),spread(random)
        groups={}
        for key,mask in masks(metrics[ids]).items():
            if mask.any():groups[key]=dict(n=int(mask.sum()),neighbor_future_epe_mm=float(local[mask].mean()),
                random_future_epe_mm=float(baseline[mask].mean()),mean_condition_distance=float(dd[mask].mean()))
        reports[name]=dict(n=len(ids),groups=groups,overall_neighbor_future_epe_mm=float(local.mean()),
            overall_random_future_epe_mm=float(baseline.mean()),neighbor_distance_quantiles=np.percentile(dd,[10,50,90]).tolist())
        arr.setdefault('neighbor_epe_mm',np.zeros(len(arr['index']),np.float32))[ids]=local
        arr.setdefault('random_neighbor_epe_mm',np.zeros(len(arr['index']),np.float32))[ids]=baseline
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','checkpoint','stats','census','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--train-per-source',type=int,default=1024);p.add_argument('--batch',type=int,default=32)
    p.add_argument('--seed',type=int,default=299)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.time()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    # Finish multiprocessing geometry before CUDA initialization.
    train=MixedWindows(a.data,'train');val=MixedWindows(a.data,'val')
    val_indices=np.load(a.checkpoint.parent/'validation_balanced.npy');val_metrics=geometry(val,val_indices)
    rng=np.random.default_rng(a.seed)
    train_indices=np.concatenate([train.offsets[sid]+source_draw(src,a.train_per_source,rng) for sid,src in enumerate(train.sources)])
    train_metrics=[]
    for sid,src in enumerate(train.sources):
        with np.load(a.census/(train.meta['sources'][sid]['name']+'.npz')) as d:
            local=train_indices[sid*a.train_per_source:(sid+1)*a.train_per_source]-train.offsets[sid]
            if not np.array_equal(d['indices'],np.arange(len(src))):raise ValueError('complete census required')
            train_metrics.append(d['metrics'][local])
    train_metrics=np.concatenate(train_metrics)
    if not torch.cuda.is_available():raise RuntimeError('GPU required for model inference')
    packet=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if packet.get('artificial_labels') or packet.get('checkpoint_kind')!='goal-point-wam.single.v1':raise ValueError('real frozen WAM required')
    model=GoalPointWAM(json.loads(a.stats.read_text()),packet['config']).cuda().eval()
    model.load_state_dict(packet['model'],strict=True);model.requires_grad_(False)
    inputs=dict(checkpoint=str(a.checkpoint),checkpoint_step=packet['step'],config=packet['config'],
        source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    del packet
    tr=infer(model,GoalWindows(train,'object_points',32),train_indices,a.batch,'train')
    va=infer(model,GoalWindows(val,'object_points',32),val_indices,a.batch,'val')
    result=dict(status='COMPLETED',input=inputs,train={},val={},neighbors=neighbors(tr,train_metrics,a.seed),
        elapsed_s=time.time()-started,gpu=gpu_snapshot(torch.device('cuda')),
        limitations='Single frozen model; window-macro errors; pooled-context KNN proxy, not entropy or label defect; same sequence excluded; no filtering/training')
    for sid,name in enumerate(('oakink2','grab','arctic')):
        for phase,arr,metric in [('train',tr,train_metrics),('val',va,val_metrics)]:
            sel=arr['source']==sid
            result[phase][name]=report_strata({k:v[sel] for k,v in arr.items()},metric[sel])
    for name,arr,metric in [('train',tr,train_metrics),('val',va,val_metrics)]:
        np.savez(a.output/(name+'.npz'),metrics=metric,**arr)
    (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(status='COMPLETED',elapsed_s=result['elapsed_s'],gpu=result['gpu'])),flush=True)


if __name__=='__main__':main()
