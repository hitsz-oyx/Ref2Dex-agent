#!/usr/bin/env python3
"""Observed-hand MLP diagnostic, matched H-only versus H+object-goal inputs."""
import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'tools/run'))
from train_three_domain import atomic_json, digest, gpu_snapshot, parameter_hash
from goal_point_wam.data import GoalWindows
from oakink_wm.multisource import MixedWindows, mixed_indices, MAIN_SOURCE_NAMES


def compact_features(sample, stats):
    """No action/future-hand access: current hand/history and paired object goal."""
    valid = sample['start_hand_valid']
    mean = np.asarray(stats['scene_mean'][3:12], dtype=np.float32)
    std = np.maximum(np.asarray(stats['scene_std'][3:12], dtype=np.float32),1e-6)
    hand = np.concatenate((sample['start_hand']/.05, (sample['start_hand_history']-mean)/std),axis=-1)
    hand = np.where(valid[...,None],hand,0.).astype(np.float32)
    start = sample['start_object_points'][sample['object_goal_valid']]
    goal = sample['goal_object_points'][sample['object_goal_valid']]
    if len(start)<3:raise ValueError('at least three paired goal points required')
    a,b=start-start.mean(0),goal-goal.mean(0)
    u,_,vh=np.linalg.svd(a.T@b)
    correction=np.eye(3,dtype=np.float32);correction[-1,-1]=np.linalg.det(vh.T@u.T)
    rotation=vh.T@correction@u.T
    geometry=np.concatenate((start.mean(0)/.05,start.std(0)/.05)).astype(np.float32)
    future=np.concatenate(((goal.mean(0)-start.mean(0))/.05,
                            (rotation-np.eye(3,dtype=np.float32))[:,:2].T.reshape(-1))).astype(np.float32)
    return hand,geometry,future


def prepare(base, indices, stats):
    dataset=GoalWindows(base,goal_type='object_points')
    rows={key:[] for key in ('hand','geometry','goal','target','valid','source','sample_id')}
    for index in indices:
        sample=dataset[int(index)];hand,geometry,goal=compact_features(sample,stats)
        for key,value in dict(hand=hand,geometry=geometry,goal=goal,target=sample['trajectory_target'],
                              valid=sample['trajectory_valid'],source=sample['source_id'],sample_id=sample['sample_id']).items():
            rows[key].append(value)
    packed={key:torch.from_numpy(np.stack(value)) for key,value in rows.items()}
    if any(not torch.isfinite(x).all() for x in packed.values() if x.is_floating_point()):
        raise ValueError('nonfinite compact input/target')
    return packed


class CompactForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.hand=nn.Embedding(2,8);self.keypoint=nn.Embedding(11,8);self.time=nn.Embedding(24,8)
        self.head=nn.Sequential(nn.Linear(51,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,3))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,batch,goal_enabled):
        n=len(batch['hand']);device=batch['hand'].device
        h=self.hand(torch.arange(2,device=device))[:,None].expand(2,11,8)
        k=self.keypoint(torch.arange(11,device=device))[None].expand(2,11,8)
        state=torch.cat((batch['hand'],batch['geometry'][:,None,None].expand(-1,2,11,-1),
            (batch['goal'] if goal_enabled else torch.zeros_like(batch['goal']))[:,None,None].expand(-1,2,11,-1),
            h[None].expand(n,-1,-1,-1),k[None].expand(n,-1,-1,-1)),dim=-1)
        state=state[:,None].expand(-1,24,-1,-1,-1)
        t=self.time(torch.arange(24,device=device))[None,:,None,None].expand(n,-1,2,11,-1)
        return self.head(torch.cat((state,t),dim=-1)).float()


@torch.no_grad()
def measure(model, data, goal_enabled):
    totals={name:dict(sum=0.,zero=0.,count=0,ablation=0.,difference=0.) for name in MAIN_SOURCE_NAMES}
    for left in range(0,len(data['hand']),24):
        batch={key:value[left:left+24] for key,value in data.items()}
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            pred=model(batch,goal_enabled);blind=model(batch,False)
        error=torch.linalg.vector_norm(pred-batch['target'],dim=-1)
        zero=torch.linalg.vector_norm(batch['target'],dim=-1)
        ablation=torch.linalg.vector_norm(blind-batch['target'],dim=-1)
        diff=torch.linalg.vector_norm(pred-blind,dim=-1)
        for sid,name in enumerate(MAIN_SOURCE_NAMES):
            selected=batch['source']==sid;valid=batch['valid'][selected];row=totals[name]
            row['sum']+=float(error[selected][valid].sum());row['zero']+=float(zero[selected][valid].sum())
            row['ablation']+=float(ablation[selected][valid].sum());row['difference']+=float(diff[selected][valid].sum())
            row['count']+=int(valid.sum())
    return {name:dict(trajectory_epe_m=row['sum']/row['count'],zero_hand_epe_m=row['zero']/row['count'],
                      improvement_vs_zero=1-row['sum']/row['zero'],static_goal_epe_m=row['ablation']/row['count'],
                      improvement_vs_static_goal=1-row['sum']/row['ablation'],goal_prediction_difference_m=row['difference']/row['count'])
            for name,row in totals.items()}


def fit(train,val,args,enabled,deadline):
    random.seed(247);np.random.seed(247);torch.manual_seed(247)
    model=CompactForecast().cuda();initial=parameter_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    name='h_goal' if enabled else 'h_only';out=args.output/name;out.mkdir()
    started=time.monotonic();history=[]
    with (out/'train.jsonl').open('w') as log:
        for step in range(1,601):
            if time.monotonic()>deadline:raise TimeoutError('compact diagnosis deadline')
            # Exact original600-update microbatch2 precomputed stream, one pass.
            batch={k:v[(step-1)*2:step*2] for k,v in train.items()}
            scale=step/50. if step<=50 else .1+.9*.5*(1+math.cos(math.pi*(step-50)/550.))
            for group in optimizer.param_groups:group['lr']=.001*scale
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):pred=model(batch,enabled)
            point=torch.nn.functional.huber_loss(pred/.05,batch['target']/.05,reduction='none').mean(-1)
            loss=point[batch['valid']].mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite compact loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            row=dict(step=step,loss=float(loss),gradient_norm=float(grad),elapsed_seconds=time.monotonic()-started)
            if step==1 or step%100==0:
                row['gpu']=gpu_snapshot(torch.device('cuda'));row['val']=measure(model,val,enabled)
                history.append(row);atomic_json(args.output/'progress.json',dict(status='RUNNING',arm=name,**row))
            log.write(json.dumps(row)+'\n')
    result=dict(status='COMPLETED',step=600,goal_enabled=enabled,initial_parameter_sha256=initial,
                parameters=sum(p.numel() for p in model.parameters()),val=measure(model,val,enabled),
                train=measure(model,train,enabled),history=history,elapsed_seconds=time.monotonic()-started)
    torch.save(dict(checkpoint_kind='goal-point-wam.compact-diagnosis.v1',diagnosis_only=True,model=model.state_dict(),
                    goal_enabled=enabled),out/'final.pt')
    atomic_json(out/'result.json',result);return result


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    if not torch.cuda.is_available():raise RuntimeError('GPU required for compact neural diagnosis')
    args.output.mkdir(parents=True);torch.set_num_threads(2)
    started=time.monotonic();deadline=started+args.budget_seconds
    parent=json.loads((args.panel.parent/'input_manifest.json').read_text())
    if digest(args.data/'processed/manifest.json')!=parent['data_manifest_sha256'] or digest(args.stats)!=parent['stats_sha256']:
        raise ValueError('frozen corpus/stats drift')
    stats=json.loads(args.stats.read_text());base=MixedWindows(args.data,'train')
    indices=mixed_indices(base,1200,248);validation=np.load(args.panel)
    np.save(args.output/'train_indices.npy',indices);np.save(args.output/'val_indices.npy',validation)
    train=prepare(base,indices,stats);val=prepare(MixedWindows(args.data,'val'),validation,stats)
    torch.save(dict(train=train,val=val),args.output/'compact_data.pt')
    atomic_json(args.output/'input_manifest.json',dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=digest(Path(__file__)),adapter_sha256=digest(TASK/'src/goal_point_wam/data.py'),
        data_manifest_sha256=parent['data_manifest_sha256'],stats_sha256=parent['stats_sha256'],panel_sha256=digest(args.panel),
        compact_data_sha256=digest(args.output/'compact_data.pt'),train_indices_sha256=digest(args.output/'train_indices.npy'),
        seed=247,train_seed=248,updates=600,microbatch=2,world_loss=False,source_id_model_input=False,
        budget_seconds=args.budget_seconds,gpu=gpu_snapshot(torch.device('cuda'))))
    train={k:v.cuda() for k,v in train.items()};val={k:v.cuda() for k,v in val.items()}
    results={name:fit(train,val,args,enabled,deadline) for name,enabled in (('h_only',False),('h_goal',True))}
    if results['h_only']['initial_parameter_sha256']!=results['h_goal']['initial_parameter_sha256']:
        raise ValueError('compact initial parameters differ')
    sources={}
    for name in MAIN_SOURCE_NAMES:
        h,g=results['h_only']['val'][name],results['h_goal']['val'][name]
        sources[name]=dict(h_only=h,h_goal=g,goal_improvement=1-g['trajectory_epe_m']/h['trajectory_epe_m'],
            history_gate=h['improvement_vs_zero']>=.05,
            goal_gate=g['improvement_vs_zero']>=.05 and 1-g['trajectory_epe_m']/h['trajectory_epe_m']>=.05 and g['improvement_vs_static_goal']>=.05)
    final=dict(status='COMPLETED',scope='compact input-learning diagnostic; not a matched architecture attribution to WAM',
               sources=sources,history_gate=sum(x['history_gate'] for x in sources.values())>=2,
               goal_gate=sum(x['goal_gate'] for x in sources.values())>=2,elapsed_seconds=time.monotonic()-started)
    atomic_json(args.output/'result.json',final);atomic_json(args.output/'progress.json',final)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','stats','panel','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--budget-seconds',type=float,default=480.)
    args=parser.parse_args();existed=args.output.exists()
    try:run(args)
    except BaseException as exc:
        if not existed and args.output.exists():atomic_json(args.output/'progress.json',dict(status='FAILED',error=repr(exc)))
        raise
