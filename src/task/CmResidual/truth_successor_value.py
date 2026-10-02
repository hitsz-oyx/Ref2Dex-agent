"""Fixed-policy factual successor-value upperbound, no deployable future inputs."""
from pathlib import Path
import json
import numpy as np
import torch

SCHEMA='truth-successor-value-current72-command12-v1'
VARIANTS=('current_value','direct_q','oracle_successor')

def read_panel(directory):
    directory=Path(directory);initial=torch.load(directory/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False);rows=json.loads((directory/'rows.json').read_text())
    assert len(rows)==768 and [x['environment'] for x in rows]==list(range(768))
    progress=trace['progress'];assert torch.equal(progress,torch.arange(1,203)[:,None].expand(202,768))
    arm=initial['policy_group'];stops=initial['phase_stop'][initial['motion']];end=stops+30
    valid=(trace['object_root'][...,2]-initial['initial_height'][None]>=.03)&(trace['clearance']>=.02)
    window=(progress>=stops[None]-74)&(progress<=end[None]);assert (window.sum(0)==105).all()
    bad=window&~valid;reward=torch.tensor([r['physical105'] for r in rows],dtype=torch.float32);assert torch.equal(reward.bool(),~bad.any(0))
    post_fail=bad.cumsum(0)>0;pre_fail=torch.cat((torch.zeros_like(post_fail[:1]),post_fail[:-1]),0)
    clock=torch.arange(202)[:,None].expand(202,768);pre_done=pre_fail|(clock>=end[None]);post_done=post_fail|((clock+1)>=end[None])
    eligible=(arm[None]>0)&~pre_done&(clock<201);ticks,envs=eligible.nonzero(as_tuple=True)
    assert len(ticks)>1024 and (ticks+1<202).all()
    current=torch.cat((trace['normalized_context'][ticks,envs],(~pre_fail[ticks,envs]).float()[:,None],((end[envs]-ticks).clamp_min(0).float()/202)[:,None]),-1)
    successor=torch.cat((trace['normalized_context'][ticks+1,envs],(~post_fail[ticks,envs]).float()[:,None],((end[envs]-ticks-1).clamp_min(0).float()/202)[:,None]),-1)
    command=trace['executed_action12'][ticks,envs];target=reward[envs];future_known=post_done[ticks,envs];future_value=(~post_fail[ticks,envs]).float()
    assert torch.equal(target[future_known],future_value[future_known])
    clusters=torch.full((768,),-1,dtype=torch.long)
    for group in (1,2,3):
        ids=(arm==group).nonzero().flatten();assert len(ids)==192;clusters[ids]=torch.arange(192)
    return dict(current=current,next=successor,command=command,target=target,next_known=future_known,next_value=future_value,tick=ticks,environment=envs,motion=initial['motion'][envs],cluster=clusters[envs],timebin=(ticks*16//202).clamp_max(15),eligible_rows=len(ticks),episodes=int(envs.unique().numel()),next_known_fraction=float(future_known.float().mean()))

def features(panel,variant):
    x=panel['next'] if variant=='oracle_successor' else panel['current']
    a=panel['command'] if variant=='direct_q' else torch.zeros_like(panel['command'])
    return torch.cat((x,a),-1)

def initialized_model():
    torch.manual_seed(3531)
    return torch.nn.Sequential(torch.nn.Linear(84,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1),torch.nn.Sigmoid())

def predictions(model,x,panel,variant):
    p=model(x).flatten()
    return torch.where(panel['next_known'],panel['next_value'],p) if variant=='oracle_successor' else p

def phase_template(fit):
    table=torch.full((3,16),float(fit['target'].mean()))
    for m in range(3):
        for b in range(16):
            mask=(fit['motion']==m)&(fit['timebin']==b)
            if mask.any():table[m,b]=fit['target'][mask].mean()
    return table

def analyze(panel,pred):
    labels=panel['target'].double().numpy();cluster=panel['cluster'].numpy();weight=np.bincount(cluster,minlength=192).astype(np.float64)
    risk={k:(v.double().numpy()-labels)**2 for k,v in pred.items()};mse={k:float(v.mean()) for k,v in risk.items()}
    indices=np.random.default_rng(3533).integers(0,192,size=(2000,192));interval={};gates={}
    for k in pred:
        if k=='oracle_successor':continue
        grouped=np.bincount(cluster,weights=risk['oracle_successor']-risk[k],minlength=192)
        samples=grouped[indices].sum(1)/weight[indices].sum(1);interval[k]=np.quantile(samples,[.025,.975]).tolist()
        gates[k]=dict(gain1percent=mse['oracle_successor']<=.99*mse[k],paired_upper95_negative=interval[k][1]<0)
    return dict(brier=mse,paired_difference_interval95=interval,gates=gates,label='PROMISING' if all(all(q.values()) for q in gates.values()) else 'UNPROMISING')
