"""Joint physical/Q representations and fresh short/complete data contracts."""
import json
from pathlib import Path
import numpy as np
import torch
from src.task.CmResidual.option_model_policy import trace_extra,known_plan,DYNAMIC

SCHEMA='joint-measured-physical-auxiliary-task-Q-budget-v1'


def policy_groups(motion,seed):
    n=len(motion);assert n in (384,768);rng=torch.Generator(device='cpu').manual_seed(seed+16000);group=torch.full((n,),-1,dtype=torch.long)
    for m in range(3):
        ids=(motion.cpu()==m).nonzero().flatten();assert len(ids)==n//3
        group[ids]=torch.arange(4).repeat_interleave(n//12)[torch.randperm(n//3,generator=rng)]
    return group


def matched_inputs(motion,assignment,seed):
    device=motion.device;n=len(motion);per=n//12;assert n in (384,768);motion=motion.cpu();assignment=assignment.cpu()
    rng=torch.Generator(device='cpu').manual_seed(seed+11000);placements=(torch.rand((n//4,2),generator=rng)*2-1)*.01
    rng=torch.Generator(device='cpu').manual_seed(seed+18000);options=torch.randn((n//4,12),generator=rng)
    offsets=torch.zeros((n,2));raw=torch.zeros((n,12));cluster=torch.full((n,),-1,dtype=torch.long)
    for m in range(3):
        slots=torch.arange(m*per,(m+1)*per)
        for arm in range(4):
            env=((motion==m)&(assignment==arm)).nonzero().flatten();assert len(env)==per;offsets[env]=placements[slots];cluster[env]=slots
            if arm in (2,3):raw[env]=options[slots]*(1 if arm==2 else -1)
    return offsets.to(device),cluster.to(device),raw.to(device)


def read_panel(directory,base):
    directory=Path(directory);report=json.loads((directory/'results.json').read_text());assert report['run_status']=='COMPLETED' and report['statistical_option_collection'];assert json.loads((directory/'panel_audit.json').read_text())['run_status']=='COMPLETED'
    initial=torch.load(directory/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False);meta=json.loads((directory/'physical_metadata.json').read_text());n=len(initial['motion']);assert n in (384,768)
    env=np.arange(n);tick=initial['decision_steps'].numpy();motion=initial['motion'].numpy();stop=initial['phase_stop'].numpy()[motion];assert (tick>=2).all() and (tick+8<stop-74).all() and (tick+8<len(trace['progress'])).all()
    def compact(t):return np.concatenate((trace['normalized_context'].numpy()[t,env],np.ones((n,1),np.float32),((stop+30-t)/np.float32(202))[:,None]),-1).astype(np.float32)
    current=compact(tick);future=compact(tick+8);known=known_plan(initial,tick+8,base);assert np.max(np.abs(future[:,51:72]-known))<=2e-6
    labelled=report['budget_panel_kind']!='short';rows=json.loads((directory/'rows.json').read_text());assert len(rows)==n
    if labelled:
        reward=np.array([r['physical105'] for r in rows],np.float32);progress=trace['progress'].numpy();window=(progress>=stop[None]-74)&(progress<=stop[None]+30);assert np.all(window.sum(0)==105)
        valid=(trace['object_root'].numpy()[...,2]-initial['initial_height'].numpy()[None]>=np.float32(.03))&(trace['clearance'].numpy()>=np.float32(.02));assert np.array_equal(reward.astype(bool),~(window&~valid).any(0))
    else:
        assert report['physical_steps_each']==101 and report['no_terminal_task_labels'] and all(not r['terminal_task_label_available'] and 'physical105' not in r for r in rows);reward=np.full(n,np.nan,np.float32)
    return dict(current=current,extra=trace_extra(trace,tick,env,meta),future=future,future_extra=trace_extra(trace,tick+8,env,meta),known=known,raw=initial['option_raw12'].numpy(),reward=reward,label_available=np.full(n,labelled,bool),motion=motion,arm=initial['policy_group'].numpy(),environment=env,tick=tick)


class JointCritic(torch.nn.Module):
    def __init__(self):
        super().__init__();self.encoder=torch.nn.Sequential(torch.nn.Linear(164,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU());self.task=torch.nn.Linear(64,1);self.physical=torch.nn.Linear(64,131);torch.nn.init.zeros_(self.physical.weight);torch.nn.init.zeros_(self.physical.bias)
    def task_value(self,x):return torch.sigmoid(self.task(self.encoder(x))).flatten()
    def physical_value(self,x):return self.physical(self.encoder(x))


def initialized_critic():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3651);return JointCritic()


def numpy_head(state,x,head):
    x=np.asarray(x,np.float64)
    for layer in (0,2):x=np.maximum(x@state[f'encoder.{layer}.weight'].numpy().astype(np.float64).T+state[f'encoder.{layer}.bias'].numpy(),0)
    x=x@state[head+'.weight'].numpy().astype(np.float64).T+state[head+'.bias'].numpy()
    return 1/(1+np.exp(-np.clip(x,-700,700))) if head=='task' else x
