"""Observed-state to rigid short-horizon endpoint; not a reachability certificate."""
import torch
from torch import nn
from torch.nn import functional as F
from .task_readout import TaskReadout


def rotation_from_columns(columns):
    """Orthonormalize two proposed columns with finite degenerate fallbacks."""
    a,b=columns.split(3,-1)
    first=F.normalize(a,dim=-1,eps=1e-6)
    e1=torch.zeros_like(first);e1[...,0]=1
    first=torch.where((a.norm(dim=-1)<1e-5)[...,None],e1,first)
    second=b-(b*first).sum(-1,keepdim=True)*first
    axis=F.one_hot(first.abs().argmin(-1),3).to(first)
    fallback=axis-(axis*first).sum(-1,keepdim=True)*first
    second=torch.where((second.norm(dim=-1)<1e-5)[...,None],fallback,second)
    second=F.normalize(second,dim=-1,eps=1e-6)
    third=torch.linalg.cross(first,second,dim=-1)
    return torch.stack((first,second,third),-1)


def point_displacement(points,rotation,translation):
    return torch.einsum('bij,bpj->bpi',rotation,points)+translation[:,None]-points


class ShortGoal(nn.Module):
    def __init__(self,state_width,token_width,mean=False):
        super().__init__()
        self.readout=TaskReadout(state_width,token_width,mean=mean)
        del self.readout.value,self.readout.physics
        self.pose=nn.Linear(128,9)
        nn.init.zeros_(self.pose.weight);nn.init.zeros_(self.pose.bias)
        self.register_buffer('identity_columns',torch.tensor([1.,0.,0.,0.,1.,0.]))

    def forward(self,state,tokens,valid,query):
        z=self.readout.encode(state,tokens,valid)
        pose=self.pose(z)
        translation=.05*pose[:,:3]
        rotation=rotation_from_columns(pose[:,3:]+self.identity_columns)
        delta=point_displacement(query,rotation,translation)
        return dict(delta=delta,rotation=rotation,translation=translation)


def fit_standardizer(x,fit,valid=None):
    selected=x[fit].float() if valid is None else x[fit][valid[fit]].float()
    return selected.mean(0),selected.std(0,unbiased=False).clamp_min(.01)


def normalized_inputs(state,tokens,fit,*,mean,valid=None):
    sm,ss=fit_standardizer(state,fit)
    state=(state.float()-sm)/ss
    stats=dict(state_mean=sm,state_scale=ss)
    if mean:
        tokens=F.layer_norm(tokens.float().reshape(-1,3,768),(768,)).flatten(1)
    else:
        tm,ts=fit_standardizer(tokens,fit,valid)
        tokens=torch.where(valid[...,None],(tokens.float()-tm)/ts,0)
        stats.update(token_mean=tm,token_scale=ts)
    return state,tokens,stats
