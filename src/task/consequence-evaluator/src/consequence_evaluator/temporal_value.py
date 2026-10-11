"""Outcome/time weak supervision; geometry defines labels, never dense rewards."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .contracts import K, FUTURE_DIM
from .model import Evaluator
from .supervision import consecutive
from .value_outcomes import PARAMETERS

SCHEMA='ref2dex.consequence-temporal-value.v1'
LABEL_RULE='final-controlled-completion-signed-relative-time-positive-recovery-mask-v2'


def episode_labels(trace, velocity):
    """Permit regrasp, then stable hold and safe placing; mask local recovery.

    Failed episodes retain weak negative supervision. The recovery mask prevents
    failed local behavior in an eventually successful episode becoming positive.
    All physical cutoffs inherit the already fixed binary task criterion.
    """
    valid=trace['valid']; near=trace['near']; support=trace['supported']
    velocity=np.asarray(velocity)
    if velocity.shape!=(len(valid),6) or not np.isfinite(velocity).all():
        raise ValueError('measured full-episode velocity required')
    place=int(trace['place_start']); clock=np.arange(len(valid))
    stable=trace['held_run']>=PARAMETERS['stable_frames']
    completions=np.flatnonzero(stable & (clock<place))
    missing=valid & ~near & ~support
    loss_run=consecutive(missing)
    unsafe=(loss_run>=PARAMETERS['lost_geometry_frames'])
    unsafe |= valid & ~near & ~support & (velocity[:,2]<-PARAMETERS['unheld_fall_mps'])
    # Once a sustained loss is confirmed, mask its entire onset, not only
    # the sixth frame at which the persistence criterion becomes true.
    confirmed_loss=np.zeros(len(valid),bool)
    for tick in np.flatnonzero(loss_run>=PARAMETERS['lost_geometry_frames']):
        confirmed_loss[tick-PARAMETERS['lost_geometry_frames']+1:tick+1]=True
    success=False
    if len(completions):
        last=int(completions[-1])
        success=bool(not trace['drop'][last+1:place].any() and not unsafe[max(last+1,place):].any()
                     and consecutive(trace['settled'])[-1]>=PARAMETERS['settled_frames'])
    recovery=np.zeros(len(valid),bool); grasped=False; uncertain=False
    grasp=(consecutive(near)>=PARAMETERS['grasp_frames']) & (trace['height']>=PARAMETERS['lift_m']) & ~support
    for tick in range(len(valid)):
        # Supported, intended final release is part of the successful task.
        failure=bool(confirmed_loss[tick] or unsafe[tick] or (tick<place and trace['drop'][tick]))
        if grasped and failure:
            uncertain=True
        if uncertain and stable[tick]:
            uncertain=False
        recovery[tick]=uncertain
        grasped |= bool(grasp[tick])
    return int(success), recovery


def relative_time_labels(success, tick, length):
    tick=np.asarray(tick); length=np.asarray(length); success=np.asarray(success)
    if ((length-tick<K).any() or (tick<0).any() or not np.isin(success,[0,1]).all()):
        raise ValueError('binary episode outcomes and complete24step windows required')
    return np.where(success==1,1.,-1.)*K/(length-tick)


def soft_bins(target,bins=41):
    if bins<3 or not torch.isfinite(target).all() or (target.abs()>1+1e-6).any():
        raise ValueError('bounded signed relative-time targets required')
    position=(target.clamp(-1,1)+1)*(bins-1)/2
    low=position.floor().long(); high=position.ceil().long(); weight=position-low
    result=torch.zeros(*target.shape,bins,device=target.device,dtype=target.dtype)
    result.scatter_add_(-1,low[...,None],(1-weight)[...,None])
    result.scatter_add_(-1,high[...,None],weight[...,None])
    return result


class TemporalValue(nn.Module):
    """Same128-wide transformer in HA/HAZ; scalar is a signed-bin expectation."""
    def __init__(self,history_dim,width=128,layers=2,bins=41):
        super().__init__()
        base=Evaluator(history_dim,width,layers)
        self.history=base.history; self.action=base.action; self.future=base.future
        self.time=base.time; self.encoder=base.encoder
        self.readout=nn.Linear(width,bins)
        self.register_buffer('centers',torch.linspace(-1,1,bins))

    def forward(self,history,action,future,use_future=True):
        if action.shape[1:]!=(K,18) or future.shape[1:]!=(K,FUTURE_DIM):
            raise ValueError('decision-known plan24 and physical future24 required')
        if not use_future:
            future=torch.zeros_like(future)
        chunk=self.action(action)+self.future(future)
        tokens=torch.cat((self.history(history)[:,None],chunk),1)+self.time
        logits=self.readout(self.encoder(tokens)[:,1:].mean(1))
        score=(logits.softmax(-1)*self.centers).sum(-1)
        return dict(logits=logits,score=score)


def temporal_loss(logits,target):
    return -(soft_bins(target,logits.shape[-1])*F.log_softmax(logits,-1)).sum(-1).mean()
