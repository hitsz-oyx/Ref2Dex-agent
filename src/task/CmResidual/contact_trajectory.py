"""Physical trajectories with an explicit, numeric base-relative action effect."""
from __future__ import annotations
import torch
from torch import nn

BASE_INDEX=4


def trajectory_targets(future,contact,trigger_z,rest_z):
    if future.shape[-2:]!=(10,49) or contact.shape!=future.shape[:-1]:
        raise ValueError('complete10step physical trajectory required')
    if not torch.isfinite(future).all():raise ValueError('nonfinite trajectory')
    contact=contact.bool();z=future[...,38]
    height=(z-trigger_z[...,None])*1000
    joint=contact[...,-3:].all(-1)
    retained=height[...,-3:].amin(-1).clamp_min(0)*joint.float()
    seen=trigger_z-rest_z>=.03
    release=torch.zeros_like(seen);lost=torch.zeros_like(trigger_z,dtype=torch.long)
    for step in range(10):
        seen=seen|(z[...,step]-rest_z>=.03)
        lost=torch.where(seen&~contact[...,step],lost+1,0)
        release|=seen&((z[...,step]-rest_z<.02)|(lost>=6))
    return dict(height_mm=height,contact=contact.float(),joint_contact=joint.float(),
                release=release.float(),retained_lift_mm=retained,initially_lifted=(trigger_z-rest_z>=.03))


class TrajectoryNetwork(nn.Module):
    def __init__(self,state_only=False):
        super().__init__();self.state_only=state_only
        self.history=nn.GRU(69,32,batch_first=True)
        self.context=nn.Sequential(nn.Linear(123,96),nn.SiLU())
        self.action=nn.Linear(18,96)
        if state_only:
            self.slot_head=nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Linear(64,6*22))
        else:
            self.base_head=nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Linear(64,22))
            self.effect_head=nn.Sequential(nn.Linear(288,64),nn.SiLU(),nn.Linear(64,22))
            nn.init.zeros_(self.effect_head[-1].weight);nn.init.zeros_(self.effect_head[-1].bias)

    def components(self,history,action,base_action,context):
        if context.shape!=(len(history),22):raise ValueError('pre-action context required')
        _,hidden=self.history(history)
        x=self.context(torch.cat((hidden[-1],history[:,-1],context),-1))
        if self.state_only:return self.slot_head(x).reshape(-1,6,22),None
        if action.shape!=(len(history),18) or base_action.shape!=action.shape:raise ValueError('numeric current/base action required')
        a=torch.tanh(self.action(action));b=torch.tanh(self.action(base_action))
        effect=self.effect_head(torch.cat((x,a,x*a),-1))-self.effect_head(torch.cat((x,b,x*b),-1))
        return self.base_head(x),effect

    def forward(self,history,action=None,base_action=None,context=None):
        baseline,effect=self.components(history,action,base_action,context)
        return baseline if self.state_only else baseline+effect


def all_trajectories(model,history,candidate,context):
    if model.state_only:return model(history,context=context)
    n=len(history)
    return model(history[:,None].expand(-1,6,-1,-1).reshape(n*6,10,69),
                 candidate.reshape(n*6,18),candidate[:,BASE_INDEX:BASE_INDEX+1].expand(-1,6,-1).reshape(n*6,18),
                 context[:,None].expand(-1,6,-1).reshape(n*6,22)).reshape(n,6,22)


def decode_trajectories(raw,height_mean,height_scale):
    height=raw[...,:10]*height_scale+height_mean
    probability=raw[...,10:].sigmoid()
    score=height[...,-3:].amin(-1).clamp_min(0)*probability[...,10]
    return dict(height_mm=height,contact=probability[...,:10],joint_contact=probability[...,10],
                release=probability[...,11],retained_score_mm=score)


def retained_choice(predictions,margin_mm,release_supported):
    score=predictions['retained_score_mm']
    difference=score-score[:,:,BASE_INDEX:BASE_INDEX+1]
    lower=difference.mean(0)-difference.std(0,unbiased=False)
    contact=predictions['joint_contact'].mean(0)
    safe=contact>=contact[:,BASE_INDEX:BASE_INDEX+1]-.02
    risk=predictions['release']-predictions['release'][:,:,BASE_INDEX:BASE_INDEX+1]
    safe&=(risk.mean(0)+risk.std(0,unbiased=False)<=0)
    if not release_supported:safe[:]=False
    lower=lower.masked_fill(~safe,-torch.inf)
    proposal=lower.argmax(-1)
    rows=torch.arange(len(proposal),device=proposal.device)
    proposal=torch.where((proposal!=BASE_INDEX)&(lower[rows,proposal]>margin_mm),proposal,BASE_INDEX)
    return proposal,lower
