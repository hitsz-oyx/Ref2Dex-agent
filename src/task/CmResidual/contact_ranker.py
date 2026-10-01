"""Physical consequence ensemble and honest randomized policy measurement."""
from __future__ import annotations

import math
import torch
from torch import nn

BASE_INDEX=4


def frame_group_split(motion,frame,seed=9331):
    """Labels never enter this split; duplicate starts across seeds stay together."""
    generator=torch.Generator().manual_seed(seed)
    mapping={}
    for m in sorted(set(motion.tolist())):
        groups=sorted(set(int(f) for mm,f in zip(motion.tolist(),frame.tolist()) if mm==m))
        if len(groups)<5:raise ValueError('insufficient distinct reference-frame groups')
        order=torch.randperm(len(groups),generator=generator).tolist()
        nfit=max(1,int(.6*len(groups)));ncal=max(1,(len(groups)-nfit)//2)
        for i,index in enumerate(order):mapping[(m,groups[index])]=0 if i<nfit else 1 if i<nfit+ncal else 2
    return torch.tensor([mapping[(int(m),int(f))] for m,f in zip(motion,frame)],dtype=torch.long),mapping


def physical_history(history,rest):
    value=history.clone().float()
    value[:,:,:3]-=value[:,:,36:39]
    value[:,:,36:38]-=history[:,-1:,36:38]
    value[:,:,38]-=rest[:,None]
    q=torch.nn.functional.normalize(value[:,:,39:43],dim=-1)
    pivot=q.gather(-1,q.abs().argmax(-1,keepdim=True))
    value[:,:,39:43]=q*torch.where(pivot<0,-1.,1.)
    return value


def shuffled_numeric_actions(actions,assignment,seed=9431):
    generator=torch.Generator().manual_seed(seed)
    offsets=torch.randint(1,6,(len(actions),),generator=generator)
    corrupted=(assignment.cpu()+offsets)%6
    return actions[torch.arange(len(actions),device=actions.device),corrupted.to(actions.device)]


class ConsequenceNetwork(nn.Module):
    def __init__(self,state_only=False):
        super().__init__();self.state_only=state_only
        self.history=nn.GRU(69,32,batch_first=True)
        self.context=nn.Sequential(nn.Linear(123,96),nn.SiLU())
        self.action=nn.Linear(18,96)
        self.head=nn.Sequential(nn.Linear(288,64),nn.SiLU(),nn.Linear(64,18 if state_only else 3))

    def forward(self,history,action=None,context=None):
        if context is None or context.shape!=(len(history),22):
            raise ValueError('current base intent/motion/phase context required')
        _,hidden=self.history(history)
        context=self.context(torch.cat((hidden[-1],history[:,-1],context),-1))
        if self.state_only:
            zeros=torch.zeros_like(context)
            return self.head(torch.cat((context,zeros,zeros),-1)).reshape(-1,6,3)
        if action is None:raise ValueError('numeric action required')
        a=torch.tanh(self.action(action))
        return self.head(torch.cat((context,a,context*a),-1))


def all_predictions(model,history,candidate,context):
    if model.state_only:return model(history,context=context)
    n=len(history)
    h=history[:,None].expand(-1,6,-1,-1).reshape(n*6,10,69)
    c=context[:,None].expand(-1,6,-1).reshape(n*6,22)
    return model(h,candidate.reshape(n*6,18),context=c).reshape(n,6,3)


def policy_choice(predictions,margin_mm,*,drop_supported,eligible):
    """Uncertainty-aware local rule, not a certified causal confidence bound.

    predictions [ensemble,B,6,3] in physical mm/probability units. Calibrated
    factual error provides a fixed margin; ensemble effect dispersion is a
    heuristic, since randomized data have no individual counterfactual truth.
    """
    mean=predictions.mean(0)
    difference=predictions[:,:,:,0]-predictions[:,:,BASE_INDEX:BASE_INDEX+1,0]
    lower=difference.mean(0)-difference.std(0,unbiased=False)
    safe=(mean[:,:,1]>=mean[:,BASE_INDEX:BASE_INDEX+1,1]-.05)
    safe &= ((mean[:,:,2]<=mean[:,BASE_INDEX:BASE_INDEX+1,2]+.05) | ~eligible[:,None])
    if not drop_supported:safe[eligible]=False
    scores=lower.masked_fill(~safe,-torch.inf)
    choice=scores.argmax(-1)
    rows=torch.arange(len(choice),device=choice.device)
    intervene=(choice!=BASE_INDEX)&(scores[rows,choice]>margin_mm)
    choice=torch.where(intervene,choice,BASE_INDEX)
    return choice,mean,lower


def randomized_value(choice,assignment,outcome,episode_ids):
    """Horvitz value plus episode-cluster descriptive uncertainty, p=1/6."""
    matched=choice.cpu()==assignment.cpu()
    y=outcome.cpu().double()
    contributions=matched.double()*y*6
    mean=float(contributions.mean())
    episodes=sorted(set(episode_ids))
    clusters={name:[] for name in episodes}
    for index,name in enumerate(episode_ids):clusters[name].append(index)
    # Center row contributions before aggregation: episodes of different
    # lengths do not implicitly receive equal weight in this window estimand.
    sums=torch.tensor([float((contributions[indices]-mean).sum()) for indices in clusters.values()],dtype=torch.double)
    se=math.sqrt(float(sums.square().sum())*len(episodes)/max(len(episodes)-1,1))/len(y)
    return dict(value=mean,approximate_episode_cluster95=[mean-1.96*se,mean+1.96*se],
                matched_windows=int(matched.sum()),matched_episodes=len({episode_ids[i] for i in matched.nonzero().flatten().tolist()}),
                episodes=len(episodes),windows=len(y),
                boundary='randomized window-policy value; no individual oracle/regret or formal validation')
