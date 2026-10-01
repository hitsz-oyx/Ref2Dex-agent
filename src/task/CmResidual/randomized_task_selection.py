"""Frozen factual candidate scores and explicit bounded policy selection."""
from __future__ import annotations
import torch
from src.task.CmResidual.direct_randomized_response import context


class FrozenFactualScores:
    def __init__(self, checkpoints, device):
        self.models=[];self.scales=[]
        for checkpoint in checkpoints:
            if checkpoint['method']!='factual' or checkpoint['updates']!=1000:
                raise ValueError('only frozen simple factual control allowed')
            model=torch.nn.Sequential(torch.nn.Linear(87,128),torch.nn.SiLU(),
                torch.nn.Linear(128,128),torch.nn.SiLU(),torch.nn.Linear(128,3)).to(device).eval()
            model.load_state_dict(checkpoint['model']);model.requires_grad_(False)
            self.models.append(model)
            self.scales.append({key:checkpoint[key].to(device) for key in ('mean','std','target_mean','target_std')})
        if len(self.models)!=3:raise ValueError('three equally weighted seeds required')

    @torch.no_grad()
    def __call__(self, state, reference_action, bridge):
        x=context(state,reference_action,bridge);n=len(x)
        pulse=torch.cat((torch.zeros(1,6,device=x.device),torch.eye(6,device=x.device)))
        candidates=torch.cat((x[:,None].expand(-1,7,-1),pulse[None].expand(n,-1,-1)),-1).reshape(-1,87)
        outputs=[]
        for model,scale in zip(self.models,self.scales):
            outputs.append((model((candidates-scale['mean'])/scale['std'])*scale['target_std']+scale['target_mean']).reshape(n,7,3))
        return torch.stack(outputs).mean(0)


def candidate_validity(reference_action):
    valid=torch.ones(len(reference_action),7,dtype=torch.bool,device=reference_action.device)
    for arm in range(1,7):
        value=reference_action[:,(arm-1)//2]+(.01 if arm%2 else -.01)
        valid[:,arm]=(value>=-1)&(value<=1)
    return valid


def choose_actions(policy, scores, global_scores, valid, random_choice):
    conditional=scores.masked_fill(~valid,-torch.inf).argmax(-1)
    global_arm=global_scores[None].expand_as(scores).masked_fill(~valid,-torch.inf).argmax(-1)
    selected=torch.zeros_like(policy)
    selected[policy==1]=random_choice[policy==1]
    selected[policy==2]=global_arm[policy==2]
    selected[policy==3]=conditional[policy==3]
    if not valid[torch.arange(len(policy),device=policy.device),selected].all():raise ValueError('invalid selected action')
    return selected
