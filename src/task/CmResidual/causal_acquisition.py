"""Fixed current-only acquisition and multidirectional native interventions."""
from __future__ import annotations
import torch


def eligible_steps(state_before,done):
    steps=torch.arange(len(done),device=done.device)[:,None]
    # Windows exclude the terminal step and need all five following transitions.
    valid=~done.bool().cumsum(0).bool()
    suffix=valid.clone()
    for offset in range(1,5):
        suffix[:-offset]&=valid[offset:]
        suffix[-offset:]=False
    return (steps>=10)&suffix&state_before[...,49:51].bool().all(-1)


def first_geometry_trigger(eligible,gaps):
    selected=eligible&(gaps<=.02)
    found=selected.any(0)
    times=selected.long().argmax(0)
    return torch.where(found,times,torch.full_like(times,-1))


def axis_pulse(reference,triggers,tick,axis,amplitude):
    action=reference.clone()
    if not 0<=axis<3 or amplitude not in (-.01,0.,.01):
        raise ValueError('frozen wrist translation interventions only')
    chosen=(triggers.to(action.device)==tick)&(triggers.to(action.device)>=0)
    request=action[chosen,axis]+amplitude
    if (request.abs()>1).any():raise ValueError('pulse would clip')
    action[chosen,axis]=request
    return action
