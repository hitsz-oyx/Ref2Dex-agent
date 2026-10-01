"""Frozen hand-support intervention; changes native hand targets only."""
import torch
from src.task.CmResidual.frame0_tracking import tracking_target


def assign_witness_arms(motion,seed):
    g=torch.Generator(device='cpu').manual_seed(seed+9000)
    result=torch.full((len(motion),),-1,dtype=torch.long)
    for m in range(3):
        ids=(motion.cpu()==m).nonzero().flatten()
        if len(ids)!=32:raise ValueError('frozen coverage')
        result[ids]=torch.arange(2).repeat_interleave(16)[torch.randperm(32,generator=g)]
    return result


def witness_target(reference_q,next_reference_progress,actual_next_progress,motion,assignment,lift_start,phase_stop,lower,upper):
    target,_=tracking_target(reference_q,next_reference_progress,motion,torch.ones_like(assignment),lift_start,lower,upper)
    release=(actual_next_progress>phase_stop[motion])&assignment.bool()
    target[release,2]+=.20
    target[release,6:]=0
    return target,release
