"""Fixed label-time controller; no learned policy and no object intervention."""
import torch
from src.task.CmResidual.finger_preload import preload_target


def assign_tracking_arms(motion,seed):
    g=torch.Generator(device='cpu').manual_seed(seed+8000)
    result=torch.full((len(motion),),-1,dtype=torch.long)
    for m in range(3):
        ids=(motion.cpu()==m).nonzero().flatten()
        if len(ids)!=32:raise ValueError('frozen coverage')
        result[ids]=torch.arange(2).repeat_interleave(16)[torch.randperm(32,generator=g)]
    return result


def tracking_target(reference_q,next_progress,motion,assignment,lift_start,lower,upper):
    fraction=((next_progress-(lift_start[motion]-15))/15).clamp(0,1)
    dose=.30*assignment*fraction
    base=reference_q.clone();base[:,14]=base[:,14].clamp(float(lower[14]),float(upper[14]))
    return preload_target(base,dose,lower,upper),dose
