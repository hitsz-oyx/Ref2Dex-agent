"""Learn absolute reference-relative targets, with explicit native PD feedback."""
import torch
from src.task.CmResidual.finger_preload import preload_target
SCALES=(.02,)*3+(.10,)*3+(.40,)*12
SCHEMA='hold70-reference-target-v1'
NULL=(7,9,11,13,16,17)

def residual_labels(goal,reference):
    y=(goal-reference)/torch.tensor(SCALES,device=goal.device)
    y[...,NULL]=0
    if not torch.isfinite(y).all() or (y.abs()>1+1e-6).any():raise ValueError('target residual exceeds fixed range')
    return y

def target_from_residual(reference,residual,lower,upper):
    if residual.shape!=reference.shape or residual.shape[-1]!=18 or not torch.isfinite(residual).all() or (residual.abs()>1+1e-6).any():raise ValueError('bounded target residual schema')
    goal=reference+residual*torch.tensor(SCALES,device=reference.device)
    goal[:,14]=goal[:,14].clamp(float(lower[14]),float(upper[14]))
    return preload_target(goal,torch.zeros(len(goal),device=goal.device),lower,upper)
