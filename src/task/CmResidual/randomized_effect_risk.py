"""Identified conditional-effect risk differences under known random assignment."""
from __future__ import annotations
import torch


def pseudo_contrast(response,arm,propensity):
    if response.ndim!=2 or response.shape[1]!=3 or propensity.shape!=(len(response),7):
        raise ValueError('response[N,3] and propensity[N,7] required')
    if arm.shape!=(len(response),) or (arm<0).any() or (arm>6).any():raise ValueError('arm0..6 required')
    if (propensity<=0).any() or not torch.allclose(propensity.sum(-1),torch.ones_like(propensity[:,0])):
        raise ValueError('positive normalized propensities required')
    if not torch.isfinite(response).all() or not torch.isfinite(propensity).all():raise ValueError('finite values required')
    result=torch.zeros(len(response),3,3,dtype=response.dtype,device=response.device)
    for axis in range(3):
        plus=2*axis+1;minus=plus+1
        weight=(arm==plus).to(response.dtype)/propensity[:,plus]-(arm==minus).to(response.dtype)/propensity[:,minus]
        result[:,:,axis]=response*weight[:,None]
    return result


def risk_difference(first,second,pseudo):
    if first.shape!=second.shape or first.shape!=pseudo.shape or first.ndim!=3 or first.shape[1:]!=(3,3):
        raise ValueError('matching[N,3,3] contrast matrices required')
    return (first.square()-second.square()-2*(first-second)*pseudo).sum((1,2))
