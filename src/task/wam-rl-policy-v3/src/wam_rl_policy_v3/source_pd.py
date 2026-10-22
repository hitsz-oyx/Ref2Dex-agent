"""Exact inverse of the production Inspire PD map, never clipped for control."""
import numpy as np
from .inspire_contract import ACTIVE,expand_action


def source_pd_action(target,q,offset,scale):
    target=np.asarray(target,np.float64).copy();q=np.asarray(q,np.float64)
    offset=np.asarray(offset,np.float64);scale=np.asarray(scale,np.float64)
    if target.shape!=q.shape or target.shape[-1]!=18 or offset.shape!=(18,) or scale.shape!=(18,) or not np.isfinite(np.r_[target.ravel(),q.ravel(),offset,scale]).all() or (scale<=0).any():raise ValueError('finite production PD map required')
    target[...,3:6]+=2*np.pi*np.round((q[...,3:6]-target[...,3:6])/(2*np.pi))
    value=(target-offset)/scale
    value[...,:6]=(target[...,:6]-q[...,:6]-offset[:6])/scale[:6]
    value[...,6:]=2*value[...,6:]-1
    active=value[...,ACTIVE]
    excess=np.maximum(np.abs(active)-1,0)
    if excess.max()>1e-6:raise ValueError('source-PD needs saturation: max excess='+str(float(excess.max())))
    # Float32 roundoff at an exactly saturated source joint is bounded/recorded;
    # substantive saturation above is rejected rather than silently clipped.
    canonical=np.clip(active,-1,1).astype(np.float32)
    expanded=expand_action(canonical)
    represented=(expanded+1)/2*scale+offset
    represented[...,:6]=q[...,:6]+offset[:6]+expanded[...,:6]*scale[:6]
    for dst,src,ratio in ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)):
        represented[...,dst]=represented[...,src]*ratio
    if np.abs(represented-target).max()>2e-5:raise ValueError('source target violates native coupling or cannot be represented')
    return canonical,target,float(excess.max())
