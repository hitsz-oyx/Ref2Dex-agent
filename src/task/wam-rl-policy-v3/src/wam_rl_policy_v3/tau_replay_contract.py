"""Contracts for clocked reference deltas; these are not live-conditioned oracles."""
import numpy as np
from .inspire_contract import expand_action


def reference_tau(reference,frame,*,horizon=8,available=True):
    reference=np.asarray(reference)
    if reference.ndim!=3 or reference.shape[1:]!=(11,3) or len(reference)<1 or frame<0:raise ValueError('reference Tx11x3 and nonnegative reference clock required')
    valid=bool(available and frame>=3 and frame+horizon<len(reference))
    tau=(reference[frame+np.arange(1,horizon+1)]-reference[frame]).astype(np.float32) if valid else np.zeros((horizon,11,3),np.float32)
    return tau,valid


def require_saved_clock(data):
    controls=len(data['action']);tick=np.asarray(data['tick']).reshape(-1)
    if len(data['q'])!=controls+1 or not np.array_equal(tick,np.arange(controls+1)):
        raise ValueError('state[t] -> action[t] -> state[t+1] consecutive clock required')
    if len(data['tau'])!=controls+1 or len(data['tau_valid'])!=controls+1:raise ValueError('one tau/valid per observed state required')


def pd_targets(action,q,offset,scale):
    """Independent NumPy copy of the native PD conversion, including coupling."""
    full=expand_action(action).astype(np.float64);q=np.asarray(q,np.float64)
    target=np.asarray(offset)+np.asarray(scale)*np.concatenate((full[...,:6],(1+full[...,6:])/2),-1)
    target[...,:6]+=q[...,:6]
    for dst,src,ratio in ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)):
        target[...,dst]=target[...,src]*ratio
    return target


def first_exceeds(value,threshold):
    hits=np.flatnonzero(np.asarray(value)>threshold)
    return int(hits[0]) if len(hits) else None


def bouts(mask):
    edges=np.flatnonzero(np.diff(np.r_[False,np.asarray(mask,bool),False].astype(np.int8)))
    return [(int(a),int(b)) for a,b in zip(edges[::2],edges[1::2])]
