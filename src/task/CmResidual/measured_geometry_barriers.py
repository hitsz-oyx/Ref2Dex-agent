"""Observed SDK geometry for task-barrier transition information, not contact truth."""
import numpy as np
import torch
from scripts.analyze_static_hold_feasibility import rotation
SCHEMA='measured-sdk-geometry80-current70-action12-barriers2-v1'

def relative_geometry(object_root,body_position,body_quaternion):
    inv=rotation(object_root[...,3:7]).swapaxes(-1,-2)
    pos=np.einsum('nij,nbj->nbi',inv,body_position-object_root[:,None,:3])
    rot=np.einsum('nij,nbjk->nbik',inv,rotation(body_quaternion))
    return pos,rot[...,:2].reshape(len(pos),30)

def initialized_model(seed=3211):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);m=torch.nn.Sequential(torch.nn.Linear(162,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,2))
        torch.nn.init.zeros_(m[-1].weight);torch.nn.init.zeros_(m[-1].bias)
    return m
