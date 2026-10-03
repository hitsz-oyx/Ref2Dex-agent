"""Frozen state anchor plus proximity-gated differentiable transport mixture."""
import numpy as np
import torch
from torch import nn

ARMS = ('full', 'state_only', 'shuffled')
STEPS = 1500
STATE = np.asarray([0, 1]*6+[0], np.int64)


class TransportHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(120,128), nn.ReLU(),
                                    nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1,bias=False))
        nn.init.zeros_(self.layers[-1].weight)

    def forward(self, x):
        return self.layers(x)[...,0]


def tokens(raw, arm):
    anchor = raw[:,:1].copy()
    anchor[...,-3:] = 0
    correction = raw[:,STATE] if arm == 'state_only' else raw[:,2:]
    return np.concatenate((anchor, correction),1)


def log_prior(arm):
    other = -np.log(2*np.bincount(STATE)[STATE]) if arm == 'state_only' else np.full(13,-np.log(13))
    return np.r_[np.log(99), other].astype(np.float32)


def predict(model, x, base, endpoints, gate, prior):
    logits = model(x)
    probability = torch.softmax(logits+prior,dim=1)
    # Algebraic B+weighted directions preserves B exactly when gate is false.
    other = probability[:,1:] * gate[:,None].to(probability.dtype)
    effective = torch.cat((1-other.sum(1,keepdim=True), other),1)
    prediction = base + (other.to(base.dtype)[...,None,None]*(endpoints-base[:,None])).sum(1)
    return dict(logits=logits,weights=effective,prediction=prediction)


def flow_loss(prediction, target):
    error = (prediction-target)*1000
    return (error.square().sum(-1)+1e-12).sqrt().mean()


def classify(reports, baselines, near):
    full = reports['full']['episode_epe_mm']
    gates = dict(frozen_state=full<=.9*baselines['frozen_state']['episode_epe_mm'],
                 state_only=full<=.9*reports['state_only']['episode_epe_mm'],
                 shuffled=full<=.95*reports['shuffled']['episode_epe_mm'],
                 persistence=full<=.9*baselines['persistence']['episode_epe_mm'],
                 near_state_only=near['full']['episode_epe_mm']<=.95*near['state_only']['episode_epe_mm'])
    useful = gates['frozen_state'] and gates['state_only']
    return gates, 'PROMISING' if all(gates.values()) else ('UNCLEAR' if useful else 'UNPROMISING')
