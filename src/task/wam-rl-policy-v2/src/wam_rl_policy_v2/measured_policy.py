"""Shared Shadow MLP capacity, with a trainable Inspire CAD feature encoder."""
import torch
from torch import nn
from .bc_mlp import NormalizedCommandMLP
from .object_shape import ObjectShapeEncoder
from .executor_contract import TASK_SCALE


class MeasuredInspirePolicy(nn.Module):
    def __init__(self,*,with_tau=False):
        super().__init__();self.with_tau=with_tau;self.shape=ObjectShapeEncoder()
        self.actor=NormalizedCommandMLP(686 if with_tau else 421,312,12)
        self.actor.x_scale[411:421].copy_(torch.as_tensor(TASK_SCALE))

    def forward(self,measured,task,object_points,*,tau=None,tau_valid=None):
        if measured.shape[-1]!=347 or task.shape!=measured.shape[:-1]+(10,):raise ValueError('measured347/task10 required')
        feature=self.shape(object_points)
        robot=torch.cat((measured[...,:345],feature,measured[...,345:]),-1)
        parts=[robot,task]
        if self.with_tau:
            if tau is None or tau_valid is None or tau.shape!=measured.shape[:-1]+(8,11,3):raise ValueError('explicit native-time canonical tau and validity required')
            valid=tau_valid.to(dtype=torch.bool).reshape(measured.shape[:-1]+(1,))
            parts.extend((torch.where(valid,tau.flatten(-3),0.),valid.to(measured.dtype)))
        elif tau is not None or tau_valid is not None:raise ValueError('current policy does not consume tau')
        return self.actor(torch.cat(parts,-1))
