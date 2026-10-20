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

    def initialize_current(self, packet):
        """Expand Current421 while preserving its output for every tau input."""
        if not self.with_tau or packet.get('schema') != 'ref2dex.inspire-measured-current-bc.v1':
            raise ValueError('Scheduled686 and a measured Current421 parent required')
        state = self.state_dict()
        source = packet['model']
        if set(source) != set(state):
            raise ValueError('Current421 state fields differ')
        with torch.no_grad():
            for key, value in source.items():
                if key == 'actor.net.0.weight':
                    state[key].zero_()
                    state[key][:, :421].copy_(value)
                elif key in ('actor.x_mean', 'actor.x_scale'):
                    state[key][:421].copy_(value)
                else:
                    state[key].copy_(value)
        self.load_state_dict(state, strict=True)
        self.actor.x_mean[421:].zero_()
        self.actor.x_scale[421:685].fill_(.05)
        self.actor.x_scale[685].fill_(1.)

    @torch.no_grad()
    def shape_report(self, object_points):
        """Track learned-feature drift without changing normalization mid-run."""
        feature = self.shape(object_points)
        normalized = (feature - self.actor.x_mean[345:409]) / self.actor.x_scale[345:409]
        return dict(objects=len(feature), finite=bool(torch.isfinite(normalized).all()),
                    feature_mean_norm=float(feature.mean(0).norm()),
                    feature_std_mean=float(feature.std(0, unbiased=False).mean()),
                    normalized_abs_p99=float(torch.quantile(normalized.abs().flatten(), .99)),
                    normalized_abs_max=float(normalized.abs().max()))

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
