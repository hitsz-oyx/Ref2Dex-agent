"""Learned native decoder with action-aligned local hand motion queries."""
import torch
from torch import nn

from .learned_retargeter import LearnedRetargeter
from .retargeter_data import STATE_DIM

SCHEMA = 'ref2dex.local-motion-retargeter.v1'


class LocalMotionRetargeter(LearnedRetargeter):
    def __init__(self, width=128, use_tau=True):
        super().__init__(width, use_tau)
        self.local_encoder = nn.Sequential(nn.Linear(66, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, state, tau, use_tau=None):
        if state.shape != (len(state), STATE_DIM) or tau.shape != (len(state), 24, 33):
            raise ValueError('current state / future hand contract mismatch')
        if use_tau is None:
            use_tau = self.use_tau
        normalized = (tau-self.tau_mean)/self.tau_scale
        local = tau[:, :8]
        previous = torch.cat((torch.zeros_like(local[:, :1]), local[:, :-1]), 1)
        motion = torch.cat((local, local-previous), -1)/.01
        if not use_tau:
            normalized = torch.zeros_like(normalized)
            motion = torch.zeros_like(motion)
        memory = torch.cat((self.tau_encoder(normalized)+self.step_embedding,
            self.state_encoder((state-self.state_mean)/self.state_scale)[:, None]), 1)
        queries = self.action_queries[None]+self.local_encoder(motion)
        return self.head(self.decoder(queries, memory))*self.action_scale+self.action_mean


def load_retargeter(packet, device):
    if packet.get('schema') != SCHEMA:
        raise ValueError('local motion retargeter schema differs')
    model = LocalMotionRetargeter(packet['width'], packet['use_tau']).to(device)
    model.load_state_dict(packet['model'], strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('nonfinite checkpoint')
    if any((getattr(model, key) <= 0).any() for key in ('state_scale', 'tau_scale', 'action_scale')):
        raise ValueError('nonpositive normalization')
    return model
