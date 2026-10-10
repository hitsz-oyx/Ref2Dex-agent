"""Independent Gaussian trajectory actor; reference behavior is offline only."""
import torch
from torch import nn

from .history import INPUT_DIM, SCHEMA as HISTORY_SCHEMA
from .dense_decoder import LATENT_DIM

SCHEMA = 'ref2dex.independent-dense-trajectory-actor.v1'


class HistoryActor(nn.Module):
    def __init__(self, width=512):
        super().__init__()
        self.register_buffer('history_mean', torch.zeros(INPUT_DIM))
        self.register_buffer('history_scale', torch.ones(INPUT_DIM))
        self.register_buffer('action_mean', torch.zeros(LATENT_DIM))
        self.register_buffer('action_scale', torch.ones(LATENT_DIM))
        self.mean_net = nn.Sequential(nn.Linear(INPUT_DIM, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(), nn.Linear(width, LATENT_DIM))
        # Fixed physical coordinate units: 1mm translation / .01rad per frame.
        # BC trains only the mean. PPO may subsequently optimize this parameter.
        self.log_std = nn.Parameter(torch.full((LATENT_DIM,), -2.302585093))

    def forward(self, history):
        if history.shape[-1] != INPUT_DIM:
            raise ValueError('measured-history actor input mismatch')
        normalized = (history-self.history_mean)/self.history_scale
        return self.mean_net(normalized)*self.action_scale+self.action_mean

    def distribution(self, history):
        return torch.distributions.Normal(self(history), self.log_std.exp())


def load_actor(packet, device):
    if packet.get('schema') != SCHEMA or packet.get('history_schema') != HISTORY_SCHEMA:
        raise ValueError('independent actor/history schema mismatch')
    actor = HistoryActor(packet['width']).to(device)
    actor.load_state_dict(packet['model'])
    if not all(torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise ValueError('finite actor weights required')
    if (actor.history_scale <= 0).any() or (actor.action_scale <= 0).any():
        raise ValueError('positive actor normalization required')
    return actor
