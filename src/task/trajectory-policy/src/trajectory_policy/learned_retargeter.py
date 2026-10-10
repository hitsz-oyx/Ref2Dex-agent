"""Small trajectory-conditioned native action decoder; current measured state only."""
import torch
from torch import nn

from .retargeter_data import ACTIVE, STATE_DIM

SCHEMA = 'ref2dex.learned-native-retargeter.v1'


class LearnedRetargeter(nn.Module):
    def __init__(self, width=128, use_tau=True):
        super().__init__()
        self.use_tau = use_tau
        self.register_buffer('state_mean', torch.zeros(STATE_DIM))
        self.register_buffer('state_scale', torch.ones(STATE_DIM))
        self.register_buffer('tau_mean', torch.zeros(24, 33))
        self.register_buffer('tau_scale', torch.ones(24, 33))
        self.register_buffer('action_mean', torch.zeros(8, 12))
        self.register_buffer('action_scale', torch.ones(8, 12))
        self.state_encoder = nn.Sequential(nn.Linear(STATE_DIM, width), nn.GELU(), nn.Linear(width, width))
        self.tau_encoder = nn.Linear(33, width)
        self.step_embedding = nn.Parameter(torch.randn(24, width)*.02)
        self.action_queries = nn.Parameter(torch.randn(8, width)*.02)
        layer = nn.TransformerDecoderLayer(width, 4, width*4, dropout=0., activation='gelu', batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, 2)
        self.head = nn.Linear(width, 12)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, state, tau, use_tau=None):
        if state.shape != (len(state), STATE_DIM) or tau.shape != (len(state), 24, 33):
            raise ValueError('current state / future hand contract mismatch')
        if use_tau is None:
            use_tau = self.use_tau
        normalized = (tau-self.tau_mean)/self.tau_scale
        if not use_tau:
            normalized = torch.zeros_like(normalized)
        memory = torch.cat((self.tau_encoder(normalized)+self.step_embedding,
            self.state_encoder((state-self.state_mean)/self.state_scale)[:, None]), 1)
        queries = self.action_queries[None].expand(len(state), -1, -1)
        return self.head(self.decoder(queries, memory))*self.action_scale+self.action_mean

    def native_action(self, state, tau, use_tau=None):
        raw = self(state, tau, use_tau)
        native = raw.new_zeros(len(state), 8, 18)
        native[..., list(ACTIVE)] = raw.clamp(-1, 1)
        return native


def load_retargeter(packet, device):
    if packet.get('schema') != SCHEMA:
        raise ValueError('learned native retargeter schema differs')
    model = LearnedRetargeter(packet['width'], packet['use_tau']).to(device)
    model.load_state_dict(packet['model'], strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('nonfinite checkpoint')
    if any((getattr(model, key) <= 0).any() for key in ('state_scale', 'tau_scale', 'action_scale')):
        raise ValueError('nonpositive normalization')
    return model
