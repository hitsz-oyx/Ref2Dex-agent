"""Actual PD-target supervision; all controller feedback is measured at this tick."""
import numpy as np
import torch
from torch import nn

from .retargeter_data import ACTIVE, STATE_DIM, features

SCHEMA = 'ref2dex.competent-pd-retargeter.v1'
COUPLING = ((7, 6, 1.05), (9, 8, 1.05), (11, 10, 1.05),
            (13, 12, 1.05), (16, 15, .6), (17, 15, .8))


def couple(target):
    target = target.clone()
    for distal, parent, ratio in COUPLING:
        target[..., distal] = target[..., parent] * ratio
    return target


def encode_target(pd_target, current_q):
    """Wrist target relative to *current* q; finger PD targets are absolute."""
    if pd_target.shape != current_q.shape or pd_target.shape[-1] != 18:
        raise ValueError('matching 18-D PD target and measured q required')
    value = pd_target[..., list(ACTIVE)].copy()
    value[..., :6] -= current_q[..., :6]
    return value.astype(np.float32)


def decode_target(encoded, current_q):
    if encoded.shape != (*current_q.shape[:-1], 12) or current_q.shape[-1] != 18:
        raise ValueError('12 independent PD coordinates and 18 measured joints required')
    value = torch.zeros_like(current_q)
    value[..., list(ACTIVE)] = encoded
    value[..., :6] = value[..., :6] + current_q[..., :6]
    return couple(value)


def command(target, current_q, offset, scale):
    """Invert native targets, then report clipping; do not add feedforward twice."""
    if target.shape != current_q.shape or current_q.shape[-1] != 18:
        raise ValueError('matching live q and PD target required')
    if offset.shape != (18,) or scale.shape != (18,) or (scale <= 0).any():
        raise ValueError('positive 18-D native scale required')
    normalized = (target - offset) / scale
    normalized[..., :6] = (target[..., :6] - current_q[..., :6] - offset[:6]) / scale[:6]
    normalized[..., 6:] = normalized[..., 6:] * 2 - 1
    mask = normalized.new_zeros(18)
    mask[list(ACTIVE)] = 1
    intended = normalized * mask
    applied = intended.clamp(-1, 1)
    if not torch.isfinite(intended).all():
        raise FloatingPointError('nonfinite PD conversion')
    return intended, applied, (intended - applied).abs().amax(-1) > 1e-6


def pd_from_action(action, current_q, offset, scale):
    """Pure independent reconstruction of Inspire's in-place native conversion."""
    action = action.clone()
    action[..., 6:] = (1 + action[..., 6:]) / 2
    target = offset + scale * action
    target[..., :6] = target[..., :6] + current_q[..., :6]
    return couple(target)


def windows(data, envs):
    """Whole-episode windows. PD[t] causes measured hand[t+1], never PD[t+1]."""
    controls = len(data['pd_targets'])
    if any(len(data[k]) != controls + 1 for k in ('q', 'dq', 'hand', 'obj', 'velocity')):
        raise ValueError('terminal measured state and all control labels required')
    indices = np.arange(controls + 1 - 24)
    future = data['hand'][indices[:, None] + np.arange(1, 25)][:, :, envs]
    future = np.moveaxis(future, 2, 1)
    state, tau = features(*(data[k][indices][:, envs] for k in
                            ('q', 'dq', 'hand', 'obj', 'velocity')), future)
    target = encode_target(data['pd_targets'][indices][:, envs], data['q'][indices][:, envs])
    return (state.reshape(-1, STATE_DIM), tau.reshape(-1, 24, 33),
            target.reshape(-1, 12), np.repeat(indices, len(envs)))


class PDRetargeter(nn.Module):
    def __init__(self, width=256, use_tau=True):
        super().__init__()
        self.width, self.use_tau = width, use_tau
        for name, shape in (('state', (STATE_DIM,)), ('tau', (24, 33)), ('target', (12,))):
            self.register_buffer(name + '_mean', torch.zeros(shape))
            self.register_buffer(name + '_scale', torch.ones(shape))
        self.net = nn.Sequential(nn.Linear(STATE_DIM + 24 * 33, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 12))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state, tau):
        if state.shape != (len(state), STATE_DIM) or tau.shape != (len(state), 24, 33):
            raise ValueError('measured state / hand trajectory shape mismatch')
        local = (tau - self.tau_mean) / self.tau_scale
        if not self.use_tau:
            local = torch.zeros_like(local)
        value = torch.cat(((state - self.state_mean) / self.state_scale, local.flatten(1)), -1)
        return self.net(value) * self.target_scale + self.target_mean


def load_model(packet, device):
    if packet.get('schema') != SCHEMA:
        raise ValueError('actual PD-target model schema required')
    model = PDRetargeter(packet['width'], packet['use_tau']).to(device)
    model.load_state_dict(packet['model'], strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('nonfinite model')
    if any((getattr(model, name + '_scale') <= 0).any() for name in ('state', 'tau', 'target')):
        raise ValueError('nonpositive normalization')
    return model.eval()
