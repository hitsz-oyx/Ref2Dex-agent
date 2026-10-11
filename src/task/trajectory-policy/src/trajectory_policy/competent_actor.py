"""Measured H / known object goal -> actual future hand; no teacher at inference."""
import numpy as np
import torch
from torch import nn

from .history import INPUT_DIM, measured_history
from .pd_retargeter import windows

SCHEMA = 'ref2dex.competent-trajectory-actor.v1'


def goal_features(current, goal):
    rotation = current[..., :3, :3]
    position = np.einsum('nj,njk->nk', goal[..., :3, 3] - current[..., :3, 3], rotation)
    relative = np.matmul(rotation.swapaxes(-1, -2), goal[..., :3, :3])
    return np.concatenate((position, relative.reshape(-1, 9)), -1).astype(np.float32)


def actor_input(obj, hand, q, dq, velocity, goal):
    return np.concatenate((measured_history(obj, hand, q, dq, velocity),
                           goal_features(obj[-1], goal)), -1)


def actor_windows(data, envs):
    state, tau, target, ticks = windows(data, envs)
    controls = len(data['pd_targets'])
    inputs = []
    for tick in range(controls + 1 - 24):
        past = np.maximum(tick + np.arange(-3, 1), 0)
        inputs.append(actor_input(*(data[key][past][:, envs] for key in
                      ('obj', 'hand', 'q', 'dq', 'velocity')), data['goal'][envs]))
    return np.concatenate(inputs), state, tau, target, ticks


def world_plan(tau, current_hand, rotation):
    return current_hand[:, None] + np.einsum('ntpj,nkj->ntpk',
        np.asarray(tau).reshape(-1, 24, 11, 3), rotation)


def shifted_plan(plan, slot):
    if not 0 <= slot < 8:
        raise ValueError('eight-control actor period required')
    return plan[:, np.minimum(np.arange(24) + slot, 23)]


class CompetentActor(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.width = width
        for name, shape in (('input', (INPUT_DIM + 12,)), ('tau', (24, 33))):
            self.register_buffer(name + '_mean', torch.zeros(shape))
            self.register_buffer(name + '_scale', torch.ones(shape))
        self.net = nn.Sequential(nn.Linear(INPUT_DIM + 12, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 24 * 33))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, inputs):
        if inputs.shape != (len(inputs), INPUT_DIM + 12):
            raise ValueError('H328 + relative goal12 required')
        return self.net((inputs - self.input_mean) / self.input_scale).reshape(-1, 24, 33) * self.tau_scale + self.tau_mean


def load_actor(packet, device):
    if packet.get('schema') != SCHEMA:
        raise ValueError('competent actor checkpoint required')
    actor = CompetentActor(packet['width']).to(device)
    actor.load_state_dict(packet['model'], strict=True)
    if not all(torch.isfinite(v).all() for v in actor.state_dict().values()):
        raise ValueError('nonfinite actor checkpoint')
    if (actor.input_scale <= 0).any() or (actor.tau_scale <= 0).any():
        raise ValueError('nonpositive actor scale')
    return actor.eval()
