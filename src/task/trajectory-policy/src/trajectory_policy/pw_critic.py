"""Action-conditioned physical critic contracts for the ref3 bounded RL pilot."""
import numpy as np
import torch
from torch import nn

SCHEMA = 'ref2dex.trajectory-pw-critic.v1'
PW_SHA = 'f70573c4d06d368a7d8ab921ff03c936bb0e1b8565df9363b0f1faeb9ab6ba13'
ORDER = [0, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10]


def local_inputs(data, tick, env, tau):
    """Past/current measurements plus a proposed displacement plan, never future data."""
    past = np.maximum(tick + np.arange(-3, 1), 0)
    current = data['obj'][tick, env]
    rotation, center = current[:3, :3], current[:3, 3]
    obj = np.linalg.inv(current) @ data['obj'][past, env]
    hand = (data['hand'][past, env] - center) @ rotation
    future = hand[-1][None] + np.asarray(tau).reshape(24, 11, 3)
    if not np.allclose(obj[-1], np.eye(4), atol=1e-5):
        raise ValueError('fixed current-object frame required')
    return obj.astype(np.float32), hand.astype(np.float32), future.astype(np.float32)


def rewards(data, reference):
    """Physical next-state labels; the reference is an authorized exogenous task plan."""
    obj = data['obj'][1:]
    pos = np.linalg.norm(obj[..., :3, 3] - reference['obj'][1:, None, :3, 3], axis=-1)
    relative = reference['obj'][1:, None, :3, :3].swapaxes(-1, -2) @ obj[..., :3, :3]
    angle = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    gap = data['surface_gap'][1:]
    supported = data['table_footprint'][1:] & (np.abs(data['support_gap'][1:]) <= .02)
    held = (gap <= .01) & (obj[..., 2, 3] - data['obj'][0, :, 2, 3] >= .03) & ~supported
    previous = np.concatenate((np.zeros_like(data['action'][:1]), data['action'][:-1]))
    rate = np.square(data['action'] - previous).mean(-1)
    return (np.exp(-pos / .05) + .2 * np.exp(-angle / .5) + .5 * held
            + .2 * np.exp(-gap / .02) - .1 * data['clipped'] - .01 * rate).astype(np.float32)


def truncated_returns(reward, ticks, envs, horizon=24, gamma=.99):
    ticks, envs = np.asarray(ticks), np.asarray(envs)
    if (ticks < 0).any() or (ticks + horizon > len(reward)).any():
        raise ValueError('complete future return window required; no terminal clamping')
    values = reward[ticks[:, None] + np.arange(horizon), envs[:, None]]
    return (values * gamma ** np.arange(horizon)).sum(-1).astype(np.float32)


def critic_input(high, tau, physical, use_pw):
    physical = np.asarray(physical, np.float32).reshape(len(high), 288)
    return np.concatenate((high, np.asarray(tau).reshape(len(high), 792),
                           physical if use_pw else np.zeros_like(physical)), -1).astype(np.float32)


class PhysicalQ(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(1816, 256), nn.Tanh(), nn.Linear(256, 256),
                                 nn.Tanh(), nn.Linear(256, 1))

    def forward(self, inputs):
        return self.net((inputs - self.mean) / self.scale).squeeze(-1)


def advantage_weights(advantage, train):
    temperature = max(float(np.std(advantage[train])), .001)
    weights = np.exp(np.clip(advantage / temperature, np.log(.5), np.log(2.)))
    weights /= weights[train].mean()
    return weights.astype(np.float32), temperature
