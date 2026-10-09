"""Full native-action retargeter for the ref7_2 execution route."""

import numpy as np
import torch
from torch import nn


SCHEMA = "ref2dex.hand-action-retargeter.v1"
CONTEXT_SCHEMA = "ref2dex.hand-action-retargeter.v2"
HORIZON = 24
ACTION_DIM = 18
CONTEXT_DIM = 11 * 3 + ACTION_DIM


def trajectory_input(current_hand, future_hand):
    current_hand = np.asarray(current_hand)
    future_hand = np.asarray(future_hand)
    if current_hand.shape[-2:] != (11, 3) or future_hand.shape[-2:] != (11, 3):
        raise ValueError("11-point hand geometry required")
    if future_hand.shape[-3] != HORIZON or future_hand.shape[:-3] != current_hand.shape[:-2]:
        raise ValueError("future hand horizon/current batch mismatch")
    return (future_hand - current_hand[..., None, :, :]).astype("float32")


def chunk_offset(tick, query_tick, horizon=HORIZON):
    """Return the action offset relative to the chunk's query tick."""
    offset = int(tick) - int(query_tick)
    if offset < 0 or offset >= int(horizon):
        raise ValueError("tick is outside the active action chunk")
    return offset


def hand_object_context(current_hand, object_pose, previous_action):
    """Encode current hand/object geometry and the previous native command.

    The object frame is query-time only; no future object state or contact label
    is exposed.  This context is deliberately controller-local: the previous
    command carries the preload/history that q and dq alone do not identify.
    """
    current_hand = np.asarray(current_hand)
    object_pose = np.asarray(object_pose)
    previous_action = np.asarray(previous_action)
    if (current_hand.shape[-2:] != (11, 3) or object_pose.shape[-2:] != (4, 4)
            or previous_action.shape[-1:] != (ACTION_DIM,)
            or current_hand.shape[:-2] != object_pose.shape[:-2]
            or current_hand.shape[:-2] != previous_action.shape[:-1]):
        raise ValueError("current hand/object/previous action batch mismatch")
    relative = (current_hand - object_pose[..., None, :3, 3]) @ object_pose[..., :3, :3]
    return np.concatenate((relative.reshape(*relative.shape[:-2], 33), previous_action), -1).astype("float32")


class Standardizer:
    """Train-only per-coordinate statistics, including horizon coordinates."""

    def __init__(self, mean, scale):
        self.mean = np.asarray(mean, dtype="float32")
        self.scale = np.asarray(scale, dtype="float32")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("invalid standardizer")

    @classmethod
    def fit(cls, value, floor=1e-4):
        value = np.asarray(value)
        return cls(value.mean(axis=0, dtype=np.float64),
                   np.maximum(value.std(axis=0, dtype=np.float64), floor))

    def encode(self, value):
        return ((np.asarray(value) - self.mean) / self.scale).astype("float32")

    def decode(self, value):
        return (np.asarray(value) * self.scale + self.mean).astype("float32")

    def as_dict(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


class HandActionRetargeter(nn.Module):
    """Predict the full commanded native action for a 24-step hand future."""

    def __init__(self, width=128):
        super().__init__()
        self.width = int(width)
        self.hand = nn.Sequential(nn.Linear(33, width), nn.SiLU(), nn.Linear(width, width))
        self.state = nn.Sequential(nn.Linear(36, width), nn.SiLU(), nn.Linear(width, width))
        self.time = nn.Parameter(torch.randn(1, HORIZON, width) * .02)
        self.query = nn.Parameter(torch.randn(1, HORIZON, width) * .02)
        layer = nn.TransformerDecoderLayer(width, 4, width * 4, dropout=0., batch_first=True,
                                           norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, 2)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, ACTION_DIM))

    def forward(self, hand, state):
        if hand.ndim != 4 or tuple(hand.shape[1:]) != (HORIZON, 11, 3):
            raise ValueError("hand input must be [N,24,11,3]")
        if state.ndim != 2 or tuple(state.shape[1:]) != (36,) or len(state) != len(hand):
            raise ValueError("state input must be [N,36]")
        context = torch.cat((self.hand(hand.flatten(2)) + self.time,
                             self.state(state)[:, None]), dim=1)
        decoded = self.decoder((self.query + self.time).expand(len(hand), -1, -1), context)
        return self.output(decoded)


class ContextHandActionRetargeter(nn.Module):
    """Full-action retargeter with controller-history/contact geometry context."""

    def __init__(self, width=128):
        super().__init__()
        self.width = int(width)
        self.hand = nn.Sequential(nn.Linear(33, width), nn.SiLU(), nn.Linear(width, width))
        self.state = nn.Sequential(nn.Linear(36 + CONTEXT_DIM, width), nn.SiLU(), nn.Linear(width, width))
        self.time = nn.Parameter(torch.randn(1, HORIZON, width) * .02)
        self.query = nn.Parameter(torch.randn(1, HORIZON, width) * .02)
        layer = nn.TransformerDecoderLayer(width, 4, width * 4, dropout=0., batch_first=True,
                                           norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, 2)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, ACTION_DIM))

    def forward(self, hand, state, context):
        if hand.ndim != 4 or tuple(hand.shape[1:]) != (HORIZON, 11, 3):
            raise ValueError("hand input must be [N,24,11,3]")
        if (state.ndim != 2 or tuple(state.shape[1:]) != (36,) or len(state) != len(hand)
                or context.ndim != 2 or tuple(context.shape[1:]) != (CONTEXT_DIM,)
                or len(context) != len(hand)):
            raise ValueError("context retarget input shape mismatch")
        state_context = torch.cat((state, context), -1)
        encoded = torch.cat((self.hand(hand.flatten(2)) + self.time,
                             self.state(state_context)[:, None]), dim=1)
        decoded = self.decoder((self.query + self.time).expand(len(hand), -1, -1), encoded)
        return self.output(decoded)
