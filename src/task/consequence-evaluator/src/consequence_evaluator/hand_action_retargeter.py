"""Full native-action retargeter for the ref7_2 execution route."""

import numpy as np
import torch
from torch import nn


SCHEMA = "ref2dex.hand-action-retargeter.v1"
CONTEXT_SCHEMA = "ref2dex.hand-action-retargeter.v2"
CONTACT_CONTEXT_SCHEMA = "ref2dex.hand-action-retargeter.v3"
HORIZON = 24
ACTION_DIM = 18
CONTEXT_DIM = 11 * 3 + ACTION_DIM
CONTACT_CONTEXT_DIM = 10
CONTACT_CONTEXT_FIELDS = (
    "pair", "surface_gap", "support_gap", "object_velocity_linear_x",
    "object_velocity_linear_y", "object_velocity_linear_z",
    "object_velocity_angular_x", "object_velocity_angular_y",
    "object_velocity_angular_z", "table_footprint")


def contact_context_features(pair, surface_gap, support_gap, object_velocity,
                             table_footprint):
    """Pack query-time contact proxies without exposing a future label.

    The returned order is fixed by ``CONTACT_CONTEXT_FIELDS``. Callers must
    pass fields measured at query state ``t``; the window loader skips the
    reset frame because its contact buffer is not a physical measurement.
    """
    pair = np.asarray(pair)
    surface_gap = np.asarray(surface_gap)
    support_gap = np.asarray(support_gap)
    object_velocity = np.asarray(object_velocity)
    table_footprint = np.asarray(table_footprint)
    prefix = pair.shape
    if pair.dtype != np.bool_ or table_footprint.dtype != np.bool_:
        raise ValueError("pair and table footprint must be boolean query-state fields")
    if (surface_gap.shape != prefix or support_gap.shape != prefix
            or table_footprint.shape != prefix
            or object_velocity.shape != prefix + (6,)):
        raise ValueError("contact proxy batch shapes must share a 6-D velocity suffix")
    values = np.concatenate((
        pair.astype("float32")[..., None],
        surface_gap.astype("float32")[..., None],
        support_gap.astype("float32")[..., None],
        object_velocity.astype("float32"),
        table_footprint.astype("float32")[..., None]), axis=-1)
    if values.shape[-1] != CONTACT_CONTEXT_DIM or not np.isfinite(values).all():
        raise ValueError("contact proxy values must be finite 10-D features")
    return values.astype("float32")


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
    """Full-action retargeter with optional query-time contact context.

    ``contact_dim=0`` is the frozen v2 architecture. A positive contact
    branch is initialized to zero so it can be matched to a v2 checkpoint
    without changing the shared hand/state/context computation.
    """

    def __init__(self, width=128, contact_dim=0):
        super().__init__()
        self.width = int(width)
        self.contact_dim = int(contact_dim)
        if self.contact_dim < 0:
            raise ValueError("contact_dim must be nonnegative")
        self.hand = nn.Sequential(nn.Linear(33, width), nn.SiLU(), nn.Linear(width, width))
        self.state = nn.Sequential(nn.Linear(36 + CONTEXT_DIM, width), nn.SiLU(), nn.Linear(width, width))
        self.contact = nn.Linear(self.contact_dim, width) if self.contact_dim else None
        if self.contact is not None:
            nn.init.zeros_(self.contact.weight)
            nn.init.zeros_(self.contact.bias)
        self.time = nn.Parameter(torch.randn(1, HORIZON, width) * .02)
        self.query = nn.Parameter(torch.randn(1, HORIZON, width) * .02)
        layer = nn.TransformerDecoderLayer(width, 4, width * 4, dropout=0., batch_first=True,
                                           norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, 2)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, ACTION_DIM))

    def forward(self, hand, state, context, contact=None):
        if hand.ndim != 4 or tuple(hand.shape[1:]) != (HORIZON, 11, 3):
            raise ValueError("hand input must be [N,24,11,3]")
        if (state.ndim != 2 or tuple(state.shape[1:]) != (36,) or len(state) != len(hand)
                or context.ndim != 2 or tuple(context.shape[1:]) != (CONTEXT_DIM,)
                or len(context) != len(hand)):
            raise ValueError("context retarget input shape mismatch")
        if self.contact_dim:
            if (contact is None or contact.ndim != 2
                    or tuple(contact.shape[1:]) != (self.contact_dim,)
                    or len(contact) != len(hand)):
                raise ValueError("contact context input shape mismatch")
        elif contact is not None:
            raise ValueError("contact input requires a positive contact_dim")
        state_context = torch.cat((state, context), -1)
        state_encoded = self.state(state_context)
        if self.contact is not None:
            state_encoded = state_encoded + self.contact(contact)
        encoded = torch.cat((self.hand(hand.flatten(2)) + self.time,
                             state_encoded[:, None]), dim=1)
        decoded = self.decoder((self.query + self.time).expand(len(hand), -1, -1), encoded)
        return self.output(decoded)
