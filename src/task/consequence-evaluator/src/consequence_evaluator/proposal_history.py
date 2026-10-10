"""Measured four-state hand/object history, without actor reference observations."""
import numpy as np
import torch
from torch import nn

SCHEMA = 'ref2dex.measured-history-tau-proposal.v1'
HISTORY_STATES = 4
INPUT_DIM = 300
HORIZON = 24


def condition(object_history, hand_history, q_history, dq_history, velocity_history):
    """Only four already measured states; express geometry in query object frame.

    Finger q/dq complement measured hand geometry. Wrist geometry/velocity is
    represented by the hand history, avoiding world-origin joint translations.
    No actor obs, reference, action, clock, reward or contact field is accepted.
    """
    obj = np.asarray(object_history); hand = np.asarray(hand_history)
    q = np.asarray(q_history); dq = np.asarray(dq_history)
    vel = np.asarray(velocity_history)
    if (obj.shape != (4, 4, 4) or hand.shape != (4, 11, 3)
            or q.shape != (4, 18) or dq.shape != (4, 18) or vel.shape != (4, 6)):
        raise ValueError('exactly four measured states required')
    rotation = obj[-1, :3, :3]; translation = obj[-1, :3, 3]
    local_hand = (hand - translation) @ rotation
    local_obj = np.linalg.inv(obj[-1])[None] @ obj
    relative_fingers = local_hand[:, 1:] - local_hand[:, :1]
    # Root translation and finger shape have separate coordinates; velocities
    # are vectors, so neither angular nor linear velocity receives translation.
    local_velocity = np.concatenate((vel[:, :3] @ rotation, vel[:, 3:] @ rotation), -1)
    value = np.concatenate((local_hand[:, 0].reshape(-1), relative_fingers.reshape(-1),
        local_obj[:, :3].reshape(-1), q[:, 6:].reshape(-1), dq[:, 6:].reshape(-1),
        local_velocity.reshape(-1))).astype('float32')
    if value.shape != (INPUT_DIM,) or not np.isfinite(value).all():
        raise ValueError('finite measured history required')
    return value, local_hand[-1].astype('float32')


class MeasuredHistoryToTau(nn.Module):
    """Identical model for the matched absolute/displacement target comparison."""
    def __init__(self, width=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(INPUT_DIM, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, HORIZON * 33))

    def forward(self, history):
        if history.ndim != 2 or history.shape[1] != INPUT_DIM:
            raise ValueError('measured history shape mismatch')
        return self.net(history).reshape(-1, HORIZON, 11, 3)
