"""Trajectory-conditioned consequence evaluator for the complete-chain route.

The candidate branch is deliberately separated from the object-effect branch:
``H + tau`` is C0, ``H + tau + E_GT`` is C1, and ``H + tau + E_PW`` is C2.
Here ``tau`` is a 24-step hand trajectory in the *current object frame* and
``E`` is a 24-step object effect (rotation and translation) in that same
query frame.  This avoids feeding the old native-action tensor as a proxy for
the trajectory interface.
"""

import torch
from torch import nn


SCHEMA = "ref2dex.consequence-trajectory-utility.v1"
HORIZON = 24
TRAJECTORY_DIM = 3 * 11
OBJECT_EFFECT_DIM = 12


class TrajectoryUtility(nn.Module):
    """Equal-capacity scalar evaluator with separate trajectory/effect arms."""

    def __init__(self, history_dim=1442, width=128, layers=2):
        super().__init__()
        self.history = nn.Linear(history_dim, width)
        self.trajectory = nn.Linear(TRAJECTORY_DIM, width)
        self.effect = nn.Linear(OBJECT_EFFECT_DIM, width)
        self.time = nn.Parameter(torch.zeros(1, HORIZON + 1, width))
        block = nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0.,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.score = nn.Linear(width, 1)

    def forward(self, history, trajectory, effect, use_effect=True):
        if (history.ndim != 2 or trajectory.ndim != 3 or effect.ndim != 3
                or trajectory.shape[1:] != (HORIZON, TRAJECTORY_DIM)
                or effect.shape[1:] != (HORIZON, OBJECT_EFFECT_DIM)
                or len(history) != len(trajectory) or len(effect) != len(trajectory)):
            raise ValueError("history/trajectory/effect shape mismatch")
        if use_effect not in (False, True):
            raise ValueError("use_effect must be boolean")
        if not use_effect:
            effect = torch.zeros_like(effect)
        tokens = torch.cat((self.history(history)[:, None],
                            self.trajectory(trajectory) + self.effect(effect)), dim=1)
        encoded = self.encoder(tokens + self.time)
        return self.score(encoded[:, 1:].mean(1)).squeeze(-1)
