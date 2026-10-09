"""Small feed-forward H -> current-object-frame hand trajectory predictor."""

import torch
from torch import nn


SCHEMA = "ref2dex.history-to-tau.v1"
HISTORY_DIM = 1442
CURRENT_HAND_DIM = 33
HORIZON = 24
TAU_DIM = HORIZON * CURRENT_HAND_DIM


class HistoryToTau(nn.Module):
    """Bounded Probe model; one deterministic future trajectory per H query."""

    def __init__(self, width=256):
        super().__init__()
        self.width = int(width)
        self.net = nn.Sequential(
            nn.Linear(HISTORY_DIM + CURRENT_HAND_DIM, self.width), nn.SiLU(),
            nn.Linear(self.width, self.width), nn.SiLU(),
            nn.Linear(self.width, TAU_DIM))

    def forward(self, history, current_hand):
        if (history.ndim != 2 or history.shape[1] != HISTORY_DIM
                or current_hand.shape != (len(history), 11, 3)):
            raise ValueError("history/current hand shape mismatch")
        return self.net(torch.cat((history, current_hand.flatten(1)), dim=1)).reshape(
            len(history), HORIZON, 11, 3)
