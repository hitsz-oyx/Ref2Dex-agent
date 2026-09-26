"""Compact action-conditioned Cm with short-horizon grasp-stability heads."""
from __future__ import annotations

import torch
from torch import nn

from src.task.CmResidual.tools.probe_local_geometric_cm import (
    FEATURE_DIM, NUM_REGIONS, SixRegionCm,
)


class ContactAwareCm(SixRegionCm):
    """Predict one-step effect, five-step effect and mean five-step contact."""

    def __init__(self, width=32):
        super().__init__(width=width)
        self.followup_translation = nn.Linear(width, 3)
        nn.init.zeros_(self.followup_translation.weight)
        nn.init.zeros_(self.followup_translation.bias)

    def forward(self, region_features, object_velocity):
        if region_features.ndim != 3 or region_features.shape[1:] != (NUM_REGIONS, FEATURE_DIM):
            raise ValueError("region_features must be [B,6,19]")
        if object_velocity.shape != (len(region_features), 6):
            raise ValueError("object_velocity must be [B,6]")
        encoded = self.input(region_features) + self.region_embedding[None]
        attended = self.attention(encoded, encoded, encoded, need_weights=False)[0]
        tokens = self.norm(encoded + attended)
        pooled = torch.cat((tokens.mean(dim=1), tokens.amax(dim=1), object_velocity), dim=-1)
        hidden = self.head(pooled)
        return {"cm_tokens": tokens,
                "delta_local": self.translation(hidden) * .01,
                "followup_delta_local": self.followup_translation(hidden) * .02,
                "contact_fraction": torch.sigmoid(self.contact(hidden).squeeze(-1))}


class RawContactAwareCm(nn.Module):
    """Small nongeometric state+action capacity control, same three targets."""

    def __init__(self, mean, std, width=64):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp_min(.01))
        self.net = nn.Sequential(nn.Linear(len(mean), width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU())
        self.step = nn.Linear(width, 3)
        self.followup = nn.Linear(width, 3)
        self.contact = nn.Linear(width, 1)
        for head in (self.step, self.followup):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, raw):
        hidden = self.net((raw - self.mean) / self.std)
        return {"delta_local": self.step(hidden) * .01,
                "followup_delta_local": self.followup(hidden) * .02,
                "contact_fraction": torch.sigmoid(self.contact(hidden).squeeze(-1))}
