"""Sequence-conditioned Cm with explicit first/second action effect paths."""
from __future__ import annotations

import torch
from torch import nn


class StructuredTwoStepCm(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, width: int = 64):
        super().__init__()
        if mean.shape != (67,) or std.shape != (67,):
            raise ValueError("pre-treatment raw state must have 67 features")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp_min(.01))
        self.trunk = nn.Sequential(nn.Linear(67, width), nn.SiLU(),
                                   nn.Linear(width, width), nn.SiLU())
        self.step_base = nn.Linear(width, 3)
        self.step_first = nn.Linear(width, 3)
        self.followup_base = nn.Linear(width, 3)
        self.followup_first = nn.Linear(width, 3)
        self.followup_second = nn.Linear(width, 3)
        self.contact_base = nn.Linear(width, 1)
        self.contact_first = nn.Linear(width, 1)
        self.contact_second = nn.Linear(width, 1)
        for head in (self.step_base, self.step_first,
                     self.followup_base, self.followup_first,
                     self.followup_second, self.contact_first, self.contact_second):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
        if raw.ndim != 2 or raw.shape[1] != 69:
            raise ValueError("sequence Cm expects 67 pre-treatment + two dose features")
        first, second = (raw[:, -2:] / .1).unbind(-1)
        if (first.abs() > 1 + 1e-5).any() or (second.abs() > 1 + 1e-5).any():
            raise ValueError("sequence Cm dose outside randomized support")
        hidden = self.trunk((raw[:, :67] - self.mean) / self.std)
        step = (self.step_base(hidden) +
                first[:, None] * self.step_first(hidden)) * .01
        future = (self.followup_base(hidden) +
                  first[:, None] * self.followup_first(hidden) +
                  second[:, None] * self.followup_second(hidden)) * .02
        contact_logit = (self.contact_base(hidden).squeeze(-1) +
                         first * self.contact_first(hidden).squeeze(-1) +
                         second * self.contact_second(hidden).squeeze(-1))
        return {"delta_local": step, "followup_delta_local": future,
                "contact_fraction": torch.sigmoid(contact_logit)}
