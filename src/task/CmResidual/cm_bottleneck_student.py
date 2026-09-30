"""Matched observation-driven students with a four-dimensional inference input."""
from __future__ import annotations

import torch
from torch import nn


class BottleneckStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1446, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 18))

    def forward(self, values):
        return self.net(values)


class PhysicalFeatures(nn.Module):
    """Frozen physical predictions; candidate is the pinned source expert."""
    def __init__(self, checkpoint, arm):
        super().__init__()
        if arm not in ("on", "off", "random", "action"):
            raise ValueError(arm)
        self.arm = arm
        self.net = nn.Sequential(nn.Linear(checkpoint["mean"].numel(), 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 6))
        if arm == "random":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(9278)
                for layer in self.net:
                    if isinstance(layer, nn.Linear):
                        layer.reset_parameters()
        else:
            self.net.load_state_dict(checkpoint["state_dict"], strict=True)
        for name in ("mean", "std", "delta_mean", "delta_std"):
            self.register_buffer(name, checkpoint[name].clone())
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, obs, source_action):
        ids = torch.zeros(obs.shape[0], 6, device=obs.device)
        ids[:, 4] = 1
        x = torch.cat([obs, source_action, ids], dim=1)
        prediction = self.net((x - self.mean) / self.std)
        delta = prediction[:, :3] * self.delta_std + self.delta_mean
        contact = prediction[:, 3:].sort(dim=1).values[:, 1:2].clamp(0, 1)
        features = torch.cat([delta, contact], dim=1)
        if self.arm == "off":
            features = torch.zeros_like(features)
        elif self.arm == "action":
            features = source_action[:, :4]
        if not torch.isfinite(features).all():
            raise FloatingPointError("nonfinite bottleneck input")
        return features
