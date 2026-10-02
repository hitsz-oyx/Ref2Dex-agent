"""Frozen task-value representation built from direct-Q and Cm consequences."""
from __future__ import annotations

import torch
from torch import nn


class CmTaskResidualMLP(nn.Module):
    """Predict the realized-return residual over a frozen direct-Q."""

    def __init__(self, input_dim: int = 60):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def load_frozen_cm_residual(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.cm_task_residual_mlp.v1":
        raise ValueError("task representation schema mismatch")
    model = CmTaskResidualMLP(len(payload["input_mean"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    input_mean = payload["input_mean"].to(device)
    input_scale = payload["input_scale"].to(device).clamp_min(1e-6)
    target_mean = payload["target_mean"].to(device)
    target_scale = payload["target_scale"].to(device).clamp_min(1e-6)
    return model, input_mean, input_scale, target_mean, target_scale


def predict_cm_residual(model, input_mean, input_scale, target_mean, target_scale, features):
    normalized = (features - input_mean) / input_scale
    return model(normalized) * target_scale + target_mean
