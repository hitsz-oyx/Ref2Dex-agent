"""Compact one-step object-effect model for low-overhead RL reward inference."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


SCHEMA = "ref2dex.cmlite.v1"
INPUT_DIM = 49


def _quat_rotate_inverse_xyzw(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz, w = quat[..., :3], quat[..., 3:4]
    return vector * (2 * w.square() - 1) - 2 * w * torch.cross(xyz, vector, dim=-1) + 2 * xyz * (
        xyz * vector).sum(-1, keepdim=True)


def _quat_rotate_xyzw(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz, w = quat[..., :3], quat[..., 3:4]
    return vector * (2 * w.square() - 1) + 2 * w * torch.cross(xyz, vector, dim=-1) + 2 * xyz * (
        xyz * vector).sum(-1, keepdim=True)


def compact_features(q: torch.Tensor, action: torch.Tensor,
                     object_state: torch.Tensor) -> torch.Tensor:
    if (q.ndim != 2 or q.shape != action.shape or q.shape[1] != 18 or
            object_state.shape != (q.shape[0], 13)):
        raise ValueError("CmLite expects q/action [B,18] and object_state [B,13]")
    if not (torch.isfinite(q).all() and torch.isfinite(action).all() and
            torch.isfinite(object_state).all()):
        raise FloatingPointError("CmLite inputs must be finite")
    quat = F.normalize(object_state[:, 3:7], dim=-1, eps=1e-8)
    quat = torch.where(quat[:, 3:4] < 0, -quat, quat)
    return torch.cat((q, action, object_state[:, :3], quat, object_state[:, 7:]), dim=-1)


def local_translation_target(object_state: torch.Tensor,
                             next_object_state: torch.Tensor) -> torch.Tensor:
    if object_state.shape != next_object_state.shape or object_state.shape[-1] != 13:
        raise ValueError("object states must match [B,13]")
    quat = F.normalize(object_state[:, 3:7], dim=-1, eps=1e-8)
    return _quat_rotate_inverse_xyzw(quat, next_object_state[:, :3] - object_state[:, :3])


def local_to_world_translation(object_state: torch.Tensor,
                               local_translation: torch.Tensor) -> torch.Tensor:
    if object_state.ndim != 2 or object_state.shape[1] != 13 or local_translation.shape != (
            object_state.shape[0], 3):
        raise ValueError("invalid CmLite translation conversion")
    quat = F.normalize(object_state[:, 3:7], dim=-1, eps=1e-8)
    return _quat_rotate_xyzw(quat, local_translation)


@torch.inference_mode()
def goal_reward(current_position: torch.Tensor, goal_position: torch.Tensor,
                predicted_delta: torch.Tensor, contact_probability: torch.Tensor,
                *, sigma_m: float = 0.02, positive_only: bool = True) -> torch.Tensor:
    if (current_position.shape != goal_position.shape or
            current_position.shape != predicted_delta.shape or
            current_position.ndim != 2 or current_position.shape[1] != 3 or
            contact_probability.shape != current_position.shape[:1] or sigma_m <= 0):
        raise ValueError("invalid CmLite reward inputs")
    zero_cost = (current_position - goal_position).square().sum(-1) / sigma_m ** 2
    predicted_cost = (current_position + predicted_delta - goal_position).square().sum(-1) / sigma_m ** 2
    reward = torch.tanh(zero_cost - predicted_cost) * contact_probability
    return reward.clamp_min(0) if positive_only else reward


class _ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.layers = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.layers(self.norm(value))


class CmLite(nn.Module):
    def __init__(self, width: int = 128, blocks: int = 3) -> None:
        super().__init__()
        if width < 16 or blocks < 1:
            raise ValueError("CmLite width/blocks are too small")
        self.width, self.blocks = int(width), int(blocks)
        self.input = nn.Sequential(nn.Linear(INPUT_DIM, width), nn.SiLU())
        self.backbone = nn.Sequential(*[_ResidualBlock(width) for _ in range(blocks)])
        self.output_norm = nn.LayerNorm(width)
        self.translation = nn.Linear(width, 3)
        self.contact = nn.Linear(width, 1)

    def forward(self, normalized_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if normalized_features.ndim != 2 or normalized_features.shape[1] != INPUT_DIM:
            raise ValueError(f"normalized_features must be [B,{INPUT_DIM}]")
        hidden = self.output_norm(self.backbone(self.input(normalized_features)))
        return {
            "delta_local_normalized": self.translation(hidden),
            "contact_logit": self.contact(hidden).squeeze(-1),
        }


class FrozenCmLite:
    def __init__(self, checkpoint: str, device: str | torch.device,
                 expected_sha256: str | None = None) -> None:
        if expected_sha256 is not None:
            actual = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
            if actual != expected_sha256:
                raise ValueError(f"CmLite checkpoint SHA256 mismatch: {actual}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("schema") != SCHEMA:
            raise ValueError("CmLite checkpoint schema mismatch")
        config = payload["model_config"]
        self.model = CmLite(**config).to(device).eval()
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.requires_grad_(False)
        self.feature_mean = payload["feature_mean"].to(device)
        self.feature_std = payload["feature_std"].to(device)
        self.target_mean = payload["target_mean"].to(device)
        self.target_std = payload["target_std"].to(device)

    @torch.inference_mode()
    def predict(self, q: torch.Tensor, action: torch.Tensor,
                object_state: torch.Tensor) -> dict[str, torch.Tensor]:
        features = compact_features(q, action, object_state)
        output = self.model((features - self.feature_mean) / self.feature_std)
        local = output["delta_local_normalized"] * self.target_std + self.target_mean
        return {
            "delta_local": local,
            "delta_world": local_to_world_translation(object_state, local),
            "contact_probability": output["contact_logit"].sigmoid(),
        }
