"""History-conditioned Cm credit for a frozen simulator transition model.

The checkpoint consumed here is trained by ``probe_history_value_cm.py``.  It
predicts several future outcomes from a short sequence ending in the current
action.  Online reward shaping uses only the predicted contact-supported lift
head; it does not use a hard predicted-contact gate.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn


SCHEMA = "ref2dex.history_value_cm.v1"
OBS_DIM = 18 + 18 + 13
ACTION_DIM = 18
STEP_DIM = OBS_DIM + ACTION_DIM
SUPPORTED_LIFT_INDEX = 3


class HistoryValueCm(nn.Module):
    """Architecture used by the offline history-value Cm probe."""

    def __init__(self, history_len: int, width: int = 96, continuous_count: int = 4):
        super().__init__()
        if not 1 <= history_len <= 10 or width < 16:
            raise ValueError("invalid history model dimensions")
        self.history_len = int(history_len)
        input_dim = self.history_len * (STEP_DIM + 1)
        self.net = nn.Sequential(
            nn.Linear(input_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.continuous_mean = nn.Linear(width, continuous_count)
        self.continuous_logvar = nn.Linear(width, continuous_count)
        self.final_contact_logit = nn.Linear(width, 1)

    def forward(self, history: torch.Tensor,
                history_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if (history.ndim != 3 or history.shape[1] != self.history_len or
                history.shape[2] != STEP_DIM):
            raise ValueError("history has an unexpected shape")
        if history_mask.shape != history.shape[:2]:
            raise ValueError("history mask has an unexpected shape")
        inputs = torch.cat((history, history_mask.unsqueeze(-1)), dim=-1).flatten(1)
        hidden = self.net(inputs)
        return {
            "continuous_mean": self.continuous_mean(hidden),
            "continuous_logvar": self.continuous_logvar(hidden).clamp(-5.0, 4.0),
            "final_contact_logit": self.final_contact_logit(hidden).squeeze(-1),
        }


def temporal_step_features(q: torch.Tensor, dof_vel: torch.Tensor,
                           object_state: torch.Tensor,
                           action: torch.Tensor) -> torch.Tensor:
    """Build the exact 67D feature block used by the offline probe."""
    shape = q.shape
    if (q.ndim != 2 or shape[1] != 18 or dof_vel.shape != shape or
            action.shape != shape or object_state.shape != (shape[0], 13)):
        raise ValueError("temporal Cm expects q/velocity/action [B,18] and object [B,13]")
    values = (q, dof_vel, object_state, action)
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError("temporal Cm inputs must be finite")
    relative_q = torch.cat((q[:, :3] - object_state[:, :3], q[:, 3:]), dim=-1)
    return torch.cat((relative_q, dof_vel, object_state, action), dim=-1)


class TemporalHistoryBuffer:
    """Per-environment history with the offline probe's explicit padding mask."""

    def __init__(self, num_envs: int, history_len: int, device: torch.device | str):
        if num_envs < 1 or not 1 <= history_len <= 10:
            raise ValueError("invalid temporal history buffer dimensions")
        self.history = torch.zeros(num_envs, history_len, STEP_DIM, device=device)
        self.mask = torch.zeros(num_envs, history_len, device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            ids = torch.arange(len(self.initialized), device=self.history.device)
        elif isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(device=self.history.device, dtype=torch.long).reshape(-1)
        else:
            ids = torch.as_tensor(env_ids, device=self.history.device, dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            return
        if (ids < 0).any() or (ids >= len(self.initialized)).any():
            raise IndexError("temporal history reset ids are out of range")
        self.history[ids] = 0
        self.mask[ids] = 0
        self.initialized[ids] = False

    def append(self, step: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if step.ndim != 2 or step.shape != (len(self.history), STEP_DIM):
            raise ValueError("temporal history step has an unexpected shape")
        if not torch.isfinite(step).all():
            raise FloatingPointError("temporal history step must be finite")
        fresh = ~self.initialized
        if fresh.any():
            self.history[fresh] = step[fresh].unsqueeze(1)
            self.mask[fresh] = 0
        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.mask = torch.roll(self.mask, shifts=-1, dims=1)
        self.history[:, -1] = step
        self.mask[:, -1] = 1
        self.initialized[:] = True
        return self.history, self.mask


class FrozenTemporalHistoryCm:
    """Load and run a history-conditioned value model without gradients."""

    def __init__(self, checkpoint: str | Path, device: torch.device | str,
                 expected_sha256: str | None = None):
        path = Path(checkpoint)
        if expected_sha256 is not None:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected_sha256:
                raise ValueError(f"temporal Cm checkpoint SHA256 mismatch: {actual}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != SCHEMA or payload.get("variant") != "history_action":
            raise ValueError("temporal Cm checkpoint schema or variant mismatch")
        state = payload["model"]
        input_dim = int(state["net.0.weight"].shape[1])
        if input_dim % (STEP_DIM + 1):
            raise ValueError("temporal Cm checkpoint input width is invalid")
        history_len = input_dim // (STEP_DIM + 1)
        width = int(state["net.0.weight"].shape[0])
        continuous_count = int(state["continuous_mean.weight"].shape[0])
        self.model = HistoryValueCm(history_len, width, continuous_count).to(device).eval()
        self.model.load_state_dict(state, strict=True)
        self.model.requires_grad_(False)
        self.history_len = history_len
        self.feature_mean = payload["feature_mean"].to(device)
        self.feature_std = payload["feature_std"].to(device).clamp_min(1e-3)
        self.target_mean = payload["target_mean"].to(device)
        self.target_std = payload["target_std"].to(device).clamp_min(1e-3)
        if self.feature_mean.shape != (STEP_DIM,) or self.feature_std.shape != (STEP_DIM,):
            raise ValueError("temporal Cm feature statistics have an invalid shape")
        if self.target_mean.shape[0] <= SUPPORTED_LIFT_INDEX:
            raise ValueError("temporal Cm checkpoint has no supported-lift head")

    @torch.inference_mode()
    def predict(self, history: torch.Tensor, history_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if history.shape[1:] != (self.history_len, STEP_DIM):
            raise ValueError("history does not match temporal Cm checkpoint")
        normalized = (history - self.feature_mean) / self.feature_std
        output = self.model(normalized, history_mask)
        outcomes = output["continuous_mean"] * self.target_std + self.target_mean
        return {
            "outcomes": outcomes,
            "supported_lift_mm": outcomes[:, SUPPORTED_LIFT_INDEX],
            "final_contact_probability": output["final_contact_logit"].sigmoid(),
        }
