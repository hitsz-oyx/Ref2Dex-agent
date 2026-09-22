"""Low-cost CmLite action proposals for a learned DExplore policy.

The selector deliberately keeps the learned policy as candidate zero.  CmLite
can replace it only when a small, fixed proposal set predicts a better next
object position.  This makes the Cm-on/off comparison local and auditable:
both arms execute the same learned checkpoint and the same five candidates.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.task.CmResidual.cmlite import FrozenCmLite


@dataclass(frozen=True)
class ProposalConfig:
    """Native action perturbations, expressed in DExplore normalized action space."""

    wrist_z_delta: float = 0.01
    finger_delta: float = 0.04
    contact_threshold: float = 0.25
    positive_only: bool = True
    require_actual_contact: bool = True

    def __post_init__(self) -> None:
        if self.wrist_z_delta < 0 or self.finger_delta < 0:
            raise ValueError("proposal deltas must be non-negative")
        if not 0 <= self.contact_threshold <= 1:
            raise ValueError("contact threshold must be in [0,1]")


FINGER_INDICES = (6, 8, 10, 12, 14, 15)


def proposal_actions(base_action: torch.Tensor, config: ProposalConfig = ProposalConfig()) -> torch.Tensor:
    """Return five deterministic candidates with candidate zero equal to policy output."""
    if base_action.ndim != 2 or base_action.shape[1] != 18:
        raise ValueError("base_action must be [B,18]")
    if not torch.isfinite(base_action).all():
        raise FloatingPointError("base_action contains non-finite values")
    base = base_action.clamp(-1, 1)
    candidates = base[:, None, :].expand(-1, 5, -1).clone()
    candidates[:, 1, 2] += config.wrist_z_delta
    candidates[:, 2, 2] -= config.wrist_z_delta
    candidates[:, 3, list(FINGER_INDICES)] += config.finger_delta
    candidates[:, 4, 2] += config.wrist_z_delta
    candidates[:, 4, list(FINGER_INDICES)] += config.finger_delta
    return candidates.clamp(-1, 1)


@torch.inference_mode()
def select_cmlite_action(
    cm: FrozenCmLite,
    q: torch.Tensor,
    base_action: torch.Tensor,
    object_state: torch.Tensor,
    goal_position: torch.Tensor,
    *,
    actual_contact: torch.Tensor | None = None,
    config: ProposalConfig = ProposalConfig(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select a proposal by predicted one-step object progress.

    Returns selected actions, all candidate scores, and selected candidate ids.
    A predicted contact threshold prevents a low-confidence model from
    perturbing the policy before it has a plausible interaction.
    """
    if q.ndim != 2 or q.shape[1] != 18 or base_action.shape != q.shape:
        raise ValueError("q and base_action must be [B,18]")
    if object_state.shape != (q.shape[0], 13) or goal_position.shape != (q.shape[0], 3):
        raise ValueError("object_state/goal_position shapes are invalid")
    if actual_contact is not None and (actual_contact.shape != (q.shape[0],) or
                                       actual_contact.dtype != torch.bool):
        raise ValueError("actual_contact must be boolean [B]")
    candidates = proposal_actions(base_action, config)
    batch, count, _ = candidates.shape
    repeated_q = q[:, None, :].expand(-1, count, -1).reshape(batch * count, 18)
    repeated_state = object_state[:, None, :].expand(-1, count, -1).reshape(batch * count, 13)
    prediction = cm.predict(repeated_q, candidates.reshape(batch * count, 18), repeated_state)
    delta = prediction["delta_world"].reshape(batch, count, 3)
    contact = prediction["contact_probability"].reshape(batch, count)
    current = object_state[:, None, :3]
    goal = goal_position[:, None, :]
    current_cost = (current - goal).square().sum(-1)
    predicted_cost = (current + delta - goal).square().sum(-1)
    scores = torch.tanh((current_cost - predicted_cost) / (0.02 ** 2)) * contact
    if config.positive_only:
        # A proposal is allowed to override the learned policy only when the
        # one-step model predicts positive progress.  Candidate zero remains
        # the zero-score fallback when all proposals are harmful or uncertain.
        scores = scores.clamp_min(0)
    eligible = contact >= config.contact_threshold
    if config.require_actual_contact:
        if actual_contact is None:
            raise ValueError("actual_contact is required by the default safe gate")
        eligible = eligible & actual_contact[:, None]
    elif actual_contact is not None:
        eligible = eligible | actual_contact[:, None]
    scores = torch.where(eligible, scores, torch.full_like(scores, -torch.inf))
    scores[:, 0] = torch.where(torch.isfinite(scores[:, 0]), scores[:, 0], torch.zeros_like(scores[:, 0]))
    selected_id = scores.argmax(dim=1)
    selected = candidates[torch.arange(batch, device=q.device), selected_id]
    return selected, scores, selected_id
