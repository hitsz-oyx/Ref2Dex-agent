"""Held-lift shaping rewards grasped height, not ballistic object motion."""
import pytest
import torch

from src.task.CmResidual.dexplore_grasp_reward import (
    contact_lift_progress_reward, held_lift_reward,
)


def test_held_lift_reward_is_bounded_and_requires_both_contacts():
    reward = held_lift_reward(
        torch.tensor([0.90, 0.915, 0.93, 0.96]), torch.full((4,), 0.90),
        torch.tensor([True, True, False, True]),
        torch.tensor([True, True, True, False]))
    torch.testing.assert_close(reward, torch.tensor([0.0, 0.5, 0.0, 0.0]))


def test_held_lift_reward_saturates_at_target_height():
    reward = held_lift_reward(
        torch.tensor([0.929, 0.93, 1.10]), torch.full((3,), 0.90),
        torch.ones(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool))
    torch.testing.assert_close(reward, torch.tensor([29 / 30, 1.0, 1.0]))


def test_held_lift_reward_rejects_nonfinite_height():
    with pytest.raises(FloatingPointError):
        held_lift_reward(torch.tensor([float("nan")]), torch.zeros(1),
                         torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool))


def test_contact_lift_progress_is_signed_bounded_and_contact_supported():
    reward = contact_lift_progress_reward(
        torch.tensor([0.90, 0.90, 0.90, 0.90, 0.90]),
        torch.tensor([0.906, 0.897, 0.902, 0.902, 0.902]),
        torch.tensor([True, True, False, True, True]),
        torch.tensor([True, True, True, False, True]),
        torch.tensor([False, False, False, False, True]))
    torch.testing.assert_close(
        reward, torch.tensor([1.0, -1.0, 0.0, 0.0, 0.0]), atol=2e-5, rtol=0)
