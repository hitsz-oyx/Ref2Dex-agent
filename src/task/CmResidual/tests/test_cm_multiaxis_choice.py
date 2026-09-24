from __future__ import annotations

import torch

from src.task.CmResidual.cm_multiaxis_choice import apply_choice, select_x_or_z


def test_selector_prioritizes_safe_lift_then_contact_then_base():
    future = torch.zeros(3, 4, 3)
    contact = torch.full((3, 4), .5)
    future[2, :, 2] = torch.tensor([.006, .006, .001, .006])
    contact[2] += torch.tensor([-.01, -.03, 0, -.03])
    future[1, :, 2] = torch.tensor([-.004, -.004, -.006, -.004])
    contact[1] += torch.tensor([.04, .04, .04, .02])
    choice = select_x_or_z(future, contact)
    torch.testing.assert_close(choice, torch.tensor([2, 1, 0, 0], dtype=torch.int8))
    action = torch.zeros(4, 18)
    result = apply_choice(action, choice)
    torch.testing.assert_close(result[:, 0], torch.tensor([0, .1, 0, 0]))
    torch.testing.assert_close(result[:, 2], torch.tensor([.1, 0, 0, 0]))
