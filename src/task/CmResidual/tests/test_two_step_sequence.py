from __future__ import annotations

import numpy as np
import pytest
import torch

from src.task.CmResidual.randomized_action import (
    balanced_axis_assignment, execute_sequence_axis_dose, execute_sequence_x_dose,
)
from src.task.CmResidual.tools.analyze_two_step_sequence import contrasts


def test_sequence_dose_is_exact_and_only_long_arm_repeats():
    action = torch.zeros(5, 18)
    assignment = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.int8)
    first = execute_sequence_x_dose(action, assignment, .1)
    second = execute_sequence_x_dose(action, assignment, .1, second=True)
    torch.testing.assert_close(first[:, 0], torch.tensor([-.1, -.1, 0, .1, .1]))
    torch.testing.assert_close(second[:, 0], torch.tensor([-.1, 0, 0, 0, .1]))
    assert torch.all(first[:, 1:] == 0)
    near_limit = action.clone()
    near_limit[0, 0] = -.95
    with pytest.raises(ValueError, match="clip"):
        execute_sequence_x_dose(near_limit, assignment, .1, second=True)
    z_second = execute_sequence_axis_dose(action, assignment, .1, 2, second=True)
    torch.testing.assert_close(z_second[:, 2], torch.tensor([-.1, 0, 0, 0, .1]))
    assert torch.all(z_second[:, :2] == 0)


def test_four_sequence_cells_balance_and_interaction():
    mask = torch.ones(64, dtype=torch.bool)
    assignment = balanced_axis_assignment(mask, torch.Generator().manual_seed(1), 2)
    assert sorted([(assignment == label).sum().item()
                   for label in (-2, -1, 1, 2)]) == [16] * 4
    labels = np.tile(np.array([-2, -1, 1, 2]), 2)
    steps = np.repeat([50, 60], 4)
    outcome = np.tile(np.array([-4., -1., 1., 4.]), 2)
    result = contrasts(outcome, labels, steps)
    assert result["one_effect"] == 2
    assert result["two_effect"] == 8
    assert result["interaction"] == 6
    assert result["two_minus_one_contact"] == 0
