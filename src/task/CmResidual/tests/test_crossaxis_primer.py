from __future__ import annotations

import numpy as np
import pytest
import torch

from src.task.CmResidual.randomized_action import (
    balanced_three_arm_assignment, execute_crossaxis_primer,
)
from src.task.CmResidual.tools.analyze_crossaxis_primer import contrasts


def test_balanced_three_arm_and_exact_two_step_doses():
    labels = balanced_three_arm_assignment(
        torch.ones(63, dtype=torch.bool), torch.Generator().manual_seed(7))
    assert [(labels == code).sum().item() for code in (-1, 1, 2)] == [21] * 3
    action = torch.zeros(4, 18)
    assignment = torch.tensor([-1, 0, 1, 2], dtype=torch.int8)
    first = execute_crossaxis_primer(action, assignment, .1)
    second = execute_crossaxis_primer(action, assignment, .1, second=True)
    torch.testing.assert_close(first[:, 0], torch.tensor([-.1, 0, .1, 0]))
    torch.testing.assert_close(second[:, 2], torch.tensor([.1, 0, .1, .1]))
    near_limit = action.clone()
    near_limit[0, 2] = .95
    with pytest.raises(ValueError, match="clip"):
        execute_crossaxis_primer(near_limit, assignment, .1, second=True)


def test_primer_contrast_uses_control_and_step_strata():
    labels = np.array([-1, 1, 2, -1, 1, 2])
    steps = np.array([50, 50, 50, 60, 60, 60])
    value = np.array([5., 2., 1., 15., 12., 11.])
    result = contrasts(value, labels, steps)
    assert result["x_minus_vs_control"] == 4
    assert result["x_plus_vs_control"] == 1
