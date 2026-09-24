from __future__ import annotations

import torch

from src.task.CmResidual.randomized_action import (
    balanced_assignment, balanced_axis_assignment, execute_signed_axis_dose,
)


def test_balanced_assignment_is_masked_balanced_and_reproducible():
    mask = torch.tensor([True, False, True, True, False, True, True])
    first = balanced_assignment(mask, torch.Generator().manual_seed(17))
    second = balanced_assignment(mask, torch.Generator().manual_seed(17))
    torch.testing.assert_close(first, second)
    assert torch.equal(first[~mask], torch.zeros(2, dtype=torch.int8))
    assert (first[mask] != 0).all()
    assert abs(int((first == 1).sum()) - int((first == -1).sum())) <= 1


def test_balanced_assignment_handles_empty_mask():
    assignment = balanced_assignment(torch.zeros(4, dtype=torch.bool),
                                     torch.Generator().manual_seed(1))
    assert torch.equal(assignment, torch.zeros(4, dtype=torch.int8))


def test_multiaxis_assignment_is_masked_balanced_and_reproducible():
    mask = torch.tensor([True] * 37 + [False] * 3)
    first = balanced_axis_assignment(mask, torch.Generator().manual_seed(42), 3)
    second = balanced_axis_assignment(mask, torch.Generator().manual_seed(42), 3)
    torch.testing.assert_close(first, second)
    assert (first[~mask] == 0).all()
    counts = [int((first == code).sum()) for code in (-3, -2, -1, 1, 2, 3)]
    assert max(counts) - min(counts) <= 1


def test_multiaxis_dose_changes_exactly_one_unclipped_axis():
    action = torch.zeros(7, 18)
    assignment = torch.tensor([1, -1, 2, -2, 3, -3, 0], dtype=torch.int8)
    actual = execute_signed_axis_dose(action, assignment, (0, 1, 2), .1)
    for row, code in enumerate(assignment):
        expected = torch.zeros(18)
        if code:
            expected[abs(int(code)) - 1] = .1 * int(code.sign())
        torch.testing.assert_close(actual[row], expected)
    action[0, 0] = .95
    import pytest
    with pytest.raises(ValueError, match="clip"):
        execute_signed_axis_dose(action, assignment, (0, 1, 2), .1)
