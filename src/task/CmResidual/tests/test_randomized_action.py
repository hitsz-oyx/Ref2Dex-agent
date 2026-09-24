from __future__ import annotations

import torch

from src.task.CmResidual.randomized_action import (
    FINGER_SYNERGY_INDICES, balanced_assignment, balanced_axis_assignment,
    execute_finger_synergy_dose, execute_finger_primer_lift,
    execute_signed_axis_dose,
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


def test_finger_synergy_changes_only_five_flexion_commands():
    action = torch.zeros(3, 18)
    assignment = torch.tensor([1, -1, 0], dtype=torch.int8)
    actual = execute_finger_synergy_dose(action, assignment, .2)
    expected = torch.zeros_like(action)
    expected[0, list(FINGER_SYNERGY_INDICES)] = .2
    expected[1, list(FINGER_SYNERGY_INDICES)] = -.2
    torch.testing.assert_close(actual, expected)
    action[0, 6] = .9
    import pytest
    with pytest.raises(ValueError, match="clip"):
        execute_finger_synergy_dose(action, assignment, .2)


def test_finger_primer_then_common_lift_exact_doses():
    action = torch.zeros(3, 18)
    assignment = torch.tensor([1, -1, 0], dtype=torch.int8)
    first = execute_finger_primer_lift(action, assignment,
                                       finger_delta=.2, lift_delta=.1)
    second = execute_finger_primer_lift(action, assignment,
                                        finger_delta=.2, lift_delta=.1,
                                        second=True)
    assert torch.allclose(first[0, list(FINGER_SYNERGY_INDICES)],
                          torch.full((5,), .2))
    assert torch.allclose(first[1, list(FINGER_SYNERGY_INDICES)],
                          torch.full((5,), -.2))
    assert torch.allclose(second[:, 2], torch.tensor([.1, .1, 0.]))
    assert (second[:, list(FINGER_SYNERGY_INDICES)] == 0).all()
