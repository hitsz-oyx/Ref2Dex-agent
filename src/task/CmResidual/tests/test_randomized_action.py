from __future__ import annotations

import torch

from src.task.CmResidual.randomized_action import balanced_assignment


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
