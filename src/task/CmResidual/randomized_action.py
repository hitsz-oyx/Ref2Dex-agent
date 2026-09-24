"""CPU-testable balanced randomized assignment for physical interventions."""
from __future__ import annotations

import torch


def balanced_assignment(mask: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if mask.ndim != 1 or mask.dtype != torch.bool:
        raise ValueError("mask must be a one-dimensional boolean tensor")
    selected = mask.nonzero(as_tuple=False).reshape(-1)
    assignment = torch.zeros(len(mask), dtype=torch.int8, device=mask.device)
    if len(selected):
        labels = torch.ones(len(selected), dtype=torch.int8)
        labels[:len(selected) // 2] = -1
        labels = labels[torch.randperm(len(labels), generator=generator)]
        assignment[selected] = labels.to(mask.device)
    return assignment
