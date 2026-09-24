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


def balanced_axis_assignment(mask: torch.Tensor, generator: torch.Generator,
                             num_axes: int) -> torch.Tensor:
    """Assign eligible rows to signed axis codes ±1..±num_axes."""
    if mask.ndim != 1 or mask.dtype != torch.bool:
        raise ValueError("mask must be a one-dimensional boolean tensor")
    if not 2 <= num_axes <= 6:
        raise ValueError("num_axes must be in [2, 6]")
    selected = mask.nonzero(as_tuple=False).reshape(-1)
    assignment = torch.zeros(len(mask), dtype=torch.int8, device=mask.device)
    if len(selected):
        codes = torch.arange(len(selected), dtype=torch.long) % (2 * num_axes)
        labels = codes // 2 + 1
        labels[codes % 2 == 1] *= -1
        labels = labels[torch.randperm(len(labels), generator=generator)]
        assignment[selected] = labels.to(mask.device, dtype=torch.int8)
    return assignment


def execute_signed_axis_dose(action: torch.Tensor, assignment: torch.Tensor,
                             axes: tuple[int, ...], delta: float) -> torch.Tensor:
    """Apply exactly one unclipped signed dose to each assigned row."""
    if action.ndim != 2 or assignment.shape != (len(action),) or (
            len(set(axes)) != len(axes) or not axes or
            any(axis < 0 or axis >= action.shape[1] for axis in axes) or
            not 0 < delta <= .5 or not torch.isfinite(action).all()):
        raise ValueError("invalid axis-dose input")
    if (assignment.abs() > len(axes)).any():
        raise ValueError("assignment code outside available axes")
    executed = action.detach().clone()
    for code, axis in enumerate(axes, 1):
        chosen = assignment.abs() == code
        executed[chosen, axis] += delta * assignment[chosen].sign()
        if (executed[chosen, axis].abs() > 1 + 1e-6).any():
            raise ValueError("axis dose would clip")
    return executed
