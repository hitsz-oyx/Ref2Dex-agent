"""Pure metadata helpers for corrected native residual probes."""
from __future__ import annotations

from collections.abc import Sequence

import torch


def simulator_seed_from_argv(arguments: Sequence[str]) -> int:
    """Read the simulator seed passed through to DExplore exactly once."""
    values = []
    for index, argument in enumerate(arguments):
        if argument == "--seed":
            if index + 1 >= len(arguments):
                raise ValueError("--seed requires an integer simulator seed")
            values.append(arguments[index + 1])
        elif argument.startswith("--seed="):
            values.append(argument.split("=", 1)[1])
    if len(values) != 1:
        raise ValueError(f"expected exactly one DExplore --seed, found {len(values)}")
    try:
        return int(values[0])
    except ValueError as error:
        raise ValueError(f"invalid simulator seed: {values[0]!r}") from error


def snapshot_window_metadata(
    motion_id: torch.Tensor,
    start_frame: torch.Tensor,
    rest_z: torch.Tensor,
    rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy episode metadata at the trigger, before a later reset can change it."""
    for name, value in (("motion_id", motion_id), ("start_frame", start_frame), ("rest_z", rest_z)):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
    if rows.ndim != 1 or rows.dtype not in (torch.int32, torch.int64):
        raise ValueError("rows must be a one-dimensional integer tensor")
    if len(motion_id) != len(start_frame) or len(motion_id) != len(rest_z):
        raise ValueError("episode metadata lengths differ")
    if len(rows) and (rows.min() < 0 or rows.max() >= len(motion_id)):
        raise IndexError("metadata row index out of range")
    return motion_id[rows].clone(), start_frame[rows].clone(), rest_z[rows].clone()
