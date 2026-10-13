"""Axial 3D RoPE for vision tokens.

Head dimension 64 is not divisible by 3. Channels are split into even groups
20/22/22 over x/y/z. The base is 10000. Instruction tokens are not rotated.
"""

from __future__ import annotations

import torch


def rope_channel_splits(head_dim: int) -> tuple[int, int, int]:
    """Even axial widths. Extra pairs go to y, then z, then x."""
    if head_dim < 6 or head_dim % 2:
        raise ValueError("axial RoPE needs an even head dimension of at least 6")
    each = (head_dim // 3) // 2 * 2
    if each == 0:
        raise ValueError("head dimension is too small to split across xyz")
    parts = [each, each, each]
    extra = head_dim - each * 3
    axis = 1
    while extra:
        parts[axis % 3] += 2
        extra -= 2
        axis += 1
    return parts[0], parts[1], parts[2]


def _rotate_pairs(values: torch.Tensor, position: torch.Tensor, base: float) -> torch.Tensor:
    half = values.shape[-1] // 2
    exponent = torch.arange(half, device=values.device, dtype=torch.float32) / half
    frequency = base ** (-exponent)
    angle = position.to(torch.float32)[:, None] * frequency[None, :]
    cos = angle.cos().to(values.dtype)
    sin = angle.sin().to(values.dtype)
    first = values[..., 0::2]
    second = values[..., 1::2]
    rotated_first = first * cos - second * sin
    rotated_second = first * sin + second * cos
    return torch.stack((rotated_first, rotated_second), dim=-1).flatten(-2)


def apply_axial_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    coords: torch.Tensor,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate vision query and key. ``coords`` is ``[M, 3]`` patch coordinates."""
    if query.shape != key.shape or query.ndim != 4:
        raise ValueError("query and key must be [batch, heads, tokens, head_dim]")
    if coords.shape != (query.shape[2], 3):
        raise ValueError("patch coordinates must align with the token axis")
    splits = rope_channel_splits(query.shape[-1])
    rotated_query = []
    rotated_key = []
    start = 0
    for axis, width in enumerate(splits):
        stop = start + width
        rotated_query.append(_rotate_pairs(query[..., start:stop], coords[:, axis], base))
        rotated_key.append(_rotate_pairs(key[..., start:stop], coords[:, axis], base))
        start = stop
    return torch.cat(rotated_query, dim=-1), torch.cat(rotated_key, dim=-1)
