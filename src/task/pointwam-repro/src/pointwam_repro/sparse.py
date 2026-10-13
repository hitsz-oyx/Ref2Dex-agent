"""1 cm sparse stem from Appendix B.3.

The stem is a 3^3 sparse convolution to 64 channels, LayerNorm, GELU, and two
residual blocks. Empty neighbors contribute nothing, which is the zero fill of
a sparse kernel. PointWAM does not specify the residual block, so each block
is two 3^3 sparse convolutions with LayerNorm and GELU and a residual add.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def kernel_offsets() -> torch.Tensor:
    """27 neighbor offsets with the center at index 13."""
    values = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    return torch.tensor(values, dtype=torch.int64)


def pack_voxel_key(coords: torch.Tensor) -> torch.Tensor:
    """Pack integer voxel coordinates into one signed-safe int64 key."""
    bias = 1 << 20
    shifted = coords.to(torch.int64) + bias
    if bool((shifted < 0).any() or (shifted >= (1 << 21)).any()):
        raise ValueError("voxel coordinates exceed +/- 10 km at 1 cm")
    return (shifted[:, 0] << 42) | (shifted[:, 1] << 21) | shifted[:, 2]


class SparseConv3d(nn.Module):
    """3^3 sparse convolution defined on occupied voxels only."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        weight = torch.empty(out_channels, 27 * in_channels)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight.view(out_channels, 27, in_channels).permute(1, 2, 0).contiguous())
        self.bias = nn.Parameter(torch.empty(out_channels))
        fan_in = 27 * in_channels
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.register_buffer("offsets", kernel_offsets(), persistent=False)

    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or coords.shape != (features.shape[0], 3):
            raise ValueError("sparse conv expects one feature row per voxel")
        keys = pack_voxel_key(coords)
        order = torch.argsort(keys)
        sorted_keys = keys[order]
        occupied = features.shape[0]
        output = features.new_zeros((occupied, self.weight.shape[-1]))
        for index, offset in enumerate(self.offsets):
            query = pack_voxel_key(coords + offset)
            position = torch.searchsorted(sorted_keys, query).clamp(max=occupied - 1)
            found = sorted_keys[position] == query
            neighbor = features[order[position]]
            contribution = neighbor @ self.weight[index]
            output = output + contribution * found.to(features.dtype).unsqueeze(-1)
        return output + self.bias


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = SparseConv3d(channels, channels)
        self.norm1 = nn.LayerNorm(channels)
        self.conv2 = SparseConv3d(channels, channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(self.norm1(self.conv1(features, coords)))
        update = self.norm2(self.conv2(hidden, coords))
        return torch.nn.functional.gelu(features + update)


class SparseStem(nn.Module):
    """Map per-voxel features to ``d_stem`` with the published stem order."""

    def __init__(self, in_channels: int, stem_dim: int = 64) -> None:
        super().__init__()
        self.conv = SparseConv3d(in_channels, stem_dim)
        self.norm = nn.LayerNorm(stem_dim)
        self.block1 = ResidualBlock(stem_dim)
        self.block2 = ResidualBlock(stem_dim)

    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(self.norm(self.conv(features, coords)))
        hidden = self.block1(hidden, coords)
        return self.block2(hidden, coords)


def mean_pool_voxels(features: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average point features that share a voxel. Returns pooled rows, coords, inverse."""
    if features.ndim != 2 or coords.shape != (features.shape[0], 3):
        raise ValueError("features and voxel coordinates must align")
    unique, inverse = torch.unique(coords, dim=0, return_inverse=True)
    pooled = features.new_zeros((unique.shape[0], features.shape[1]))
    pooled.index_add_(0, inverse, features)
    counts = features.new_zeros((unique.shape[0], 1))
    counts.index_add_(0, inverse, torch.ones((features.shape[0], 1), device=features.device, dtype=features.dtype))
    return pooled / counts, unique, inverse
