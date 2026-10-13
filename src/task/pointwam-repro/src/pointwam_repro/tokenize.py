"""Shared 1 cm voxels and non-overlapping 5^3 patches.

Several points in one voxel are averaged. That replaces the paper's sparse
convolution stem. Empty patch slots stay zero.
"""

from __future__ import annotations

import torch


def voxel_patch_slots(voxel: torch.Tensor, patch_size: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Return patch coordinates and the fixed 0..124 slot of each voxel."""
    if voxel.ndim != 2 or voxel.shape[-1] != 3:
        raise ValueError("voxel coordinates must be [N, 3]")
    patch = torch.div(voxel, patch_size, rounding_mode="floor")
    local = voxel - patch * patch_size
    slot = local[:, 0] + patch_size * local[:, 1] + (patch_size ** 2) * local[:, 2]
    return patch, slot


def patch_slots(xyz: torch.Tensor, voxel_m: float = 0.01, patch_size: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Return integer patch coordinates and the 0..124 slot inside each patch."""
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError("xyz must be [N, 3]")
    voxel = torch.floor(xyz / voxel_m).to(torch.int64)
    return voxel_patch_slots(voxel, patch_size)


def pack_patches(features: torch.Tensor, xyz: torch.Tensor, patch_size: int = 5) -> dict[str, torch.Tensor]:
    """Pack point features into ordered patch tensors.

    The result has ``tokens [M, slots, C]``, ``point_patch [N]`` and
    ``point_slot [N]``. ``point_patch`` indexes the packed patch axis.
    """
    if features.ndim != 2 or features.shape[0] != xyz.shape[0]:
        raise ValueError("features and xyz must describe the same points")
    patch, slot = patch_slots(xyz, patch_size=patch_size)
    slots = patch_size ** 3
    keys: list[tuple[int, int, int]] = []
    index: dict[tuple[int, int, int], int] = {}
    point_patch = torch.empty(features.shape[0], dtype=torch.int64)
    for point, key in enumerate(tuple(row.tolist()) for row in patch):
        identity = (int(key[0]), int(key[1]), int(key[2]))
        if identity not in index:
            index[identity] = len(keys)
            keys.append(identity)
        point_patch[point] = index[identity]
    tokens = features.new_zeros((len(keys), slots, features.shape[1]))
    counts = features.new_zeros((len(keys), slots, 1))
    tokens.index_put_((point_patch, slot), features, accumulate=True)
    counts.index_put_((point_patch, slot), torch.ones_like(features[:, :1]), accumulate=True)
    tokens = tokens / counts.clamp_min(1)
    return {
        "tokens": tokens,
        "point_patch": point_patch,
        "point_slot": slot,
        "patch_xyz": torch.tensor(keys, dtype=torch.int64),
    }
