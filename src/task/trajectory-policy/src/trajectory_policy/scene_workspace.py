"""Drop a few live scene points that fall outside the fixed PTv3 grid.

Training still rejects every out-of-range valid point. Inference uses this
only so one boundary point cannot close the IPC peer mid-episode.
"""
import torch

MAX_SERIALIZED_GRID_COORD = (1 << 16) - 1


def mask_scene_outliers(batch, origin, voxel_size, limit=8):
    xyz = batch['xyz']
    valid = batch['point_valid'].bool()
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or valid.shape != xyz.shape[:2]:
        raise ValueError('scene xyz/point_valid contract differs')
    if not torch.isfinite(torch.as_tensor(float(voxel_size))) or float(voxel_size) <= 0:
        raise ValueError('voxel size must be positive')
    origin = torch.as_tensor(origin, device=xyz.device, dtype=torch.float32)
    if origin.shape != (3,) or not torch.isfinite(origin).all():
        raise ValueError('workspace origin must be three finite coordinates')
    grid = torch.floor((xyz.float() - origin) / float(voxel_size))
    finite = torch.isfinite(xyz).all(-1)
    outside = valid & (~finite | (grid < 0).any(-1) | (grid > MAX_SERIALIZED_GRID_COORD).any(-1))
    # The limit is per environment. A batched forward must not inherit the sum.
    row_dropped = outside.sum(dim=1)
    kept = valid & ~outside
    row_kept = kept.sum(dim=1)
    row_had = valid.any(dim=1)
    bad = (row_dropped > limit) | (row_had & (row_kept == 0))
    if bool(bad.any()):
        raise ValueError(
            f'workspace outlier count {int(row_dropped.max())} exceeds per-query limit {limit}')
    batch['point_valid'] = kept
    batch['_dropped_per_row'] = [int(value) for value in row_dropped]
    return int(row_dropped.sum())
