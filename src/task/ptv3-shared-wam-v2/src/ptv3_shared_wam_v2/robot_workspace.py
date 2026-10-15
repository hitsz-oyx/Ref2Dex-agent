"""Explicit observed-input admission to the pretrained fixed PTv3 workspace.

This is encoder compatibility, not successful-demonstration qualification.
No future hand, action, or outcome participates in the admission decision.
"""
import numpy as np
import torch

SCHEMA = 'observed-fixed-grid.v1'
# Point.serialization uses bit_length(max+1); 65535 itself requires depth17.
MAX_GRID = (1 << 16) - 2


def observed_workspace_mask(data, canonical, ticks, origin, voxel_m):
    ticks = np.asarray(ticks, dtype=np.int64)
    if not len(ticks):
        return np.zeros(0, dtype=bool)
    pose = data['obj'][ticks]
    hand = np.matmul(data['hand'][ticks] - pose[:, None, :3, 3], pose[:, :3, :3])
    # The current object is the sample anchor, so its CAD points stay unchanged.
    scene = np.asarray(canonical['points'], dtype=np.float32)
    origin = torch.as_tensor(origin, dtype=torch.float32)
    if origin.shape != (3,) or not torch.isfinite(origin).all() or not np.isfinite(voxel_m) or voxel_m <= 0:
        raise ValueError('finite fixed workspace origin and positive voxel required')

    def inside(points):
        value = torch.as_tensor(np.asarray(points, dtype=np.float32))
        grid = torch.floor((value - origin) / float(voxel_m))
        return torch.isfinite(value).all(-1) & (grid >= 0).all(-1) & (grid <= MAX_GRID).all(-1)

    return (inside(hand).all(-1) & inside(scene).all()).numpy()


def restrict_observed_workspace(dataset, origin, voxel_m):
    """Rebuild every sampler/normalization view from one explicit subset."""
    keys, ticks_by_sequence, pools, records = [], [], [], []
    original_count = len(dataset.keys)
    original_start = 0
    for seq, old_ticks in enumerate(dataset.ticks_by_sequence):
        data, canonical = dataset.sequence(seq)
        keep = observed_workspace_mask(data, canonical, old_ticks, origin, voxel_m)
        ticks = old_ticks[keep]
        start = len(keys)
        keys.extend((int(tick), seq) for tick in ticks)
        ticks_by_sequence.append(ticks)
        pools.append(np.arange(start, len(keys), dtype=np.int64))
        records.append(dict(sequence=dataset.records[seq]['sequence'],
                            source_sequence_index=seq, retained_start_index=start,
                            retained_original_indices=(original_start + np.flatnonzero(keep)).tolist(),
                            admitted_ticks=ticks.tolist(), excluded_ticks=old_ticks[~keep].tolist(),
                            original_windows=len(old_ticks), retained_windows=len(ticks)))
        original_start += len(old_ticks)
    if not keys:
        raise ValueError('no windows inside pretrained encoder workspace')
    phase_pools = []
    for startup in (True, False):
        phases = [pool[(ticks < 16) if startup else (ticks >= 16)]
                  for ticks, pool in zip(ticks_by_sequence, pools)]
        phase_pools.append([pool for pool in phases if len(pool)] or [pool for pool in pools if len(pool)])
    dataset.keys, dataset.ticks_by_sequence = keys, ticks_by_sequence
    dataset.pools, dataset.phase_pools = pools, phase_pools
    return dict(schema=SCHEMA, origin_m=list(origin), voxel_m=voxel_m, max_grid=MAX_GRID,
                input_scope='current observed hand and CAD only; not a grasp-success filter',
                original_windows=original_count, retained_windows=len(keys),
                excluded_windows=original_count-len(keys), records=records)
