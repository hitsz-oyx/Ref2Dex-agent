"""Explicit desired object-point motion, separate from observed World inputs."""
import numpy as np


def current_spread_indices(xyz, candidates, count=32):
    """Deterministic farthest-point coverage using CURRENT geometry only."""
    xyz = np.asarray(xyz)
    ids = np.asarray(candidates)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or count < 1:
        raise ValueError('integer current-point candidates and positive count required')
    if len(np.unique(ids)) != len(ids) or (ids < 0).any() or (ids >= len(xyz)).any():
        raise ValueError('unique in-range current point identities required')
    ids = np.sort(ids)
    if not len(ids):
        return ids
    points = xyz[ids]
    if not np.isfinite(points).all():
        raise ValueError('finite current target points required')
    selected = [0]
    nearest = np.sum((points - points[0])**2, -1)
    while len(selected) < min(count, len(ids)):
        nearest[selected] = -1
        choice = int(np.argmax(nearest))
        selected.append(choice)
        nearest = np.minimum(nearest, np.sum((points-points[choice])**2, -1))
    return ids[selected]


def pack_object_motion_goal(sample, current_target_indices, offsets=(8, 16, 24), count=32):
    """Hindsight object tracks become declared desired goals, never observations.

    The caller must admit target identity from current evidence. Future validity
    masks desired coordinates but NEVER selects or reorders current identities.
    This function reads no hand trajectories, robot commands or learned outputs.
    """
    xyz = np.asarray(sample['query_xyz'], np.float32)
    current_valid = np.asarray(sample['query_valid'], bool)
    future = np.asarray(sample['point_target'], np.float32)
    valid = np.asarray(sample['point_target_valid'], bool)
    offsets = np.asarray(offsets)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or current_valid.shape != xyz.shape[:1]:
        raise ValueError('current Nx3 point geometry required')
    if future.ndim != 3 or future.shape[1:] != xyz.shape or valid.shape != future.shape[:-1]:
        raise ValueError('future trajectories must preserve current point identities')
    if (offsets.ndim != 1 or offsets.dtype.kind not in 'iu' or not len(offsets)
            or (offsets < 1).any() or (offsets > len(future)).any() or (np.diff(offsets) <= 0).any()):
        raise ValueError('strictly increasing in-range future ticks required')
    ids = current_spread_indices(xyz, current_target_indices, count)
    if not current_valid[ids].all():
        raise ValueError('target selection includes unobserved current points')
    start = np.zeros((count, 3), np.float32)
    desired = np.zeros((len(offsets), count, 3), np.float32)
    mask = np.zeros((len(offsets), count), bool)
    current_mask = np.zeros(count, bool)
    identities = np.full(count, -1, np.int64)
    n = len(ids)
    start[:n] = xyz[ids]; current_mask[:n] = True; identities[:n] = ids
    available = valid[offsets-1][:, ids]
    values = future[offsets-1][:, ids]
    if not np.isfinite(values[available]).all():
        raise ValueError('nonfinite valid desired position')
    desired[:, :n] = np.where(available[..., None], values, 0.)
    mask[:, :n] = available
    return dict(start_object_points=start, desired_object_points=desired,
                current_object_valid=current_mask, desired_object_valid=mask,
                goal_offsets=offsets.astype(np.int64), goal_times_s=offsets.astype(np.float32)/30,
                selected_query_indices=identities)


def rigid_consistency(goal):
    """Raw-point consistency diagnostic, not a correction or accuracy claim."""
    reports = []
    for tick, desired, mask in zip(goal['goal_offsets'], goal['desired_object_points'], goal['desired_object_valid']):
        start = goal['start_object_points'][mask].astype(np.float64)
        end = desired[mask].astype(np.float64)
        row = dict(offset=int(tick), valid_points=len(start))
        if len(start) < 3:
            row.update(identifiable=False); reports.append(row); continue
        a, b = start-start.mean(0), end-end.mean(0)
        u, singular, vt = np.linalg.svd(a.T@b)
        correction = np.eye(3); correction[2, 2] = np.linalg.det(u@vt)
        rotation = u@correction@vt
        error = np.linalg.norm(a@rotation-b, axis=-1)*1000
        row.update(identifiable=bool(np.linalg.matrix_rank(a, tol=1e-5)>=2),
                   residual_median_mm=float(np.median(error)), residual_p95_mm=float(np.percentile(error, 95)),
                   median_displacement_mm=float(np.median(np.linalg.norm(end-start, axis=-1))*1000))
        reports.append(row)
    return reports
