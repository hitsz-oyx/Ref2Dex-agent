"""Measured robot state and planned reference poses; no future measurements."""
import numpy as np

from .retargeter_data import ACTIVE

CLOCK_OFFSETS = np.array((8, 16, 24), np.int64)
ON_TRACK_M = 0.005
BEHIND_ADVANCES_M = (0.01, 0.02, 0.04)
BEHIND_HEIGHT_M = 0.03


def progress_goal_indices(current_object, reference_object, tick, previous_index=0):
    """Keep the demonstration clock while the object matches it.

    Once the live object is more than 5 mm from reference[tick], hold a
    monotonic cursor at the last matching frame and take goals 10/20/40 mm
    ahead. A goal more than 30 mm above that cursor is clamped back, so a
    late query cannot point at the post-grasp airborne demonstration.
    """
    current = np.asarray(current_object, np.float32)
    reference = np.asarray(reference_object, np.float32)
    tick, previous = int(tick), int(previous_index)
    if current.shape != (4, 4) or reference.ndim != 3 or reference.shape[1:] != (4, 4):
        raise ValueError('current object and reference object poses required')
    last = len(reference) - 1
    if last < 24 or not 0 <= previous <= tick <= last:
        raise ValueError('tick outside reference')
    on_track = float(np.linalg.norm(reference[tick, :3, 3] - current[:3, 3])) <= ON_TRACK_M
    if on_track:
        return tick, np.minimum(tick + CLOCK_OFFSETS, last)
    distance = np.linalg.norm(reference[:tick + 1, :3, 3] - current[:3, 3], axis=-1)
    matched = np.flatnonzero(distance <= ON_TRACK_M)
    achieved = int(matched[-1]) if len(matched) else int(np.argmin(distance))
    progress = min(tick, max(previous, achieved))
    origin = reference[progress, :3, 3]
    ahead = np.linalg.norm(reference[progress:, :3, 3] - origin, axis=-1)
    z0 = float(reference[progress, 2, 3])
    indices = []
    for advance in BEHIND_ADVANCES_M:
        hit = np.flatnonzero(ahead >= advance)
        index = progress + int(hit[0]) if len(hit) else last
        if float(reference[index, 2, 3]) - z0 > BEHIND_HEIGHT_M:
            allowed = np.flatnonzero(reference[progress:index + 1, 2, 3] - z0 <= BEHIND_HEIGHT_M)
            index = progress + int(allowed[-1]) if len(allowed) else progress
        indices.append(index)
    indices = np.asarray(indices, np.int64)
    if np.any(indices < progress) or np.any(np.diff(indices) < 0):
        raise ValueError('goal indices must stay at or after progress')
    return progress, indices


def control_context(q, dq, previous_applied, current_object, reference_poses):
    q, dq, previous = [np.asarray(x, np.float32) for x in (q, dq, previous_applied)]
    obj, ref = [np.asarray(x, np.float32) for x in (current_object, reference_poses)]
    if q.shape != (18,) or dq.shape != (18,) or previous.shape != (18,):
        raise ValueError('native18 q/dq/previous actual command required')
    if obj.shape != (4, 4) or ref.shape != (3, 4, 4):
        raise ValueError('own current object and three planned reference poses required')
    rotation = np.einsum('ij,kjl->kil', obj[:3, :3].T, ref[:, :3, :3])
    translation = (ref[:, :3, 3] - obj[:3, 3]) @ obj[:3, :3]
    # Rotation6D = first column followed by second column, not row-major flatten.
    goal = np.concatenate((translation, rotation[:, :, 0], rotation[:, :, 1]), -1).reshape(27)
    prop = np.concatenate((q, dq, previous[list(ACTIVE)]))
    if not np.isfinite(prop).all() or not np.isfinite(goal).all():
        raise ValueError('finite measured control context required')
    return prop.astype(np.float32), goal.astype(np.float32)
