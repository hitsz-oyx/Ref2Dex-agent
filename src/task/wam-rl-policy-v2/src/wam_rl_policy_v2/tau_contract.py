"""Explicit conversion from WAM-v4 hand trajectories to executor tau.

The v4 training contract is 30 Hz, 24 native steps, in the current object's
fixed anchor frame.  The existing GT-tau student contract is 60 Hz, 8 native
steps, in the simulator world frame.  These are different clocks, frames and
shapes; slicing the first eight v4 rows silently halves the time horizon.
"""
from __future__ import annotations

import numpy as np

V4_FPS = 30
V4_HORIZON = 24
NATIVE_FPS = 60
STUDENT_HORIZON = 8
HAND_SLOT = {"right": 0, "left": 1}
TAU_FRAME_ANCHOR = "object_anchor"
TAU_FRAME_WORLD = "world"


def _slot(hand_slot):
    if isinstance(hand_slot, str):
        if hand_slot not in HAND_SLOT:
            raise ValueError("hand_slot must be 'right' or 'left'")
        return HAND_SLOT[hand_slot]
    index = int(hand_slot)
    if index not in (0, 1):
        raise ValueError("hand_slot index must be 0 (right) or 1 (left)")
    return index


def v4_trajectory_to_student(trajectory, *, object_rotation=None,
                             hand_slot="right", student_horizon=STUDENT_HORIZON,
                             frame=TAU_FRAME_WORLD, v4_fps=V4_FPS,
                             native_fps=NATIVE_FPS):
    """Interpolate v4 hand displacement onto the native student clock.

    Parameters
    ----------
    trajectory:
        ``[..., 24, 2, 11, 3]`` v4 trajectory or ``[..., 24, 11, 3]`` already
        selected hand.  Values are metres relative to the current hand in the
        current object anchor frame, with row ``i`` at ``(i+1)/v4_fps``.
    object_rotation:
        ``3x3`` world-from-anchor rotation of the current object.  Required
        for ``frame='world'``; the solver uses it to rotate the displacement.
    hand_slot:
        ``'right'``/``'left'`` or 0/1 for a two-hand v4 output.
    student_horizon:
        Number of native steps (default 8 at 60 Hz).
    frame:
        ``'world'`` (existing student) or ``'object_anchor'``.

    Returns
    -------
    ``[..., student_horizon, 11, 3]`` float32 tau.  The implicit
    ``tau(0)=0`` point is prepended before interpolation.
    """
    values = np.asarray(trajectory, dtype=np.float64)
    if values.shape[-4:] == (V4_HORIZON, 2, 11, 3):
        values = values[..., _slot(hand_slot), :, :]
    elif values.shape[-3:] != (V4_HORIZON, 11, 3):
        raise ValueError("trajectory must be [...,24,2,11,3] or [...,24,11,3]")
    if not np.isfinite(values).all():
        raise ValueError("finite v4 trajectory required")
    if not 0 < student_horizon <= V4_HORIZON:
        raise ValueError("bounded student horizon required")
    if not np.isfinite(v4_fps) or v4_fps <= 0 or not np.isfinite(native_fps) or native_fps <= 0:
        raise ValueError("positive finite clocks required")
    if frame not in (TAU_FRAME_ANCHOR, TAU_FRAME_WORLD):
        raise ValueError("frame must be 'world' or 'object_anchor'")

    # v4 row i is at t+(i+1)/30.  Prepend the current instant so tau(0)=0
    # is represented explicitly rather than assumed by the consumer.
    source_times = np.arange(1, V4_HORIZON + 1, dtype=np.float64) / float(v4_fps)
    source = np.concatenate((np.zeros_like(values[..., :1, :, :]), values), axis=-3)
    target_times = np.arange(1, int(student_horizon) + 1, dtype=np.float64) / float(native_fps)
    if target_times[-1] > source_times[-1] + 1e-12:
        raise ValueError("student horizon exceeds v4 coverage")
    leading = source.shape[:-3]
    channels = 11 * 3
    flat = source.reshape(-1, V4_HORIZON + 1, channels)
    full_times = np.concatenate(([0.], source_times))
    result = np.empty((len(flat), len(target_times), channels), np.float64)
    for row in range(len(flat)):
        for channel in range(channels):
            result[row, :, channel] = np.interp(target_times, full_times, flat[row, :, channel])
    result = result.reshape(*leading, len(target_times), 11, 3)
    if frame == TAU_FRAME_WORLD:
        if object_rotation is None:
            raise ValueError("world-frame tau requires the current object rotation")
        rotation = np.asarray(object_rotation, dtype=np.float64)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
            raise ValueError("proper finite 3x3 object rotation required")
        result = result @ rotation.T
    return result.astype(np.float32)
