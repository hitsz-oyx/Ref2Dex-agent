"""Measured robot state and planned reference poses; no future measurements."""
import numpy as np

from .retargeter_data import ACTIVE


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
