"""Network-free 24-frame targets. Live hand is deliberately absent from this API."""
import numpy as np
from scipy.spatial.transform import Rotation


def object_local_hand(reference_hand, reference_objects):
    hand, objects = np.asarray(reference_hand), np.asarray(reference_objects)
    if hand.shape != (len(objects), 11, 3) or objects.ndim != 2 or objects.shape[1] < 7:
        raise ValueError('aligned Tx11x3 hands and Tx7+ object poses required')
    if not np.isfinite(hand).all() or not np.isfinite(objects[:, :7]).all():
        raise ValueError('finite reference required')
    rotations = Rotation.from_quat(objects[:, 3:7]).as_matrix()
    return np.einsum('tki,tij->tkj', hand-objects[:, None, :3], rotations)


def geometry_future(local_reference, frame, live_object_pose, *, horizon=24):
    """Map EACH reference future's object-local hand to the current object.

    Clock phase has no hand matching. Missing full horizon is invalid, without
    repeating the last frame. No source object lift is added to these targets;
    postgrasp object motion belongs to the common measured consumer/task goal.
    """
    local, pose = np.asarray(local_reference), np.asarray(live_object_pose)
    if local.ndim != 3 or local.shape[1:] != (11, 3) or pose.shape != (4, 4):
        raise ValueError('Tx11x3 reference and 4x4 live object pose required')
    if frame < 0 or horizon < 1 or not np.isfinite(local).all() or not np.isfinite(pose).all():
        raise ValueError('finite geometry, nonnegative clock and positive horizon required')
    if frame+horizon >= len(local):
        return None, False
    future = local[frame+np.arange(1, horizon+1)] @ pose[:3, :3].T + pose[:3, 3]
    return future, True
