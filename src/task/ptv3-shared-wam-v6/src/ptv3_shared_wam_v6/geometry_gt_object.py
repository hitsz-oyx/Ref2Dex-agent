"""Dense future object labels from recorded poses, never from predicted32 flow."""
import numpy as np
from . import paths
from ptv3_shared_wam_v5.data import transform


def dense_object_future_gt(sample, sequence, tick, slot):
    points = sample['interaction_points']
    if points.shape != (1, 2048, 3):
        raise ValueError('same ordered2048 CAD contact queries required')
    poses = sequence['object_poses'][tick:tick+25, slot]
    valid = sequence['object_pose_valid'][tick:tick+25, slot]
    if len(poses) != 25 or not valid.all() or not np.isfinite(poses).all():
        raise ValueError('complete finite current and future GT object poses required')
    relative = np.linalg.inv(poses[0]).astype(np.float32) @ poses[1:]
    future = transform(np.broadcast_to(points[0], (24, 2048, 3)), relative)
    return future[None].astype(np.float32)
