import numpy as np
import pytest
from ptv3_shared_wam_v6.geometry_gt_object import dense_object_future_gt


def test_dense_future_uses_recorded_rotation_translation_and_all2048_ordered_points():
    rng = np.random.default_rng(6)
    points = rng.normal(0, .04, (1, 2048, 3)).astype(np.float32)
    poses = np.broadcast_to(np.eye(4, dtype=np.float32), (30, 1, 4, 4)).copy()
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32)
    poses[3, 0, :3, 3] = [.3, -.2, .1]
    for i in range(4, 28):
        poses[i, 0, :3, :3] = rotation
        poses[i, 0, :3, 3] = [.3+.001*(i-3), -.2, .1]
    sequence = dict(object_poses=poses, object_pose_valid=np.ones((30, 1), bool))
    sample = dict(interaction_points=points, scene_points=np.full((1, 32, 3), np.nan))
    actual = dense_object_future_gt(sample, sequence, 3, 0)
    expected = points[0] @ rotation.T
    np.testing.assert_allclose(actual[0, 7], expected+[.008, 0, 0], atol=1e-7)
    assert actual.shape == (1, 24, 2048, 3) and actual.dtype == np.float32
    sequence['object_pose_valid'][9, 0] = False
    with pytest.raises(ValueError, match='poses'):
        dense_object_future_gt(sample, sequence, 3, 0)
