import importlib.util
from pathlib import Path

import numpy as np
import pytest

script = Path(__file__).resolve().parents[1] / 'tools/audit/audit_video_image_motion.py'
spec = importlib.util.spec_from_file_location('image_motion_audit', script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_world_projection_uses_camera_translation_and_is_depth_ray_invariant():
    intrinsic = np.asarray([[120, 0, 10], [0, 150, 20], [0, 0, 1.]])
    w2c = np.eye(4)
    w2c[:3, 3] = [1, -2, 2]
    world = np.asarray([[1, 2, 4], [0, 3, 0.]])
    expected = np.asarray([[50, 20], [70, 95.]])
    np.testing.assert_allclose(module.project_world(world, w2c, intrinsic), expected)
    deeper_camera = (world + w2c[:3, 3]) * 2
    deeper_world = deeper_camera - w2c[:3, 3]
    np.testing.assert_allclose(module.project_world(deeper_world, w2c, intrinsic), expected)


def test_invalid_projection_fails_without_changing_metric_support():
    with pytest.raises(ValueError):
        module.project_world(np.asarray([[1, 1, -1.]]), np.eye(4), np.eye(3))
    with pytest.raises(ValueError):
        module.project_world(np.asarray([[np.nan, 1, 1.]]), np.eye(4), np.eye(3))
