"""Distinguish true rigid motion from inconsistent object correspondences."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/audit'))
from audit_video_rigidity import rigid_residual


def test_large_rigid_motion_has_zero_internal_residual():
    points = np.random.RandomState(0).normal(size=(32, 3)) * .1
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    future = points @ rotation + np.array([.3, -.2, .1])
    assert np.linalg.norm(future - points, axis=1).mean() > .2
    assert rigid_residual(points, future).max() < 1e-12


def test_nonrigid_depth_distortion_is_not_absorbed_by_se3():
    points = np.random.RandomState(1).normal(size=(32, 3)) * .1
    future = points.copy()
    future[:, 2] *= 1.5
    assert rigid_residual(points, future).mean() > .01
    assert rigid_residual(points[:7], future[:7]) is None
    line = np.stack((np.arange(12), np.zeros(12), np.zeros(12)), axis=-1)
    assert rigid_residual(line, line + 1) is None
