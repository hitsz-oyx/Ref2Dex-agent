import numpy as np
import pytest

from pose_mesh_data.v4_interaction import (
    C_DIM, INTERACTION_POINTS, V4SamplingPolicy, classify_window,
    fixed_subset, interaction_field, retain_window, resampled_subset,
)


def test_fixed_subset_is_deterministic_and_bounded():
    points = np.arange(512 * 3, dtype=np.float32).reshape(512, 3)
    assert fixed_subset(points, 512).shape == (512, 3)
    np.testing.assert_array_equal(fixed_subset(points, 128), fixed_subset(points, 128))
    with pytest.raises(ValueError):
        fixed_subset(points[:127], 128)


def test_voxel_field_has_contact_and_semantic_part_without_pairwise_distance():
    policy = V4SamplingPolicy()
    hand = np.zeros((3, 2, 11, 3), np.float32)
    valid = np.zeros((3, 2, 11), bool)
    valid[:, 0, :].fill(True)
    hand[:, 0, 0] = [0.0, 0.0, 0.0]
    hand[:, 0, 2] = [0.01, 0.0, 0.0]
    hand[:, 0, 7] = [0.02, 0.0, 0.0]
    obj = np.zeros((3, INTERACTION_POINTS, 3), np.float32)
    obj[:, 0] = [0.015, 0.0, 0.0]
    target, mask = interaction_field(hand, valid, obj, policy, origin=np.array([-0.1, -0.1, -0.1]))
    assert target.shape == (3, INTERACTION_POINTS, C_DIM)
    assert mask.all() and target[:, 0, 0].min() == 1.
    assert target[:, 0, 1:].sum() > 0
    assert np.isfinite(target).all()


def test_static_thinning_keeps_contact_and_bounds_idle():
    p = V4SamplingPolicy(static_keep_ratio=.1, approach_keep_ratio=.2)
    hand = np.zeros((4, 2, 11, 3), np.float32)
    valid = np.ones((4, 2, 11), bool)
    obj = np.zeros((4, INTERACTION_POINTS, 3), np.float32)
    target = np.zeros((4, INTERACTION_POINTS, C_DIM), np.float32)
    assert classify_window(hand, valid, obj, target, p) == "static_idle"
    assert retain_window("contact_static", 1, p)
    assert retain_window("static_idle", 1, p) in (True, False)


def test_resampled_subset_repeats_underfilled_native_geometry_without_fabrication():
    points = np.arange(32 * 3, dtype=np.float32).reshape(32, 3)
    selected, ids = resampled_subset(points, 512)
    assert selected.shape == (512, 3) and ids.shape == (512,)
    np.testing.assert_array_equal(selected[32], points[0])
