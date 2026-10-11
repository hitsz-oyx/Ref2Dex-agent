import numpy as np

from trajectory_policy.goal_policy import INPUT_DIM, RESIDUAL_INPUT_DIM
from trajectory_policy.goal_policy_data import (
    build_goal_finger_residual_dataset, build_goal_policy_dataset,
    build_goal_residual_dataset, split_by_environment,
)


def _trajectory(frames=10, envs=4):
    obj = np.tile(np.eye(4, dtype=np.float32), (frames, envs, 1, 1))
    hand = np.zeros((frames, envs, 11, 3), np.float32)
    q = np.zeros((frames, envs, 18), np.float32)
    action = np.zeros((frames - 1, envs, 18), np.float32)
    action[..., 0] = np.arange(frames - 1, dtype=np.float32)[:, None]
    action[..., 1] = 7.
    latent = np.zeros((frames - 1, envs, 12), np.float32)
    latent[..., 0] = np.arange(frames - 1, dtype=np.float32)[:, None]
    latent[..., 1] = -3.
    return dict(dof_position=q, dof_velocity=q.copy(), hand_keypoints=hand,
                object_pose=obj, object_velocity=np.zeros((frames, envs, 6), np.float32),
                action=action, latent=latent)


def test_dataset_uses_goal_geometry_and_splits_complete_environments():
    trajectory = _trajectory()
    reference_hand = np.zeros((10, 11, 3), np.float32)
    reference_hand[3, 0, 0] = .25
    reference_obj = np.tile(np.eye(4, dtype=np.float32), (10, 1, 1))
    data = build_goal_policy_dataset(trajectory, reference_hand, reference_obj,
                                     ["tracker", "tracker", "tracker", "nominal"])
    assert data.features.shape == (3 * 9, INPUT_DIM)
    assert data.labels.shape == (3 * 9, 18)
    assert np.unique(data.env_ids).tolist() == [0, 1, 2]
    train, validation = split_by_environment(data)
    assert np.intersect1d(data.env_ids[train], data.env_ids[validation]).size == 0
    assert train.sum() + validation.sum() == len(data.features)
    # A reference-only change must affect features at the corresponding future
    # goal while labels remain the recorded native action.
    assert data.features[0].max() > 0
    np.testing.assert_equal(data.labels[0, 1], 7.)
    np.testing.assert_array_equal(data.labels[0, 2:], 0)


def test_residual_dataset_uses_self_owned_latent_labels():
    trajectory = _trajectory()
    reference_hand = np.zeros((10, 11, 3), np.float32)
    reference_obj = np.tile(np.eye(4, dtype=np.float32), (10, 1, 1))
    data = build_goal_residual_dataset(
        trajectory, reference_hand, reference_obj,
        ["tracker", "tracker", "tracker", "nominal"], role="tracker")
    assert data.labels.shape == (3 * 9, 12)
    np.testing.assert_array_equal(data.labels[:, 0], np.tile(np.arange(9, dtype=np.float32), 3))
    assert data.features.shape[1] == RESIDUAL_INPUT_DIM
    previous_offset = 18 + 18 + 11 * 3 + 24 * 11 * 3 + 24 * 12 + 6
    np.testing.assert_allclose(data.features[1, previous_offset + 1], np.tanh(-3.0))
    assert not np.isclose(data.features[1, previous_offset + 1], 7.)


def test_finger_residual_dataset_masks_wrist_labels_and_history():
    trajectory = _trajectory()
    trajectory["latent"][..., :6] = 4.
    trajectory["latent"][..., 6:] = -2.
    reference_hand = np.zeros((10, 11, 3), np.float32)
    reference_obj = np.tile(np.eye(4, dtype=np.float32), (10, 1, 1))
    data = build_goal_finger_residual_dataset(
        trajectory, reference_hand, reference_obj,
        ["tracker", "tracker", "tracker", "nominal"], role="tracker")
    assert data.labels.shape == (3 * 9, 6)
    np.testing.assert_array_equal(data.labels[:, 0], -2.)
    previous_offset = 18 + 18 + 11 * 3 + 24 * 11 * 3 + 24 * 12 + 6
    np.testing.assert_array_equal(data.features[1, previous_offset:previous_offset + 6], 0.)
    np.testing.assert_allclose(data.features[1, previous_offset + 6], np.tanh(-2.))
