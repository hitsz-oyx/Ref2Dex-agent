import numpy as np
import pytest

from trajectory_policy.wam_shadow import (NATIVE_TO_WAM, active_hand_tau,
                                          robot_goal_sample)


def test_hand_selection_reorders_named_points_and_rejects_missing_hand():
    prediction = np.arange(3 * 24 * 2 * 11 * 3).reshape(3, 24, 2, 11, 3).astype(float)
    valid = np.zeros((3, 2, 11), bool)
    valid[:, 0] = True
    result = active_hand_tau(prediction, valid)
    assert result.shape == (3, 24, 33)
    assert np.array_equal(result.reshape(3, 24, 11, 3)[:, :, NATIVE_TO_WAM], prediction[:, :, 0])
    with pytest.raises(ValueError):
        active_hand_tau(prediction, valid, 'left')
    valid[0, 0, 5] = False
    with pytest.raises(ValueError):
        active_hand_tau(prediction, valid)


def test_robot_goal_scene_is_invariant_to_global_rigid_coordinate_change():
    rng = np.random.default_rng(5)
    canonical = dict(points=rng.normal(size=(512, 3)).astype(np.float32) * .03,
                     normals=np.zeros((512, 3), np.float32), center=np.zeros(3), radius=.05)
    poses = np.tile(np.eye(4), (4, 1, 1))
    poses[:, :3, 3] = np.arange(4)[:, None] * [.001, .002, .003]
    hand = rng.normal(size=(4, 11, 3)) * .1
    goal = poses[-1].copy()
    goal[:3, 3] += [.05, .01, -.02]
    original = robot_goal_sample(poses, hand, goal, canonical)
    global_transform = np.eye(4)
    global_transform[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    global_transform[:3, 3] = [1., 2., 3.]
    shifted = robot_goal_sample(global_transform @ poses,
        hand @ global_transform[:3, :3].T + global_transform[:3, 3],
        global_transform @ goal, canonical)
    for key in ('xyz', 'features', 'start_hand', 'start_hand_history',
                'start_object_points', 'goal_object_points'):
        assert np.allclose(original[key], shifted[key], atol=2e-6), key
    assert original['start_hand_valid'][0].all()
    assert not original['start_hand_valid'][1].any()
    assert not original['action_valid'].any()  # no future measured hand input
