import numpy as np
import pytest

from wam_rl_policy.tau_contract import v4_trajectory_to_student


def _constant_velocity(speed=.12):
    traj = np.zeros((24, 2, 11, 3), np.float64)
    traj[:, 0, :, 0] = speed * (np.arange(24) + 1)[:, None] / 30.
    return traj


def test_student_clock_is_not_the_first_eight_v4_rows():
    tau = v4_trajectory_to_student(_constant_velocity(.12), object_rotation=np.eye(3),
                                   frame="world")
    assert tau.shape == (8, 11, 3)
    np.testing.assert_allclose(tau[-1, :, 0], .12 * 8 / 60, atol=1e-6)
    # The old `trajectory[:, :8]` would have produced 32 mm for this motion.
    wrong = _constant_velocity(.12)[:8, 0, :, 0]
    np.testing.assert_allclose(wrong[-1], .032, atol=1e-6)
    assert tau[0, 0, 0] > 0  # half of v4 step 0


def test_tau_starts_at_zero_and_interpolates_monotonically():
    tau = v4_trajectory_to_student(_constant_velocity(.06), object_rotation=np.eye(3),
                                   frame="world")
    assert np.all(np.diff(tau[:, 0, 0]) > 0)


def test_anchor_to_world_rotation_and_slot_selection():
    traj = np.zeros((24, 2, 11, 3), np.float64)
    traj[:, 0, :, 0] = 1.
    traj[:, 1, :, 1] = 1.
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    right = v4_trajectory_to_student(traj, object_rotation=rotation, hand_slot="right")
    left = v4_trajectory_to_student(traj, object_rotation=rotation, hand_slot="left")
    np.testing.assert_allclose(right[0, ..., 1], .5, atol=1e-6)
    np.testing.assert_allclose(right[1:, ..., 1], 1., atol=1e-6)
    np.testing.assert_allclose(right[..., 0], 0., atol=1e-6)
    np.testing.assert_allclose(left[0, ..., 0], -.5, atol=1e-6)
    np.testing.assert_allclose(left[1:, ..., 0], -1., atol=1e-6)
    np.testing.assert_allclose(left[..., 1], 0., atol=1e-6)


def test_world_conversion_requires_a_proper_rotation():
    with pytest.raises(ValueError, match="current object rotation"):
        v4_trajectory_to_student(_constant_velocity(), frame="world")
    with pytest.raises(ValueError, match="proper finite"):
        v4_trajectory_to_student(_constant_velocity(), object_rotation=np.eye(3) * 2,
                                 frame="world")
