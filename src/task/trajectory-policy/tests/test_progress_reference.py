import numpy as np

from trajectory_policy.robot_context import progress_goal_indices


def _pose(x, y, z):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = (x, y, z)
    return pose


def test_on_track_static_prefix_keeps_the_clock():
    still = [_pose(0, 0, 0) for _ in range(80)]
    lift = [_pose(0, 0, 0.05 * (i + 1)) for i in range(40)]
    reference = np.stack(still + lift)
    progress, indices = progress_goal_indices(reference[32], reference, 32, 0)
    assert progress == 32
    assert indices.tolist() == [40, 48, 56]


def test_behind_a_fast_lift_stays_on_the_last_unlifted_frame():
    still = [_pose(0, 0, 0) for _ in range(61)]
    lift = [_pose(0, 0, 0.05 * (i + 1)) for i in range(40)]
    reference = np.stack(still + lift)
    progress, indices = progress_goal_indices(reference[0], reference, 80, 60)
    assert progress == 60
    assert indices.tolist() == [60, 60, 60]


def test_behind_a_slow_horizontal_motion_uses_three_advances():
    reference = np.stack([_pose(0.015 * i, 0, 0) for i in range(80)])
    progress, indices = progress_goal_indices(reference[10], reference, 40, 10)
    assert progress == 10
    assert indices.tolist() == [11, 12, 13]


def test_cursor_does_not_move_backward():
    reference = np.stack([_pose(0.01 * i, 0, 0) for i in range(80)])
    progress, _ = progress_goal_indices(reference[0], reference, 40, 25)
    assert progress == 25


def test_recorded_reference_keeps_clock_on_track_and_clamps_the_lift_when_behind():
    reference = np.load(
        'outputs/trajectory-policy/competent-reference-20261011-r1/reference.npz')['obj']
    progress, indices = progress_goal_indices(reference[32], reference, 32, 0)
    assert progress == 32
    assert indices.tolist() == [40, 48, 56]
    progress, indices = progress_goal_indices(reference[48], reference, 48, 32)
    assert indices.tolist() == [56, 64, 72]
    progress, indices = progress_goal_indices(reference[0], reference, 64, 58)
    assert progress == 61
    assert indices.tolist() == [62, 62, 62]
    height = reference[indices, 2, 3] - reference[progress, 2, 3]
    assert np.all(height <= 0.03)
