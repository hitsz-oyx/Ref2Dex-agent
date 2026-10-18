import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wam_rl_policy.goal_scheduler import (
    OrderedGoalScheduler, SchedulerConfig, interpolate_pose, rigid_goal_delta,
)


POINTS = np.array([[.05, 0, 0], [-.05, 0, 0], [0, .05, 0],
                   [0, -.05, 0], [0, 0, .05], [0, 0, -.05]])


def path(positions):
    poses = np.repeat(np.eye(4)[None], len(positions), axis=0)
    poses[:, :3, 3] = positions
    return poses


def scheduler(poses, **kwargs):
    return OrderedGoalScheduler(poses, np.arange(len(poses))*.02, POINTS, **kwargs)


def test_static_preparation_keeps_future_intent_without_claiming_progress():
    positions = np.zeros((81, 3))
    positions[30:, 2] = np.linspace(0, .15, 51)
    ref = path(positions)
    s = scheduler(ref)
    for i in range(20):
        result = s.update(ref[0], i*.1, execution_ready=False)
        assert result['mode'] == 'PREPARE'
        assert result['confirmed_index'] == 0
        assert 0 < result['goal_pose'][2, 3] <= .011
        assert not result['complete']


def test_reference_clock_never_advances_a_delayed_object_and_timeout_is_explicit():
    ref = path(np.c_[np.linspace(0, .2, 101), np.zeros((101, 2))])
    s = scheduler(ref)
    for now in (0., 2., 7.):
        result = s.update(ref[0], now)
        assert result['confirmed_index'] == 0
        assert result['goal_position_m'] <= .021
    result = s.update(ref[0], 8.1)
    assert result['mode'] == 'TIMEOUT' and not result['complete']
    assert not result['target_on_reference']
    np.testing.assert_allclose(result['goal_pose'], ref[0])


def test_nominal_path_finishes_only_after_final_pose_settles():
    ref = path(np.c_[np.linspace(0, .1, 101), np.zeros((101, 2))])
    s = scheduler(ref)
    previous = 0
    for i, pose in enumerate(ref):
        result = s.update(pose, i*.03)
        assert result['confirmed_index'] >= previous
        previous = result['confirmed_index']
    # The final tolerance can be entered shortly before the last recorded pose.
    for i in range(1, 15):
        result = s.update(ref[-1], 3+i*.03)
    assert result['complete'] and result['confirmed_progress'] == 1


def test_out_and_back_cannot_skip_to_identical_final_pose():
    x = np.r_[np.linspace(0, .1, 51), np.linspace(.098, 0, 50)]
    ref = path(np.c_[x, np.zeros((len(x), 2))])
    s = scheduler(ref)
    for i in range(10):
        result = s.update(ref[-1], i*.02)
        assert result['confirmed_index'] == 0 and not result['complete']
        assert result['target_reference_index'] < 20
    for i, pose in enumerate(ref):
        result = s.update(pose, .3+i*.03)
    for i in range(1, 15):
        result = s.update(ref[-1], 3.3+i*.03)
    assert result['complete']


def test_pure_rotation_is_a_goal_and_encodes_same_point_identity():
    ref = path(np.zeros((101, 3)))
    ref[:, :3, :3] = Rotation.from_euler('z', np.linspace(0, np.pi, 101)).as_matrix()
    s = scheduler(ref)
    result = s.update(ref[0], 0)
    assert result['goal_position_m'] == 0 and result['goal_rotation_rad'] > 0
    assert result['mode'] == 'TRACK' and result['target_on_reference']
    assert result['goal_rotation_rad'] <= s.config.max_goal_rotation_rad+1e-8
    assert np.linalg.norm(result['goal_delta']) > 0
    expected = POINTS @ result['goal_pose'][:3, :3].T - POINTS
    np.testing.assert_allclose(result['goal_delta'], expected, atol=1e-12)
    for i, pose in enumerate(ref[1:], 1):
        result = s.update(pose, i*.03)
    for i in range(1, 15):
        result = s.update(ref[-1], 3+i*.03)
    assert result['complete']


def test_recovery_connector_is_bounded_and_does_not_claim_milestones():
    ref = path(np.c_[np.linspace(0, .2, 101), np.zeros((101, 2))])
    s = scheduler(ref)
    s.update(ref[0], 0)
    off_path = ref[0].copy()
    off_path[:3, 3] = [.3, .4, 0]
    result = s.update(off_path, .1, lost_control=True)
    assert result['mode'] == 'RECOVER' and not result['target_on_reference']
    assert result['confirmed_index'] == 0 and not result['complete']
    assert result['goal_position_m'] <= .05+1e-8
    assert result['goal_point_m'] <= .05+1e-8
    result = s.update(ref[0], .2)
    assert result['mode'] == 'TRACK'


def test_reset_and_separate_instances_keep_environment_memory_isolated():
    ref = path(np.c_[np.linspace(0, .1, 101), np.zeros((101, 2))])
    first, other = scheduler(ref), scheduler(ref)
    for i, pose in enumerate(ref[:60]):
        first.update(pose, i*.02)
    assert first.pointer > other.pointer
    first.reset()
    assert first.pointer == 0 and first.last_time is None
    assert first.update(ref[0], 0)['confirmed_index'] == 0


def test_mandatory_milestone_and_numpy_readiness_cannot_be_skipped():
    ref = path(np.c_[np.linspace(0, .1, 101), np.zeros((101, 2))])
    s = scheduler(ref, mandatory_indices=(10,))
    assert 10 in s.milestones
    s.update(ref[0], 0)
    result = s.update(ref[10], .1, execution_ready=np.bool_(False))
    assert result['mode'] == 'PREPARE' and result['confirmed_index'] == 0
    result = s.update(ref[10], .2, execution_ready=True)
    assert result['confirmed_index'] == 10
    with pytest.raises(ValueError):
        s.update(ref[10], .3, lost_control='false')


def test_goal_coordinates_are_in_current_anchor_and_leave_inputs_unchanged():
    current = np.eye(4)
    current[:3, :3] = Rotation.from_euler('z', .9).as_matrix()
    current[:3, 3] = [1, 2, 3]
    goal = current.copy()
    goal[:3, 3] += [.02, -.01, .01]
    before = current.copy()
    delta = rigid_goal_delta(current, goal, POINTS)
    np.testing.assert_allclose(delta, np.broadcast_to((goal[:3, 3]-current[:3, 3]) @ current[:3, :3], POINTS.shape))
    np.testing.assert_array_equal(current, before)
    halfway = interpolate_pose(np.eye(4), current, .5)
    np.testing.assert_allclose(halfway[:3, :3].T @ halfway[:3, :3], np.eye(3), atol=1e-12)


def test_bad_poses_clocks_and_collinear_points_are_rejected():
    ref = path(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        OrderedGoalScheduler(ref, [0, 0], POINTS)
    with pytest.raises(ValueError):
        OrderedGoalScheduler(ref, [0, 1], np.zeros((3, 3)))
    s = scheduler(ref)
    s.update(ref[0], 0)
    with pytest.raises(ValueError):
        s.update(ref[0], 0)
    bad = ref[0].copy()
    bad[0, 0] = -1
    with pytest.raises(ValueError):
        s.update(bad, 1)
    with pytest.raises(ValueError):
        SchedulerConfig(milestone_spacing_m=.001)
