"""Review boundaries: goal identity, ordered lead and physical coordinates."""
import numpy as np
from scipy.spatial.transform import Rotation

from wam_rl_policy.goal_scheduler import OrderedGoalScheduler
from wam_rl_policy.goal_review import prediction_in_world, review_goals


def path():
    poses = np.repeat(np.eye(4)[None], 121, 0)
    poses[:, 0, 3] = np.linspace(0, .12, 121)
    points = np.array([[.01, 0, 0], [0, .01, 0], [-.01, 0, 0], [0, -.01, 0]])
    return OrderedGoalScheduler(poses, np.arange(121)/60, points)


def test_review_keeps_selected_goal_exact_and_cannot_skip_checkpoint():
    scheduler = path()
    for tick in range(8):
        actual = scheduler.poses[tick]
        decision = scheduler.update(actual, tick/60)
    goals = review_goals(scheduler, actual, decision)
    np.testing.assert_array_equal(goals[1], decision['goal_pose'])
    assert goals[0, 0, 3] <= goals[1, 0, 3] <= goals[2, 0, 3]
    pending = scheduler.milestones[scheduler.pointer+1]
    assert goals[2, 0, 3] <= scheduler.poses[pending, 0, 3]
    assert all(scheduler._within_budget(actual, g) for g in goals)


def test_recovery_review_does_not_label_connector_as_a_path_goal():
    scheduler = path()
    scheduler.update(scheduler.poses[0], 0.)
    actual = np.eye(4)
    actual[1, 3] = .2
    decision = scheduler.update(actual, .1)
    assert not decision['target_on_reference']
    goals = review_goals(scheduler, actual, decision)
    assert all(scheduler._within_budget(actual, g) for g in goals)
    assert goals[0, 1, 3] > goals[1, 1, 3] > 0
    np.testing.assert_array_equal(goals[1], decision['goal_pose'])


def test_prediction_is_rotated_displacement_added_to_current_points():
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    pose[:3, 3] = [2., 3., 4.]
    hand = np.tile([1., 2., 3.], (11, 1))
    flow = np.tile([.02, 0, 0], (32, 1))
    trajectory = np.zeros((24, 2, 11, 3))
    trajectory[:, 0, :, 0] = .1
    scene = np.zeros((1, 24, 32, 3))
    scene[..., 0] = .03
    predicted_hand, predicted_scene = prediction_in_world(pose, hand, flow, trajectory, scene)
    np.testing.assert_array_equal(predicted_hand[0], hand)
    np.testing.assert_allclose(predicted_hand[-1], hand+[0, .1, 0], atol=1e-12)
    np.testing.assert_allclose(predicted_scene[0, 0], [2., 3.02, 4.], atol=1e-12)
    np.testing.assert_allclose(predicted_scene[-1, 0], [2., 3.05, 4.], atol=1e-12)
