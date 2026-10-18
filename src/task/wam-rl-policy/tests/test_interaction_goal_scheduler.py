"""Regressions for failure modes hidden by exact, 60-Hz path replay."""
import numpy as np

from wam_rl_policy.goal_scheduler import OrderedGoalScheduler


POINTS = np.array([[.05, 0, 0], [-.05, 0, 0], [0, .05, 0],
                   [0, -.05, 0], [0, 0, .05], [0, 0, -.05]])


def reference():
    poses = np.repeat(np.eye(4)[None], 501, axis=0)
    poses[:, 0, 3] = np.linspace(0, .1, len(poses))
    return poses


def make():
    ref = reference()
    return OrderedGoalScheduler(ref, np.arange(len(ref))*.01, POINTS), ref


def hand(pose, near):
    values = np.tile([0., -.2, .1], (11, 1))
    if near:
        values[6] = [.05, 0, .001]
        values[7] = [-.05, 0, .001]
    return values + pose[:3, 3]


def test_continuous_slow_progress_is_not_stalling():
    scheduler, ref = make()
    for i in range(100):
        pose = ref[0].copy()
        pose[0, 3] = i*.0001
        row = scheduler.update(pose, i*.1)
        assert not row['timed_out']


def test_low_frequency_observation_can_witness_crossed_ordered_milestones():
    scheduler, ref = make()
    scheduler.update(ref[0], 0)
    row = scheduler.update(ref[150], .15)
    assert row['confirmed_index'] > 0
    for i, index in enumerate((300, 450, 500), 2):
        row = scheduler.update(ref[index], i*.15)
    for i in range(5, 9):
        row = scheduler.update(ref[-1], i*.15)
    assert row['complete']


def test_target_does_not_jump_a_whole_milestone_when_pointer_changes():
    scheduler, ref = make()
    last = None
    for i, index in enumerate(range(0, 401, 5)):
        row = scheduler.update(ref[index], i/60)
        if last is not None:
            assert np.linalg.norm(row['goal_pose'][:3, 3]-last[:3, 3]) < .004
        last = row['goal_pose']


def test_same_object_state_receives_different_preparation_goals_from_hand_history():
    far, ref = make()
    near, _ = make()
    for i in range(20):
        a = far.update(ref[0], i/60, hand_points=hand(ref[0], False))
        b = near.update(ref[0], i/60, hand_points=hand(ref[0], True))
    assert a['interaction']['state'] == 'APPROACH'
    assert b['interaction']['state'] == 'CONTACT_CANDIDATE'
    assert a['goal_point_m'] < b['goal_point_m']
    # Stationary proximity is not evidence of stable object control.
    assert not b['interaction']['coupled']


def test_motion_coupling_and_lost_control_come_from_observed_hand_object_history():
    scheduler, ref = make()
    for i in range(30):
        pose = ref[0].copy()
        pose[0, 3] = i*.0005
        row = scheduler.update(pose, i/60, hand_points=hand(pose, True))
    assert row['interaction']['coupled']
    confirmed = row['confirmed_index']
    for i in range(30, 45):
        row = scheduler.update(pose, i/60, hand_points=hand(pose, False))
    assert row['interaction']['state'] == 'LOST'
    assert row['mode'] == 'RECOVER'
    assert row['confirmed_index'] == confirmed


def test_moving_object_beside_a_stationary_hand_is_not_motion_coupling():
    scheduler, ref = make()
    fixed_hand = hand(ref[0], True)
    for i in range(25):
        pose = ref[0].copy()
        pose[0, 3] = i*.0003
        row = scheduler.update(pose, i/60, hand_points=fixed_hand)
    assert not row['interaction']['coupled']


def test_weak_hand_proxy_does_not_erase_observed_ordered_path_progress():
    scheduler, ref = make()
    for i, index in enumerate(range(0, 201, 5)):
        pose = ref[index]
        row = scheduler.update(pose, i/60, hand_points=hand(pose, False))
    assert row['interaction']['state'] == 'APPROACH'
    assert not row['interaction']['coupled']
    assert row['confirmed_progress'] > .1
