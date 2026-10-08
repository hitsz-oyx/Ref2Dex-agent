"""Tests for ref4_1's meaningful causal and temporal alignment guarantees."""
from pathlib import Path
import sys
import numpy as np
import pytest

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / 'src'))
from consequence_evaluator.reference_progress import (AlignmentConfig, ReferenceProgress,
    trajectory_features, track_cost, label_window, preference)
from consequence_evaluator.xirl_alignment import trailing_context_indices


def trajectory(length=120):
    poses = np.tile(np.eye(4), (length, 1, 1))
    phase = np.linspace(0, 2 * np.pi, length)
    poses[:, 0, 3] = .05 * np.sin(phase)
    poses[:, 2, 3] = .8 + .05 * (1 - np.cos(phase))
    local = np.tile(np.linspace(.02, .08, 33).reshape(1, 11, 3), (length, 1, 1))
    local[:, :, 0] += .01 * np.sin(phase)[:, None]
    points = local + poses[:, None, :3, 3]
    return poses, points, np.arange(length) / 30


def test_future_changes_cannot_change_a_label_or_its_prefix_alignment():
    poses, points, clock = trajectory()
    reference = trajectory_features(poses, points, clock)
    matcher = ReferenceProgress(reference)
    full = matcher.align(reference)
    tick = 40
    end = tick + 24
    # Include a violently different later failure/recovery; an episode-wide
    # normalization, centered velocity, OT or smoother would violate this.
    changed_pose = poses.copy(); changed_hand = points.copy()
    changed_pose[end + 1:, :3, 3] += 10
    changed_hand[end + 1:] -= 20
    changed = matcher.align(trajectory_features(changed_pose, changed_hand, clock))
    prefix = matcher.align(trajectory_features(poses[:end + 1], points[:end + 1], clock[:end + 1]))
    assert np.allclose(full['distribution'][:end + 1], changed['distribution'][:end + 1], atol=1e-10)
    assert np.allclose(full['distribution'][:end + 1], prefix['distribution'], atol=1e-10)
    assert label_window(full, tick)['value'] == pytest.approx(label_window(changed, tick)['value'], abs=1e-10)


def test_repeated_start_end_state_keeps_late_reference_phase():
    data = trajectory()
    features = trajectory_features(*data)
    trace = ReferenceProgress(features).align(features)
    assert np.allclose(data[0][0], data[0][-1])
    assert trace['progress'][-1] > .95
    assert trace['progress'][0] == 0
    assert np.min(np.diff(trace['progress'])) > -.01


def test_phase_can_stagnate_and_regress_without_distant_posterior_jump():
    reference = np.arange(100)[None, :]
    path = np.r_[np.arange(50), np.full(20, 49), np.arange(48, 19, -1)]
    trace = track_cost((path[:, None] - reference) ** 2 * .02)
    assert abs(trace['progress'][69] - trace['progress'][49]) < .02
    assert trace['progress'][-1] < trace['progress'][49] - .2
    assert np.max(np.abs(np.diff(trace['progress']))) <= 4 / 99 + 1e-10
    cost = np.ones((60, 100))
    cost[:, 0] = 0
    cost[30:, 99] = -0.0
    cost[30:, :99] = 1000
    jumped = track_cost(cost)
    assert np.max(np.abs(np.diff(jumped['progress']))) <= 4 / 99 + 1e-10


def test_stationary_history_does_not_become_episode_clock():
    features = trajectory_features(*trajectory())
    stationary = np.repeat(features[:1], 120, axis=0)
    trace = ReferenceProgress(features).align(stationary)
    assert trace['progress'][-1] < .05
    assert label_window(trace, 80)['value'] < .005


def test_features_include_object_frame_interaction_and_backward_motion():
    poses, points, clock = trajectory()
    base = trajectory_features(poses, points, clock)
    moved = points.copy(); moved[50:, 0, 1] += .03
    changed = trajectory_features(poses, moved, clock)
    assert base.shape == (120, 90)
    assert np.array_equal(base[:50], changed[:50])
    assert not np.array_equal(base[50], changed[50])
    # Same common rigid translation must not alter any feature.
    shifted = poses.copy(); shifted[:, :3, 3] += [1, 2, 3]
    assert np.allclose(base, trajectory_features(shifted, points + [1, 2, 3], clock))
    assert np.allclose(base[0, 45:], 0)


def test_context_is_causal_and_ties_abstain():
    indices = trailing_context_indices(12, 8)
    assert np.array_equal(indices[7], np.arange(8))
    assert np.all(indices <= np.arange(12)[:, None])
    assert preference(.03, .02, .02) == 0
    assert preference(.04, -.01, .02) == 1
    assert preference(-.01, .04, .02) == -1
    with pytest.raises(ValueError):
        preference(np.nan, 0)
