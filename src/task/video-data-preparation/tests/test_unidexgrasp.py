"""Physical frame and outcome semantics of the ShadowHand/CAD source."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from video_data_preparation.unidexgrasp_data import forecasting_sample

TASK = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('unidex_provider_audit',
    TASK/'tools/audit/audit_unidexgrasp_provider.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def translating_trajectory():
    clock = np.arange(100)/30
    obj = np.zeros((100, 7))
    obj[:, 0] = .3*clock
    obj[:, 5:] = np.sqrt(.5)  # Constant90deg world rotation aroundZ.
    hand = np.zeros((100, 11, 3))
    hand[..., 0] = .02 + .6*clock[:, None]
    surface = np.column_stack((np.linspace(0, .1, 32), np.zeros((32, 2)))).astype(np.float32)
    return dict(time=clock, object=obj, hand=hand), dict(
        surface=surface, normals=np.tile([1., 0., 0.], (32, 1)).astype(np.float32))


def test_actual_history_and_future_share_one_current_object_frame():
    trajectory, geometry = translating_trajectory()
    sample = forecasting_sample(trajectory, geometry, 27, 0)
    # The current anchor stays fixed across history and future. Hand speed is
    # .6m/s in world, not the .3m/s relative speed of per-frame moving anchors.
    np.testing.assert_allclose(sample['start_hand'][0, :, 1], -.29, atol=1e-6)
    np.testing.assert_allclose(sample['start_hand_history'][0, :, 1], -.6, atol=1e-6)
    np.testing.assert_allclose(sample['features'][:, 4], -.3, atol=1e-6)
    np.testing.assert_allclose(sample['features'][:, 6:9], 0, atol=1e-6)
    np.testing.assert_allclose(sample['features'][:, :3], geometry['normals'])
    np.testing.assert_allclose(sample['trajectory_target'][-1, 0, :, 1], -.48, atol=1e-6)
    np.testing.assert_allclose(sample['scene_target'][0, -1, :, 1], -.24, atol=1e-6)
    np.testing.assert_allclose(sample['goal_object_points']-sample['start_object_points'],
                               sample['scene_target'][0, -1], atol=1e-6)
    assert not sample['start_hand_valid'][1].any()
    assert not sample['trajectory_valid'][:, 1].any()


def test_future_intervention_changes_labels_and_task_goal_but_no_observations():
    trajectory, geometry = translating_trajectory()
    before = forecasting_sample(trajectory, geometry, 27, 0)
    trajectory['object'][28:, :3] += 2
    trajectory['hand'][28:] += 4
    after = forecasting_sample(trajectory, geometry, 27, 0)
    for key in ('xyz', 'features', 'start_hand', 'start_hand_history',
                'scene_points', 'start_object_points', 'anchor_world_to_current_object'):
        np.testing.assert_array_equal(before[key], after[key])
    for key in ('trajectory_target', 'scene_target', 'goal_object_points'):
        assert not np.array_equal(before[key], after[key])
    assert not bool(after['training_allowed'])


def test_official_hit_endpoint_and_half_second_retention_are_separate():
    trajectory, _ = translating_trajectory()
    trajectory['goal'] = trajectory['object'][:, :3].copy()
    trajectory['time'] += 1/60  # Official reset's zero-action warm step.
    trajectory['terminal_time'] = 200/60
    row = dict(path='trajectory.npz', object_code='test', split='train',
               official_success=True, endpoint_within_5cm=True)
    retained = audit.proximity_record(trajectory, row)
    assert retained['sampled_tail_half_second_within_5cm']
    assert retained['sampled_tail_duration_s'] == pytest.approx(.5)
    assert retained['official_endpoint_time_s']-retained['sampled_endpoint_time_s'] == pytest.approx(1/60)
    # A briefly reached endpoint passes endpoint proximity but fails retention.
    trajectory['goal'][-2] += .1
    brief = audit.proximity_record(trajectory, row)
    assert brief['sampled_endpoint_within_5cm']
    assert not brief['sampled_tail_half_second_within_5cm']
    # Cumulative official success remains true after the sampled object falls.
    trajectory['goal'][-1] += .1
    dropped = audit.proximity_record(trajectory, row)
    assert dropped['official_hit_then_sampled_endpoint_outside']
    assert not dropped['sampled_endpoint_within_5cm']
