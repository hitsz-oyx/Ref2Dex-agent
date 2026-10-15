import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ptv3_shared_wam_v2 import paths as _paths
from ptv3_shared_wam_v2.robot_workspace import observed_workspace_mask, restrict_observed_workspace
from trajectory_policy.wam_shadow import robot_goal_sample
from oakink_wm.pointworld import fixed_workspace_grid


def fixture():
    pose = np.broadcast_to(np.eye(4, dtype=np.float32), (50, 4, 4)).copy()
    hand = np.zeros((50, 11, 3), np.float32)
    hand[1, :, 0] = -2.01
    hand[17, :, 1] = -3
    canonical = dict(points=np.zeros((512, 3), np.float32),
                     normals=np.tile([1., 0., 0.], (512, 1)).astype(np.float32),
                     radius=.1, center=np.zeros(3, np.float32))
    return dict(hand=hand, obj=pose), canonical


def test_observed_workspace_reproduces_real_sample_grid_rejection():
    data, canonical = fixture()
    ticks = np.array([0, 1, 16, 17])
    mask = observed_workspace_mask(data, canonical, ticks, [-2, -2, -2], .01)
    assert mask.tolist() == [True, False, True, False]
    for tick, accepted in zip(ticks, mask):
        past = np.maximum(tick + np.arange(-3, 1), 0)
        sample = robot_goal_sample(data['obj'][past], data['hand'][past], data['obj'][tick], canonical)
        xyz = torch.tensor(sample['xyz'][None])
        valid = torch.ones(xyz.shape[:2], dtype=torch.bool)
        if accepted:
            fixed_workspace_grid(xyz, valid)
        else:
            with pytest.raises(ValueError, match='outside fixed PTv3 workspace'):
                fixed_workspace_grid(xyz, valid)


def test_admission_is_current_object_frame_and_does_not_use_future():
    data, canonical = fixture()
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    data['obj'][:, :3, :3] = rotation
    data['obj'][:, :3, 3] = [4, 5, 6]
    data['hand'][:] = [4, 5, 6]
    before = observed_workspace_mask(data, canonical, [0], [-2]*3, .01)
    data['hand'][1:] = 9999
    after = observed_workspace_mask(data, canonical, [0], [-2]*3, .01)
    assert before.tolist() == after.tolist() == [True]


def test_subset_rebuilds_sampling_and_normalization_ticks_without_mutating_records():
    data, canonical = fixture()
    records = [dict(sequence='train-a', admitted_ticks=[0, 1, 16, 17]),
               dict(sequence='train-b', admitted_ticks=[0, 1, 16, 17])]
    original = copy.deepcopy(records)
    dataset = SimpleNamespace(records=records, ticks_by_sequence=[np.array([0, 1, 16, 17])]*2,
                              keys=[(tick, seq) for seq in range(2) for tick in [0, 1, 16, 17]],
                              sequence=lambda _: (data, canonical))
    report = restrict_observed_workspace(dataset, [-2]*3, .01)
    assert records == original
    assert dataset.keys == [(0, 0), (16, 0), (0, 1), (16, 1)]
    assert [v.tolist() for v in dataset.ticks_by_sequence] == [[0, 16], [0, 16]]
    assert [v.tolist() for v in dataset.pools] == [[0, 1], [2, 3]]
    assert [[v.tolist() for v in phase] for phase in dataset.phase_pools] == [[[0], [2]], [[1], [3]]]
    assert report['excluded_windows'] == 4
    assert report['records'][0]['excluded_ticks'] == [1, 17]
    assert report['records'][1]['retained_original_indices'] == [4, 6]
    assert report['records'][1]['retained_start_index'] == 2


def test_no_windows_and_invalid_scene_are_rejected():
    data, canonical = fixture()
    canonical['points'][0, 0] = -3
    assert not observed_workspace_mask(data, canonical, [0, 16], [-2]*3, .01).any()
    dataset = SimpleNamespace(records=[dict(sequence='a')], keys=[(0, 0)],
                              ticks_by_sequence=[np.array([0])], sequence=lambda _: (data, canonical))
    with pytest.raises(ValueError, match='no windows'):
        restrict_observed_workspace(dataset, [-2]*3, .01)
