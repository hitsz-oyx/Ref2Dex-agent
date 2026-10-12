"""Catch future-observation leakage and fabricated labels in native auxiliary data."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

MODULE = Path(__file__).resolve().parents[1]/'src/goal_point_wam/egodex_auxiliary.py'
spec = importlib.util.spec_from_file_location('ego_aux_contract', MODULE)
aux = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aux)


def example():
    t = np.arange(40, dtype='float32')[:, None, None, None]
    hand = np.zeros((40, 2, 11, 3), 'float32')
    hand[..., 0] = t[..., 0]*.002
    hand[..., 1] = np.arange(11)[None, None]*.003
    camera = np.broadcast_to(np.eye(4, dtype='float32'), (40, 4, 4)).copy()
    camera[:, 0, 3] = np.arange(40)*.001
    valid = np.ones((40, 2, 11), bool)
    valid[:, 1] = False
    valid[:, 0, 10] = False
    dataset = object.__new__(aux.EgoDexActorWindows)
    dataset.rows = np.asarray([[0, 8]], 'int32')
    d = dict(hand=hand, camera=camera, valid=valid)
    dataset.record = lambda _:d
    return dataset, d


def test_future_hand_and_camera_never_enter_observation():
    dataset, d = example()
    before = dataset[0]
    d['hand'][9:] += .05
    d['camera'][9:, :3, 3] += .2
    after = dataset[0]
    for key in ['xyz', 'features', 'start_hand', 'start_hand_history', 'start_hand_valid',
                'start_object_points', 'goal_object_points', 'object_goal_valid']:
        np.testing.assert_array_equal(before[key], after[key])
    assert not np.array_equal(before['trajectory_target'], after['trajectory_target'])


def test_fixed_current_camera_and_masked_missing_points():
    dataset, d = example()
    sample = dataset[0]
    np.testing.assert_allclose(sample['start_hand'][0, 0], [.008, 0, 0], atol=1e-7)
    np.testing.assert_allclose(sample['trajectory_target'][0, 0, 0], [.002, 0, 0], atol=1e-7)
    assert not sample['trajectory_valid'][:, 1].any()
    assert not sample['trajectory_valid'][:, 0, 10].any()
    assert not sample['object_goal_valid'].any()
    batch = aux.ego_collate([sample])
    assert not batch['object_valid'].any()
    assert (batch['scene_object'] == -1).all()


def test_auxiliary_bypasses_world_and_action_branches():
    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))

        def scene(self, batch):
            return dict(scene_tokens=batch['xyz'], scene_mask=batch['point_valid'])

        def goal(self, start, goal, valid):
            assert not valid.any()
            return start, valid

        def trajectory(self, *args):
            return self.weight*torch.ones((1, 24, 2, 11, 3))

        def observed_hand_condition(self, batch):
            return batch['start_hand']

        def world(self, *args):
            raise AssertionError('no native world labels: branch must not run')

        def action(self, *args):
            raise AssertionError('actor must not consume future action labels')

        def trajectory_loss(self, pred, batch):
            return (pred[batch['trajectory_valid']]**2).mean()

    dataset, _ = example()
    batch = aux.ego_collate([dataset[0]])
    model = aux.auxiliary_model_class(Base)()
    loss, terms = model.loss(model(batch), batch)
    loss.backward()
    assert torch.isfinite(model.weight.grad) and model.weight.grad != 0
    assert terms['world'] == 0
