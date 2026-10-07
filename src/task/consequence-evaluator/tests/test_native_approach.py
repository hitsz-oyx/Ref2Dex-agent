"""Tiny CPU engineering checks for the per-object approach adapter."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[2]
sys.path[:0] = [str(ROOT), str(TASK/'src')]
from consequence_evaluator.native_approach import approach_gap
from src.task.CmResidual.dexplore_approach import ApproachConfig


def fixture(names=('airplane', 'cup')):
    states = torch.zeros(2, 13)
    states[:, 0] = torch.tensor([10., 20.])
    task = SimpleNamespace(object_name=list(names), object_id=torch.tensor([0, 1]),
                           data_id=torch.tensor([1, 0]), ball_size=2.,
                           _target_states=states, _dof_pos=torch.zeros(2, 18))
    agent = SimpleNamespace(grasp_link_reward_coef=0., min_grasp_links=0,
                            ppo_device='cpu', approach_config=ApproachConfig())
    return agent, task


def test_object_routing_preserves_stride_scale_and_native_batch_order():
    constructed = []
    class Bridge:
        def __init__(self, **kwargs):
            self.name = kwargs['object_urdf'].stem
            constructed.append(kwargs)
        def current(self, q, state):
            center = state[:, :3]
            pose = torch.eye(4).repeat(len(state), 1, 1)
            pose[:, :3, 3] = center
            hand = center[:, None, :].repeat(1, 17, 1)
            obj = hand.clone()
            hand[:, :, 0] += 2.
            obj[:, :, 0] += .5 if self.name == 'cup' else .25
            # An unsampled index must not change the original stride-8 gap.
            hand[:, 1, 0] = center[:, 0] + 1.
            return SimpleNamespace(object_pose=pose, object_points=obj, hand_points=hand)
    agent, task = fixture()
    gap = approach_gap(agent, task, lambda *args: pytest.fail('mixed task took airplane-only path'), Bridge)
    torch.testing.assert_close(gap, torch.tensor([1., 1.5]))
    assert {p['object_urdf'].name for p in constructed} == {'airplane.urdf', 'cup.urdf'}
    assert all(p['seed'] == 42 for p in constructed)
    again = approach_gap(agent, task, None, Bridge)
    torch.testing.assert_close(again, gap)
    assert len(constructed) == 2


def test_airplane_only_uses_unchanged_original_geometry():
    agent, task = fixture(('airplane',))
    original = lambda actual_agent, actual_task: torch.tensor([7., 8.])
    torch.testing.assert_close(approach_gap(agent, task, original), torch.tensor([7., 8.]))
    assert not hasattr(agent, '_ref2dex_object_approach_bridges')


def test_adapter_rejects_unimplemented_link_gate_instead_of_using_airplane_contact():
    agent, task = fixture()
    agent.min_grasp_links = 1
    with pytest.raises(ValueError, match='no-link-gate'):
        approach_gap(agent, task, None)
