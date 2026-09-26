from __future__ import annotations

import pytest
import torch

from src.task.CmResidual.paired_sim_step import paired_sim_step


class _Gym:
    def set_actor_root_state_tensor(self, sim, root):
        assert sim is not None and root is not None

    def set_dof_state_tensor(self, sim, dof):
        assert sim is not None and dof is not None


class _Task:
    def __init__(self, hidden_drift=0.0, rigid_restorable=True):
        self.gym = _Gym()
        self.sim = object()
        self._root_states = torch.zeros(2, 3, 13)
        self._root_states[:, :, 6] = 1.0
        self._target_states = self._root_states[:, 2]
        self._dof_state = torch.zeros(2, 18, 2)
        self._dof_pos = self._dof_state[..., 0]
        self._rigid_body_state = torch.zeros(2, 19, 13)
        self.hidden_drift = hidden_drift
        self.rigid_restorable = rigid_restorable
        self.call_count = 0
        self.action = None

    def pre_physics_step(self, action):
        self.action = action.clone()

    def _physics_step(self):
        self.call_count += 1
        self._dof_pos += self.action * .1
        self._target_states[:, 2] += self.action[:, 2] * .01 + self.call_count * self.hidden_drift
        self._rigid_body_state[:, 0, 0] += self.action[:, 0] + 0.01

    def _refresh_sim_tensors(self):
        if self.rigid_restorable and self.action is not None:
            self._rigid_body_state[:, 0, 0] = self._dof_pos[:, 0]


def test_paired_step_replays_and_restores_full_physics_state():
    task = _Task()
    base = torch.zeros(2, 18)
    alternate = base.clone()
    alternate[:, 2] = .2
    before_root = task._root_states.clone()
    before_dof = task._dof_state.clone()
    result = paired_sim_step(task, base, alternate, lambda value: value)
    torch.testing.assert_close(task._root_states, before_root)
    torch.testing.assert_close(task._dof_state, before_dof)
    torch.testing.assert_close(result["same_action_repeat_object_mm"], torch.zeros(2))
    torch.testing.assert_close(result["actual_effect_mm"], torch.full((2,), 2.0), atol=1e-4, rtol=0)
    assert task.call_count == 3


def test_paired_step_rejects_hidden_unrestored_solver_drift():
    task = _Task(hidden_drift=.0002)
    base = torch.zeros(2, 18)
    alternate = base.clone()
    alternate[:, 2] = .2
    with pytest.raises(RuntimeError, match="same-action physics replay"):
        paired_sim_step(task, base, alternate, lambda value: value)
    assert torch.equal(task._root_states[:, 2, 2], torch.zeros(2))


def test_paired_step_rejects_unrestored_rigid_body_state():
    task = _Task(rigid_restorable=False)
    base = torch.zeros(2, 18)
    alternate = base.clone()
    alternate[:, 2] = .2
    with pytest.raises(RuntimeError, match="physics state restore failed"):
        paired_sim_step(task, base, alternate, lambda value: value)
