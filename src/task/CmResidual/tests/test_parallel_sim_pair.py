from __future__ import annotations

import pytest
import torch

from src.task.CmResidual.parallel_sim_pair import parallel_sim_pair_step


class _Task:
    def __init__(self):
        self.num_dof = 18
        self._root_states = torch.zeros(3, 3, 13)
        self._dof_state = torch.zeros(3, 18, 2)
        self._dof_pos = self._dof_state[..., 0]
        self._rigid_body_state = torch.zeros(3, 19, 13)
        self._target_states = self._root_states[:, 2]
        self.progress_buf = torch.full((3,), 80)
        self.data_id = torch.zeros(3, dtype=torch.long)
        self.start_times = torch.zeros(3, dtype=torch.long)
        self.ref_index = torch.zeros(3, dtype=torch.long)
        self._contact_forces = torch.ones(3, 19, 3)
        self._contact_body_ids = torch.tensor([0])
        self._tar_contact_forces = torch.ones(3, 3)

    def step(self, action):
        self._dof_pos += action * .1
        self._target_states[:, 2] += action[:, 2] * .01
        return "environment output"


def test_parallel_pair_measures_physical_action_effect():
    task = _Task()
    output, record = parallel_sim_pair_step(task, torch.zeros(3, 18), task.step)
    assert output == "environment output"
    torch.testing.assert_close(record["actual_effect_mm"], torch.tensor([1.]))
    torch.testing.assert_close(record["same_action_repeat_object_mm"], torch.zeros(1))
    assert record["pre_contact"].item()


def test_parallel_pair_rejects_prestate_mismatch():
    task = _Task()
    task._rigid_body_state[2, 0, 0] = .002
    with pytest.raises(RuntimeError, match="prestates differ"):
        parallel_sim_pair_step(task, torch.zeros(3, 18), task.step)


def test_parallel_pair_rejects_same_action_hidden_drift():
    task = _Task()

    def step(action):
        task.step(action)
        task._target_states[1, 2] += .0002

    with pytest.raises(RuntimeError, match="same-action parallel replay"):
        parallel_sim_pair_step(task, torch.zeros(3, 18), step)
