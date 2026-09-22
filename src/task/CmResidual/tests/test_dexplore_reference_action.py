from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MODULE = (Path(__file__).resolve().parents[4] / "third_party" / "DExplore" /
          "dexplore" / "utils" / "reference_action.py")
SPEC = importlib.util.spec_from_file_location("dexplore_reference_action", MODULE)
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


def _task(frames=3):
    refs = torch.zeros(1, 1, frames, 137)
    return SimpleNamespace(
        num_envs=1, device="cpu", num_dof=18,
        data_id=torch.tensor([0]), ref_index=torch.tensor([0]),
        progress_buf=torch.tensor([1]), max_episode_length=torch.tensor([frames]),
        hoi_refs=refs, _dof_pos=torch.zeros(1, 18),
        _pd_action_offset=torch.zeros(18), _pd_action_scale=torch.ones(18),
    )


def test_inspire_reference_action_maps_incremental_wrist_and_absolute_fingers():
    task = _task()
    task.hoi_refs[0, 0, 2, 119:125] = torch.tensor(
        [0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
    task.hoi_refs[0, 0, 2, 125:137] = 0.75
    action = REFERENCE.inspire_reference_action(task, lead=1)
    torch.testing.assert_close(action[0, :6], task.hoi_refs[0, 0, 2, 119:125])
    torch.testing.assert_close(action[0, 6:], torch.full((12,), 0.5))


def test_reference_action_clamps_frame_and_action():
    task = _task(frames=2)
    task.hoi_refs[0, 0, 1, 119:137] = 10
    assert torch.equal(REFERENCE.inspire_reference_action(task, lead=5), torch.ones(1, 18))
    with pytest.raises(ValueError, match="nonnegative"):
        REFERENCE.inspire_reference_action(task, lead=-1)
