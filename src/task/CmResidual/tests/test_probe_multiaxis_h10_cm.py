from __future__ import annotations

import torch

from src.task.CmResidual.tools.probe_multiaxis_h10_cm import (
    no_action_data, stratified_schedule,
)


def test_six_cell_schedule_and_state_only_action_mask():
    assignment = torch.tensor([1, -1, 2, -2, 3, -3] * 20)
    schedule = stratified_schedule(assignment, steps=3, batch_size=96, seed=42)
    assert len(schedule) == 3
    for ids in schedule:
        assert len(ids) == 96
        assert all(int((assignment[ids] == code).sum()) == 16
                   for code in (1, -1, 2, -2, 3, -3))
    raw = torch.randn(4, 67)
    result = no_action_data({"raw": raw})
    assert (result["raw"][:, 18:36] == 0).all()
    torch.testing.assert_close(result["raw"][:, :18], raw[:, :18])
    torch.testing.assert_close(result["raw"][:, 36:], raw[:, 36:])
    assert not torch.equal(result["raw"], raw)
