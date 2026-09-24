import torch

from src.task.CmResidual.tools.probe_randomized_geometric_cm import (
    RawActionEffect, raw_inputs,
)


def test_raw_action_effect_uses_relative_root_and_predicts_three_deltas():
    rows = {"q": torch.zeros(2, 18), "action": torch.zeros(2, 18),
            "dof_vel": torch.zeros(2, 18), "object_state": torch.zeros(2, 13)}
    rows["q"][:, :3] = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    rows["object_state"][:, :3] = torch.tensor([[.5, .5, .5], [1., 1., 1.]])
    raw = raw_inputs(rows)
    assert raw.shape == (2, 67)
    torch.testing.assert_close(raw[:, :3], torch.tensor([[.5, 1.5, 2.5], [3., 4., 5.]]))
    model = RawActionEffect(torch.zeros(67), torch.ones(67))
    assert model(raw).shape == (2, 3)
