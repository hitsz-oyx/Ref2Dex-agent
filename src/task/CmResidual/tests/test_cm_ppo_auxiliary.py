from __future__ import annotations

import torch

from src.task.CmResidual.cm_ppo_auxiliary import (
    cm_candidate_targets, masked_auxiliary_loss, raw_cm_input,
)


class DummyCm(torch.nn.Module):
    def forward(self, raw):
        action_z = raw[:, 18 + 2]
        return {"contact_fraction": .5 - action_z * .1,
                "delta_local": torch.stack((action_z * 0, action_z * 0,
                                            action_z * .02), dim=-1),
                "followup_delta_local": torch.stack((action_z * 0, action_z * 0,
                                                     action_z * .01), dim=-1)}


def test_candidate_targets_are_masked_and_directional():
    q = torch.zeros(2, 18)
    obj = torch.zeros(2, 13)
    obj[:, 6] = 1
    mask = torch.tensor([True, False])
    result = cm_candidate_targets(DummyCm(), q, q, obj, q, mask)
    assert result.shape == (2, 3)
    torch.testing.assert_close(result[0], torch.tensor([.2, .2, .1]))
    torch.testing.assert_close(result[1], torch.zeros(3))
    assert raw_cm_input(q, q, obj, q).shape == (2, 67)


def test_auxiliary_loss_backpropagates_to_shared_actor_feature():
    trunk = torch.nn.Linear(4, 8)
    head = torch.nn.Linear(8, 3)
    predicted = head(trunk(torch.ones(2, 4)))
    loss = masked_auxiliary_loss(predicted, torch.ones(2, 3),
                                 torch.tensor([[1.], [0.]])).mean()
    loss.backward()
    assert trunk.weight.grad is not None
    assert trunk.weight.grad.abs().sum() > 0
    assert head.weight.grad.abs().sum() > 0
