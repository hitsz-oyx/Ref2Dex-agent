import pytest
import torch

from src.task.CmResidual.cm_critic_salience import (
    critic_salience_weight, normalized_critic_loss,
)


def test_critic_salience_uses_contact_and_one_step_effect():
    probability = torch.tensor([0., .5, 1.])
    delta = torch.tensor([[0., 0., .01], [0., 0., .0015],
                          [0., 0., -.006]])
    weight = critic_salience_weight(probability, delta, coefficient=1.)
    torch.testing.assert_close(weight, torch.tensor([1., 1.25, 2.]))
    assert not weight.requires_grad


def test_critic_loss_normalization_preserves_mean_weight():
    loss = torch.ones(3, 1, requires_grad=True)
    weighted = normalized_critic_loss(loss, torch.tensor([1., 1.5, 2.]))
    torch.testing.assert_close(weighted.mean(), torch.tensor(1.))
    assert weighted[2] > weighted[0]
    weighted.mean().backward()
    assert loss.grad[2] > loss.grad[0]


def test_critic_salience_rejects_invalid_weight():
    with pytest.raises(FloatingPointError):
        normalized_critic_loss(torch.ones(2, 1), torch.tensor([1., float("nan")]))
