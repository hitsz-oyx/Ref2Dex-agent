import unittest

import torch

from src.task.CmResidual.cm_ppo_weight import (
    align_active_weight_multiset, effect_actor_weight, permute_active_actions,
    permute_active_weights,
)


def test_component_rank_keeps_active_multiset_and_inactive_samples():
    weights = torch.tensor([1.2, 1.8, 1.4, 1.6, 1.1])
    score = torch.tensor([0.2, 0.1, 0.8, 0.4, 0.9])
    mask = torch.tensor([1., 1., 0., 1., 1.])
    ranked = align_active_weight_multiset(weights, score, mask)
    assert torch.equal(torch.sort(ranked[mask.bool()]).values,
                       torch.sort(weights[mask.bool()]).values)
    assert ranked[2] == weights[2]
    assert ranked[1] == 1.1
    assert ranked[4] == 1.8


def test_action_shuffle_preserves_active_multiset_and_inactive_samples():
    actions = torch.arange(12.).reshape(4, 3)
    mask = torch.tensor([1., 1., 0., 1.])
    shuffled = permute_active_actions(actions, mask, torch.Generator().manual_seed(153))
    assert torch.equal(shuffled[2], actions[2])
    assert torch.equal(torch.sort(shuffled[mask.bool()], dim=0).values,
                       torch.sort(actions[mask.bool()], dim=0).values)
    assert not torch.equal(shuffled[mask.bool()], actions[mask.bool()])


class CmPpoWeightTests(unittest.TestCase):
    def test_bounded_and_contact_action_conditioned(self):
        contact = torch.tensor([0.0, 0.5, 1.0])
        effect = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0015],
                               [0.0, 0.0, -0.004]])
        actual = effect_actor_weight(contact, effect, coefficient=1.0)
        torch.testing.assert_close(actual, torch.tensor([1.0, 1.25, 2.0]))
        self.assertFalse(actual.requires_grad)

    def test_zero_coefficient_is_exact_control(self):
        actual = effect_actor_weight(torch.tensor([0.2, 0.9]),
                                     torch.ones(2, 3), coefficient=0)
        torch.testing.assert_close(actual, torch.ones(2))

    def test_rejects_nonfinite(self):
        with self.assertRaises(FloatingPointError):
            effect_actor_weight(torch.tensor([float("nan")]),
                                torch.zeros(1, 3), coefficient=1)

    def test_placebo_preserves_active_marginal_and_inactive_samples(self):
        weights = torch.tensor([1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
        mask = torch.tensor([1., 0., 1., 1., 0., 1.])
        generator = torch.Generator().manual_seed(152)
        shuffled = permute_active_weights(weights, mask, generator)
        torch.testing.assert_close(shuffled[mask == 0], weights[mask == 0])
        torch.testing.assert_close(shuffled[mask == 1].sort().values,
                                   weights[mask == 1].sort().values)
        self.assertFalse(torch.equal(shuffled[mask == 1], weights[mask == 1]))

    def test_placebo_rejects_bad_mask(self):
        with self.assertRaises(ValueError):
            permute_active_weights(torch.ones(2), torch.tensor([1., 0.5]),
                                   torch.Generator())


if __name__ == "__main__":
    unittest.main()
