import unittest

import torch

from src.task.CmResidual.cm_ppo_weight import effect_actor_weight, permute_active_weights


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
