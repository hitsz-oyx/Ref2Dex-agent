import unittest

import torch

from src.task.CmResidual.cm_ppo_weight import effect_actor_weight


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


if __name__ == "__main__":
    unittest.main()
