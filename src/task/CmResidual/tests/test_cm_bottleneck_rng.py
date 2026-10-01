"""Feature construction must not reseed simulator CUDA sampling."""
import unittest
import torch
from src.task.CmResidual.cm_bottleneck_student import PhysicalFeatures


class FeatureRngTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA RNG used by simulator required")
    def test_all_arms_preserve_cuda_state_before_episode_sampling(self):
        net = torch.nn.Sequential(torch.nn.Linear(1486, 128), torch.nn.ReLU(),
                                  torch.nn.Linear(128, 128), torch.nn.ReLU(),
                                  torch.nn.Linear(128, 6))
        checkpoint = dict(mean=torch.zeros(1486), std=torch.ones(1486),
                          delta_mean=torch.zeros(3), delta_std=torch.ones(3),
                          state_dict=net.state_dict())
        for arm in ("on", "off", "random", "action"):
            with self.subTest(arm=arm):
                torch.manual_seed(281)
                before = torch.cuda.get_rng_state().clone()
                PhysicalFeatures(checkpoint, arm)
                self.assertTrue(torch.equal(before, torch.cuda.get_rng_state()), arm)


if __name__ == "__main__":
    unittest.main()
