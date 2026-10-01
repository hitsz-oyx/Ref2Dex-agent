"""Leakage, history equivalence and saved optimizer ownership contracts."""
import copy
import unittest

import torch
from torch import nn

from scripts.probe_current_policy_value_fit import (
    cached_history, episode_split, fit_value, history_rows, restore_value_adam,
)
from scripts.audit_current_policy_value import DiagnosticError
from src.task.CmResidual.physical_value_models import Features


class ValueFitContracts(unittest.TestCase):
    def test_paired_split_is_episode_disjoint_and_outcome_independent(self):
        rows = []
        groups = []
        for cp in (286, 287):
            for motion in range(3):
                for env in range(motion * 32, (motion + 1) * 32):
                    start = len(rows)
                    # Deliberately duplicate numerical IDs across checkpoints.
                    rows.extend([(cp, env, motion, env, step) for step in range(3)])
                    groups.append(((cp, env), list(range(start, start + 3))))
        tensors = torch.tensor(rows)
        data = dict(zip(('checkpoint_seed', 'episode_id', 'motion_id', 'env_id', 'step'), tensors.T))
        data['reward'] = torch.zeros(len(rows))
        fit, hold, records = episode_split(data, groups)
        for cp in (286, 287):
            for motion in range(3):
                panel = [r for r in records if r['checkpoint'] == cp and r['motion_id'] == motion]
                self.assertEqual(sum(r['split'] == 'holdout' for r in panel), 6)
        for _, indices in groups:
            self.assertEqual(len(set(hold[indices].tolist())), 1)
        self.assertFalse((fit & hold).any())
        self.assertTrue((fit | hold).all())
        first = {(r['motion_id'], r['env_id']): r['split'] for r in records if r['checkpoint'] == 286}
        second = {(r['motion_id'], r['env_id']): r['split'] for r in records if r['checkpoint'] == 287}
        self.assertEqual(first, second)
        data['reward'] = torch.randn(len(rows)) * 9999
        self.assertTrue(torch.equal(hold, episode_split(data, groups)[1]))
        data['env_id'][-1] = -1
        with self.assertRaises(DiagnosticError):
            episode_split(data, groups)

    def test_cache_equals_native_features_with_padding_and_checkpoint_boundary(self):
        features = Features(torch.randn(55), torch.rand(55) + .2,
                            torch.zeros(3), torch.ones(3))
        states = torch.randn(8, 55)
        action = torch.randn(8, 18)
        cache = dict(episode_id=torch.tensor([4, 4, 4, 4, 4, 4, 5, 5]),
                     checkpoint_seed=torch.tensor([1, 1, 1, 2, 2, 2, 2, 2]),
                     step=torch.tensor([0, 1, 2, 0, 1, 2, 0, 2]),
                     previous_action=action, state_features=features.state(states))
        idx = torch.arange(8)
        safe, mask = history_rows(cache, cache['checkpoint_seed'], idx)
        direct = features.history(states[safe] * mask, action[safe] * mask, mask)
        self.assertTrue(torch.allclose(cached_history(cache, idx), direct, atol=1e-6))
        self.assertEqual(int(mask[3].sum()), 1)  # same episode number, new checkpoint
        self.assertEqual(int(mask[7].sum()), 1)  # discontinuous predecessor step
        self.assertTrue(torch.equal(cached_history(cache, idx)[:, -1, -1], torch.ones(8)))

    def test_restore_keeps_value_moments_and_drops_q(self):
        value, q = nn.Linear(3, 1), nn.Linear(3, 1)
        joint = torch.optim.Adam(list(value.parameters()) + list(q.parameters()), lr=.001)
        (value(torch.ones(5, 3)).square().sum() + q(torch.ones(5, 3)).square().sum()).backward()
        joint.step()
        original = copy.deepcopy(joint.state_dict())
        restored = restore_value_adam(value, original)
        self.assertEqual(len(restored.state), 2)
        for parameter, pid in zip(value.parameters(), original['param_groups'][0]['params'][:2]):
            for field in ('step', 'exp_avg', 'exp_avg_sq'):
                self.assertTrue(torch.equal(restored.state[parameter][field], original['state'][pid][field]))
        self.assertEqual(len(original['state']), 4)
        bad = copy.deepcopy(original)
        bad['state'][0]['exp_avg'] = torch.zeros(99)
        with self.assertRaises(DiagnosticError):
            restore_value_adam(value, bad)

    def test_training_never_reads_holdout_targets(self):
        class SmallValue(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)
                nn.init.zeros_(self.linear.weight)
                nn.init.zeros_(self.linear.bias)

            def forward(self, history, context):
                return self.linear(context)

        cache = dict(episode_id=torch.arange(6), checkpoint_seed=torch.zeros(6, dtype=torch.long),
                     step=torch.zeros(6, dtype=torch.long), state_features=torch.zeros(6, 73),
                     previous_action=torch.zeros(6, 18), context=torch.ones(6, 1))
        fit = torch.tensor([0, 1, 2])
        first_target = torch.tensor([2., 2., 2., 1., 1., 1.])
        second_target = torch.tensor([2., 2., 2., float('nan'), 1e9, -1e9])
        snapshots = []
        for target in (first_target, second_target):
            model = SmallValue()
            optimizer = torch.optim.Adam(model.parameters(), lr=.05)
            fit_value(model, optimizer, cache, target, fit, torch.Generator().manual_seed(42),
                      1., 50, lambda: None)
            snapshots.append(copy.deepcopy(model.state_dict()))
        for name in snapshots[0]:
            self.assertTrue(torch.equal(snapshots[0][name], snapshots[1][name]))
        self.assertLess(abs(float(model(torch.zeros(1, 16, 92), torch.ones(1, 1))) - 2), .2)
        self.assertTrue(torch.isnan(second_target[3]))


if __name__ == '__main__':
    unittest.main()
