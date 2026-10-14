import numpy as np
import pytest

from trajectory_policy.residual_reset import reset_context, teacher_prefix_action


def test_all_phases_use_real_past_and_previous_applied_command():
    starts = [0, 32, 48, 62]
    n = len(starts)
    ticks = np.arange(63, dtype=np.float32)[:, None, None, None]
    prefix = dict(obj=np.broadcast_to(ticks, (63, n, 4, 4)).copy(),
                  hand=np.broadcast_to(ticks, (63, n, 11, 3)).copy(),
                  applied=np.broadcast_to(np.arange(62, dtype=np.float32)[:, None, None], (62, n, 18)).copy())
    current = {key: prefix[key][starts, np.arange(n)].copy() + .1 for key in ('obj', 'hand')}
    history, previous, z0 = reset_context(prefix, starts, current)
    for key in ('obj', 'hand'):
        actual = np.stack(history[key])
        np.testing.assert_array_equal(actual[:, 0], np.repeat(current[key][0:1], 4, axis=0))
        for env, phase in enumerate(starts[1:], 1):
            np.testing.assert_array_equal(actual[:3, env], prefix[key][phase-3:phase, env])
            np.testing.assert_array_equal(actual[-1, env], current[key][env])
    np.testing.assert_array_equal(previous[:, 0], [0, 31, 47, 61])
    np.testing.assert_array_equal(z0, np.zeros(n))
    # Neither the reset helper nor its callers may change stored prefix evidence.
    history['obj'][0][:] = -10
    assert prefix['obj'][45, 2, 0, 0] == 45
    with pytest.raises(ValueError):
        reset_context(prefix, [0, 32, 48, 63], current)


def test_teacher_query_initializes_legacy_batch_before_padding():
    import torch
    class Player:
        has_batch_dimension = False
        def get_batch_size(self, obs, dimension):
            assert obs.shape == (4, 1442) and dimension == 1
            self.has_batch_dimension = True
        def get_action(self, obs, deterministic):
            assert self.has_batch_dimension and deterministic
            assert obs['obs'].shape == (64, 1442)
            return torch.full((64, 18), 2.)
    action = teacher_prefix_action(Player(), {'obs': torch.zeros(4, 1442)}, 4)
    assert action.shape == (4, 18) and torch.equal(action, torch.ones(4, 18))
