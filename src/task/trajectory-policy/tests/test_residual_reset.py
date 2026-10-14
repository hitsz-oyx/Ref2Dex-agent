import numpy as np
import pytest

from trajectory_policy.residual_reset import reset_context


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
