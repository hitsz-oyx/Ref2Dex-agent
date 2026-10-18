import numpy as np
import pytest

from wam_rl_policy.student_inputs import student_features


@pytest.mark.parametrize('variant,pad_to', [('p2', 0), ('p2', 403), ('p2_tau', 0)])
def test_training_batch_and_online_row_inputs_match(variant, pad_to):
    rng = np.random.default_rng(16)
    p2 = rng.normal(size=(3, 139)).astype(np.float32)
    tau = rng.normal(size=(3, 8, 11, 3)).astype(np.float32)
    batched = student_features(p2, variant, tau=tau, pad_to=pad_to)
    online = np.stack([student_features(row, variant, tau=t.reshape(-1), pad_to=pad_to)
                       for row, t in zip(p2, tau)])
    np.testing.assert_array_equal(batched, online)
    np.testing.assert_array_equal(batched[:, :139], p2)
    if variant == 'p2' and pad_to:
        assert batched.shape == (3, 403) and not batched[:, 139:].any()
    elif variant == 'p2_tau':
        np.testing.assert_array_equal(batched[:, 139:], tau.reshape(3, -1))


def test_student_rejects_incompatible_or_missing_inputs():
    with pytest.raises(ValueError, match='requires'):
        student_features(np.zeros(139), 'p2_tau')
    with pytest.raises(ValueError, match='8x11x3'):
        student_features(np.zeros((2, 139)), 'p2_tau', tau=np.zeros((8, 11, 3)))
    with pytest.raises(ValueError, match='smaller'):
        student_features(np.zeros(139), 'p2', pad_to=138)
