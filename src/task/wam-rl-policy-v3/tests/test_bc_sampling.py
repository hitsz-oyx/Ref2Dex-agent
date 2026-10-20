import numpy as np
import pytest

from wam_rl_policy_v3.bc_sampling import sample_rows, startup_rows


def test_startup_window_matches_shadow_physical_time_at_inspire_clock():
    np.testing.assert_array_equal(startup_rows(np.arange(20), 60), np.arange(8))
    np.testing.assert_array_equal(startup_rows(np.arange(20), 30), np.arange(4))


def test_masked_rows_use_original_clock_instead_of_compressed_indices():
    # A missing early observation must not move tick4 into the startup pool.
    np.testing.assert_array_equal(startup_rows(np.array([0, 2, 4, 7]), 30), [0, 1])
    early = startup_rows(np.array([0, 2, 4, 7]), 30)
    drawn = sample_rows(np.random.default_rng(241), 4, early, 2048)
    assert set(drawn[:1024]) <= {0, 1}
    assert set(drawn[1024:]) == {0, 1, 2, 3}


def test_shadow_draw_is_preserved_and_uniform_is_explicit():
    early = np.arange(8)
    expected_rng = np.random.default_rng(242)
    expected = np.concatenate((expected_rng.choice(early, 1024), expected_rng.integers(100, size=1024)))
    np.testing.assert_array_equal(sample_rows(np.random.default_rng(242), 100, early, 2048), expected)
    np.testing.assert_array_equal(sample_rows(np.random.default_rng(242), 100, early, 2048, 'uniform'),
                                 np.random.default_rng(242).integers(100, size=2048))
    with pytest.raises(ValueError, match='real startup'):
        sample_rows(np.random.default_rng(0), 100, [], 2048)
