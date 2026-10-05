import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from ranking_tolerance import (same_state_ranking, within_state_shuffle,
                              inject_y_noise, assert_environment_oof,
                              closed_loop_gain_retention)


def test_missing_counterfactual_and_ties():
    gt = [[1, 1, 0, np.nan], [np.nan, 9, np.nan, np.nan], [np.nan]*4]
    pred = [[0, 0, 2, np.nan], [np.nan, 9, np.nan, np.nan], [np.nan]*4]
    mask = [[1, 1, 1, 0], [0, 1, 0, 0], [0]*4]
    result = same_state_ranking(gt, pred, mask)
    assert result['gt_tied_pairs'] == 1 and result['strict_pairs'] == 2
    assert result['pairwise_accuracy'] == 0
    assert result['selected'] == [2, 1, -1]
    assert result['top1_regret'] == [1, None, None]
    assert same_state_ranking([[1,0]], [[1,1]], [[1,1]])['pairwise_accuracy'] == .5


def test_invalid_observed_score_rejected():
    with pytest.raises(ValueError):
        same_state_ranking([[np.nan,0]], [[0,0]], [[1,1]])


def test_shuffle_never_crosses_state_or_missing_entries():
    values = np.array([[1,2,np.nan], [10,20,np.nan]])
    shuffled = within_state_shuffle(values, np.isfinite(values), seed=12)
    assert np.isnan(shuffled[:,2]).all()
    for a, b in zip(values[:,:2], shuffled[:,:2]):
        assert sorted(a) == sorted(b)


def test_noise_only_observed_and_reproducible():
    y = np.arange(12.).reshape(2,3,2)
    mask = np.array([[1,1,0], [1,0,0]], dtype=bool)
    noisy = inject_y_noise(y, mask, sigma=[.1,.2], seed=12)
    assert np.array_equal(noisy[~mask], y[~mask])
    assert np.array_equal(noisy, inject_y_noise(y, mask, sigma=[.1,.2], seed=12))
    assert np.array_equal(y, inject_y_noise(y, mask, sigma=0, seed=12))
    with pytest.raises(ValueError):
        inject_y_noise(y, mask, sigma=-1, seed=12)


def test_environment_oof_and_real_gain_contract():
    with pytest.raises(ValueError):
        assert_environment_oof([1,2], [2,3])
    assert assert_environment_oof([1,1], [2])['train_environments'] == 1
    assert closed_loop_gain_retention([0,0], [1,1], [1,0], evaluation_ids=['a','b'])['retention'] == .5
    assert closed_loop_gain_retention([1], [1], [0], evaluation_ids=['a'])['retention'] is None
    with pytest.raises(ValueError):
        closed_loop_gain_retention([0,0], [1,1], [1,0], evaluation_ids=['a','a'])


def test_oof_composite_seed_environment_identity():
    assert assert_environment_oof([(263, 1), (263, 1)], [[264, 1]]) == {
        'train_environments': 1, 'test_environments': 1}
    with pytest.raises(ValueError, match='leakage'):
        assert_environment_oof([(263, 1)], np.array([[263, 1], [264, 1]]))
