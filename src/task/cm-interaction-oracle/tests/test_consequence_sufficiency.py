from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src/task/cm-interaction-oracle/src'))
from consequence_sufficiency import readouts, matched_inputs, cluster_gain, supported_ranking, cross_half_bridge


def test_future_outcomes_do_not_enter_step8_consequences_and_height_is_separate():
    before = torch.zeros(2, 72); before[:, 2] = .04; before[:, 6] = 1; before[:, 71] = 1
    tr = before[:, None].repeat(1, 32, 1)
    p = dict(before=before, trajectory=tr, rest_height=torch.zeros(2))
    e, i, signed, y, _ = readouts(p)
    tr[:, 8:, 48:66] = 33
    tr[0, 8:, 2] = .01  # physical failure while contact proxy remains true
    tr[1, 8:, 71] = 0   # proxy loss while height is preserved
    ee, ii, ss, yy, _ = readouts(p)
    assert torch.equal(e, ee) and torch.equal(i, ii) and torch.equal(signed, ss)
    assert yy[:, 6].tolist() == [1, 0] and yy[:, 7].tolist() == [0, 1]
    assert not torch.equal(y, yy)


def test_identical_slot_capacity_and_absent_action_is_zero():
    values = [torch.ones(4, n) for n in (104, 14, 12, 14, 18)]
    inputs = matched_inputs(*values)
    assert {v.shape for v in inputs.values()} == {torch.Size([4, 162])}
    assert inputs['HEI'][:, 104:118].count_nonzero() == 0
    assert inputs['HaEI'][:, 104:118].count_nonzero() == 56
    assert inputs['HEI'][:, -18:].count_nonzero() == 0
    assert inputs['HEI_signed'][:, -18:].count_nonzero() == 72


def test_bootstrap_uses_environment_units_and_matched_ratio():
    base = np.arange(1, 13, dtype=float); clusters = np.repeat(np.arange(4), 3)
    result = cluster_gain(base, base * .8, clusters, 11, repeats=50)
    assert result['clusters'] == 4
    assert np.allclose([result['gain'], result['lower95'], result['upper95']], .2)


def test_ranking_does_not_give_tiny_strata_equal_weight():
    y = np.zeros((10, 8)); y[:, 3] = np.arange(10) / 10; y[:, 6] = np.arange(10) % 2; y[:, 7] = y[:, 3]
    groups = np.array([0] * 8 + [1] * 2); arms = np.arange(10) % 2
    result = supported_ranking(y, y, np.arange(10), groups, arms)
    assert result['contact32']['accuracy'] == 1
    assert result['contact32']['support'][-1] == dict(stratum=1, pairs=1, included=False)


def test_bridge_training_excludes_held_arm_source_outcome():
    rng = np.random.default_rng(11)
    a = rng.normal(size=(14, 52)); b = rng.normal(size=(14, 52))
    before = cross_half_bridge([a, b])['EI']['predictions']
    a[0, 44:] += 999
    after = cross_half_bridge([a, b])['EI']['predictions']
    assert np.allclose(before[0], after[0])  # first direction's held arm never trained
    assert not np.allclose(before[1], after[1])  # other arms legitimately train on source arm0
