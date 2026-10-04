"""Rank metrics and causal input/pair boundaries for the Ref2 Probe."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from action_ranking import (action_features, average_ranks, conditional_pairs,
                            pair_correct, spearman)


def test_spearman_uses_ranks_not_pearson_and_handles_ties():
    first = torch.tensor([0., 1., 2., 100.])
    second = torch.tensor([2., 1., 0., 100.])
    assert float(torch.corrcoef(torch.stack([first, second]))[0, 1]) > .99
    assert abs(spearman(first, second) - .2) < 1e-9
    assert torch.equal(average_ranks(torch.tensor([2., 1., 2., 4.])), torch.tensor([2.5, 1., 2.5, 4.], dtype=torch.double))
    assert spearman(torch.zeros(4), second) is None


def test_current_action_arm_never_reads_realized_future_actions():
    action = torch.randn(6, 4, 18)
    changed = action.clone()
    changed[:, 1:] += 100
    first, second = action_features(action, "HaK1"), action_features(changed, "HaK1")
    assert torch.equal(first, second)
    assert torch.equal(first[:, -1], action[:, 0])
    assert not first[:, :-1].any()
    assert not action_features(action, "H").any()
    assert not torch.equal(action_features(action, "HaK4_realized"), action_features(changed, "HaK4_realized"))


def test_pairs_preserve_split_motion_phase_noise_and_exclude_same_episode():
    strata = torch.tensor([[0, 0, 0]] * 4 + [[0, 1, 0]] * 4 + [[0, 0, 1]] * 4)
    episode = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
    subset = torch.ones(12, dtype=torch.bool)
    subset[1] = False
    pairs = conditional_pairs(strata, episode, subset, 207)
    assert subset[pairs].all()
    assert torch.equal(strata[pairs[:, 0]], strata[pairs[:, 1]])
    assert (episode[pairs[:, 0]] != episode[pairs[:, 1]]).all()
    assert not (pairs == 1).any()
    assert torch.equal(pairs, conditional_pairs(strata, episode, subset, 207))


def test_pair_accuracy_credits_score_ties_and_omits_exact_target_ties():
    scores = torch.tensor([2., 0., 0., 10.])
    targets = torch.tensor([3., 1., 2., 3.])
    pairs = torch.tensor([[0, 1], [1, 2], [0, 3]])
    correct, valid = pair_correct(scores, targets, pairs)
    assert torch.equal(correct[:2], torch.tensor([1., .5]))
    assert torch.equal(valid, torch.tensor([True, True, False]))
