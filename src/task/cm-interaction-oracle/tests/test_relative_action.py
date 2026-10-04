"""Scientific label contracts: terminal, split, thresholds, and action controls."""
import importlib.util
from pathlib import Path

import torch

path = Path(__file__).resolve().parents[1] / "src" / "relative_action.py"
spec = importlib.util.spec_from_file_location("relative_action", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_terminal_reward_is_included_and_terminal_never_bootstraps():
    reward = torch.tensor([1., 2., 3., 4.])
    local, endpoint, discount = module.nstep_outcome(reward, torch.tensor([0, 2, 3]), 2, .5)
    assert torch.equal(local, torch.tensor([2., 5., 4.]))
    assert torch.equal(endpoint, torch.tensor([2, -1, -1]))
    assert torch.equal(discount, torch.tensor([.25, 0., 0.]))
    assert torch.equal(module.discounted_returns(reward, .5), torch.tensor([3.25, 4.5, 5., 4.]))


def test_episode_folds_exclude_complete_groups():
    episodes = [{"stratum": (i % 3, i % 4)} for i in range(90)]
    folds = module.stratified_folds(episodes, list(range(90)))
    assert sorted(sum(folds, [])) == list(range(90))
    for fold in folds:
        fit = set(range(90)) - set(fold)
        assert not fit.intersection(fold)
    train, test = module.episode_split([(0, i // 28, i) for i in range(112)])
    assert len(train) == 90 and len(test) == 22 and not set(train).intersection(test)


def test_thresholds_use_training_only_and_do_not_manufacture_zero_labels():
    strata = torch.zeros(64, 3, dtype=torch.long)
    train = torch.arange(64) < 32
    advantage = torch.cat([torch.zeros(32), torch.full((32,), 100.)])
    thresholds = module.fit_thresholds(advantage, strata, train)
    assert thresholds[(0, 0, 0)] == (-.05, .05)
    label = module.label_advantage(advantage, strata, thresholds)
    assert not label[train].any()
    assert (label[~train] == 1).all()
    # A held-out stratum receives no invented training threshold.
    strata[32:] = 1
    assert not module.label_advantage(advantage, strata, thresholds)[32:].any()


def test_action_permutation_never_crosses_phase_motion_or_noise():
    action = torch.arange(48).reshape(8, 2, 3).float()
    strata = torch.tensor([[0, 0, 0]] * 4 + [[0, 0, 1]] * 4)
    shuffled = module.permute_within_strata(action, strata, 203)
    assert sorted(shuffled[:4, 0, 0].tolist()) == action[:4, 0, 0].tolist()
    assert sorted(shuffled[4:, 0, 0].tolist()) == action[4:, 0, 0].tolist()


def test_auc_handles_tied_predictions():
    labels = torch.tensor([1, -1, 1, -1])
    metrics = module.classification_metrics(torch.zeros(4), labels)
    assert metrics == {"balanced_accuracy": .5, "auc": .5}


def test_auc_excludes_neutral_rows_from_ranks():
    metrics = module.classification_metrics(torch.tensor([-1., 0., 1.]), torch.tensor([-1, 0, 1]))
    assert metrics == {"balanced_accuracy": 1., "auc": 1.}
