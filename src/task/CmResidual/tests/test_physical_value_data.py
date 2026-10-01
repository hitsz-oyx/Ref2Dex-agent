import pytest
import torch
from src.task.CmResidual.physical_value_data import Episodes


def rows():
    # Episode 0 is held out, episode 2 is fit. Episode 3 is incomplete.
    return dict(state=torch.arange(6.).view(6, 1).expand(6, 55).clone(),
                next_state=torch.zeros(6, 55), context=torch.zeros(6, 8), next_context=torch.zeros(6, 8),
                action=torch.zeros(6, 18), previous_action=torch.ones(6, 18),
                reward=torch.tensor([1., 2., 10., 20., 30., 1000.]),
                done=torch.tensor([False, True, False, False, True, False]),
                episode_id=torch.tensor([0, 0, 2, 2, 2, 3]),
                step=torch.tensor([0, 1, 0, 1, 2, 0]))


def test_complete_returns_and_fit_holdout_are_episode_disjoint():
    dataset = Episodes.from_data(rows(), gamma=.5)
    assert dataset.excluded_rows == 1
    torch.testing.assert_close(dataset.data["return"], torch.tensor([2., 2., 27.5, 35., 30.]))
    assert set(dataset.data["episode_id"][dataset.fit].tolist()).isdisjoint(
        dataset.data["episode_id"][dataset.hold].tolist())
    assert dataset.subset(1).tolist() == [2, 3, 4]


def test_windows_and_future_stop_at_episode_boundary():
    dataset = Episodes.from_data(rows())
    batch = dataset.batch(torch.tensor([2]), "cpu")
    assert batch["history_mask"].sum() == 1
    assert not batch["history_state"][0, :-1].any()
    future, valid = dataset.future(torch.tensor([1, 3]), 1, "cpu")
    assert valid.tolist() == [False, True]


def test_duplicate_or_missing_steps_are_invalid_not_new_episode():
    data = rows()
    data["step"][3] = 0
    with pytest.raises(ValueError, match="duplicate steps"):
        Episodes.from_data(data)
