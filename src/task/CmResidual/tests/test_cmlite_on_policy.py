import pytest
import torch

from src.task.CmResidual.tools.analyze_cmlite_on_policy import first_episode_mask


def test_first_episode_mask_excludes_terminal_and_repeats():
    done = torch.tensor([0, 0, 1, 0, 0, 1, 1, 0], dtype=torch.bool)[:, None]
    assert first_episode_mask(done, 2).tolist() == [True, True, False, True,
                                                     False, False, False, False]


def test_first_episode_mask_requires_completed_envs():
    with pytest.raises(ValueError, match="completed first episode"):
        first_episode_mask(torch.tensor([0, 0, 1, 0]), 2)
