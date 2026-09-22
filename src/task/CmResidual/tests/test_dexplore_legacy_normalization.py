from __future__ import annotations

from pathlib import Path
import sys

import torch


DEXPLORE = Path(__file__).resolve().parents[4] / "third_party" / "DExplore" / "dexplore"
if str(DEXPLORE) not in sys.path:
    sys.path.insert(0, str(DEXPLORE))

from learning.common_agent import CommonAgent  # noqa: E402
from learning.common_player import CommonPlayer  # noqa: E402


class _AddOne(torch.nn.Module):
    def forward(self, value):
        return value + 1


def test_player_preprocess_uses_external_running_mean_std():
    player = object.__new__(CommonPlayer)
    player.normalize_input = True
    player.running_mean_std = _AddOne()
    value = torch.zeros(2, 3)
    assert torch.equal(player._preproc_obs(value), torch.ones_like(value))


def test_agent_preprocess_uses_external_running_mean_std():
    agent = object.__new__(CommonAgent)
    agent.normalize_input = True
    agent.running_mean_std = _AddOne()
    value = torch.zeros(2, 3)
    assert torch.equal(agent._preproc_obs(value), torch.ones_like(value))


def test_agent_mode_controls_external_running_mean_std():
    agent = object.__new__(CommonAgent)
    agent.normalize_input = True
    agent.running_mean_std = _AddOne()
    agent.model = torch.nn.Identity()
    agent.normalize_rms_advantage = False
    agent.epochs_between_resets = 0
    agent.set_eval()
    assert not agent.running_mean_std.training
    agent.set_train()
    assert agent.running_mean_std.training
