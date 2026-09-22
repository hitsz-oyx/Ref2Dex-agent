import hashlib

import pytest
import torch

from src.task.CmResidual.cmlite import CmLite, FrozenCmLite, SCHEMA, goal_reward


def test_goal_reward_is_positive_only_and_contact_weighted():
    current = torch.zeros(2, 3)
    goal = torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]])
    delta = torch.tensor([[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]])
    reward = goal_reward(current, goal, delta, torch.tensor([0.5, 1.0]))
    assert reward[0] > 0
    assert reward[1] == 0


def test_frozen_cmlite_verifies_checksum(tmp_path):
    model = CmLite(width=16, blocks=1)
    checkpoint = tmp_path / "model.pt"
    torch.save({
        "schema": SCHEMA, "model_config": {"width": 16, "blocks": 1},
        "model": model.state_dict(), "feature_mean": torch.zeros(49),
        "feature_std": torch.ones(49), "target_mean": torch.zeros(3),
        "target_std": torch.ones(3),
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    FrozenCmLite(str(checkpoint), "cpu", digest)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        FrozenCmLite(str(checkpoint), "cpu", "0" * 64)
