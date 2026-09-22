import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch

from src.task.CmResidual.cmlite import (
    CmLite, FrozenCmLite, SCHEMA, contact_gate, goal_reward, proximity_trust,
)


TOOLS = Path(__file__).resolve().parents[1] / "tools"


def test_goal_reward_is_positive_only_and_contact_weighted():
    current = torch.zeros(2, 3)
    goal = torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]])
    delta = torch.tensor([[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]])
    reward = goal_reward(current, goal, delta, torch.tensor([0.5, 1.0]))
    assert reward[0] > 0
    assert reward[1] == 0


def test_contact_gate_requires_explicit_predicted_contact_arm():
    predicted = torch.tensor([0.25, 0.75])
    actual = torch.tensor([False, True])
    torch.testing.assert_close(
        contact_gate(predicted, actual, use_predicted_contact=False),
        torch.tensor([0.0, 0.75]))
    torch.testing.assert_close(
        contact_gate(predicted, actual, use_predicted_contact=True), predicted)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        contact_gate(torch.tensor([1.1]), torch.tensor([True]), use_predicted_contact=True)


def test_proximity_trust_rejects_far_predicted_contact_rewards():
    torch.testing.assert_close(
        proximity_trust(torch.tensor([0.0, 0.1, 0.1001]), 0.1),
        torch.tensor([1.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match="positive"):
        proximity_trust(torch.tensor([0.1]), float("inf"))


def test_cmlite_bootstrap_exposes_dense_gate_and_checkpoint_cadence(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "cmlite_bootstrap_contract", TOOLS / "dexplore_cmlite_rank_bootstrap.py")
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    checkpoint = tmp_path / "cmlite.pt"
    checkpoint.write_bytes(b"test")
    args, passthrough = bootstrap.parse_cmlite_args([
        "--cmlite-reward-coef", "5", "--cmlite-checkpoint", str(checkpoint),
        "--cmlite-sha256", "a" * 64, "--actual-epochs", "3",
        "--lift-progress-reward-coef", "5",
        "--contact-before", "3",
        "--use-predicted-contact", "--max-cmlite-gap-m", "0.1",
        "--curriculum-backtrack-start", "20", "--curriculum-backtrack-end", "60",
        "--save-frequency", "1", "--task", "Dexplore_Inspire",
    ])
    assert args.use_predicted_contact
    assert args.max_cmlite_gap_m == 0.1
    assert args.lift_progress_reward_coef == 5
    assert args.curriculum_backtrack_start == 20
    assert args.curriculum_backtrack_end == 60
    assert args.save_frequency == 1
    assert passthrough == ["--task", "Dexplore_Inspire"]
    with pytest.raises(ValueError, match="save frequency"):
        bootstrap.parse_cmlite_args([
            "--cmlite-reward-coef", "5", "--cmlite-checkpoint", str(checkpoint),
            "--cmlite-sha256", "a" * 64, "--actual-epochs", "3",
            "--save-frequency", "0",
        ])
    with pytest.raises(ValueError, match="requires predicted contact"):
        bootstrap.parse_cmlite_args([
            "--cmlite-reward-coef", "5", "--cmlite-checkpoint", str(checkpoint),
            "--cmlite-sha256", "a" * 64, "--actual-epochs", "3",
            "--max-cmlite-gap-m", "0.1",
        ])


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
