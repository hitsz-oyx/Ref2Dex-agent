import pytest
import torch

from src.task.CmResidual.cm_residual_policy import (
    consequence_score,
    world_height_score_mm,
    validate_policy_mode,
)
from src.task.CmResidual.residual_probe_contract import snapshot_window_metadata, simulator_seed_from_argv


def test_consequence_score_uses_world_height_not_local_z():
    output = torch.tensor([[0.0, 0.0, 3.0, 10.0, -10.0, -10.0]])
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    ninety_y = torch.tensor([[0.0, 2 ** -0.5, 0.0, 2 ** -0.5]])
    identity_score, _, _ = consequence_score(output, identity)
    rotated_score, _, _ = consequence_score(output, ninety_y)
    assert identity_score.item() > 2.9
    assert abs(rotated_score.item()) < 1e-5
    assert abs(world_height_score_mm(ninety_y, output[:, :3]).item()) < 1e-5
    assert abs(identity_score.item() - world_height_score_mm(identity, output[:, :3]).item()) < 0.01


def test_shuffled_policy_mode_is_rejected_as_invalid_control():
    with pytest.raises(ValueError, match="shuffled"):
        validate_policy_mode("shuffled")
    validate_policy_mode("cm")


def test_window_metadata_is_a_trigger_snapshot():
    motion = torch.tensor([11, 22])
    start = torch.tensor([101, 202])
    rest = torch.tensor([0.1, 0.2])
    saved = snapshot_window_metadata(motion, start, rest, torch.tensor([1]))
    motion[1], start[1], rest[1] = 99, 999, 9.9
    assert torch.equal(saved[0], torch.tensor([22]))
    assert torch.equal(saved[1], torch.tensor([202]))
    assert torch.equal(saved[2], torch.tensor([0.2]))


def test_simulator_seed_is_explicit_and_unambiguous():
    assert simulator_seed_from_argv(["--seed", "570", "--headless"]) == 570
    assert simulator_seed_from_argv(["--seed=571"]) == 571
    with pytest.raises(ValueError, match="exactly one"):
        simulator_seed_from_argv(["--seed", "570", "--seed", "571"])
