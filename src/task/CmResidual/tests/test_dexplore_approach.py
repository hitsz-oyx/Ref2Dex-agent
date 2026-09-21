"""Matched scratch-arm approach shaping and terminal alignment."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from src.task.CmResidual.dexplore_approach import (
    ApproachConfig, potential_approach_reward, sampled_surface_gap,
)


TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOOLS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sampled_gap_and_terminal_potential_are_aligned():
    config = ApproachConfig(point_stride=1, proximity_scale_m=0.1)
    hand = torch.tensor([[[0., 0., 0.], [0.5, 0., 0.]],
                         [[1., 0., 0.], [1.5, 0., 0.]]])
    obj = torch.tensor([[[0.03, 0., 0.]], [[1.04, 0., 0.]]])
    before = sampled_surface_gap(hand, obj, config)
    assert torch.allclose(before, torch.tensor([0.03, 0.04]), atol=1e-7)
    after = torch.tensor([0.01, 0.005])
    shaped = potential_approach_reward(before, after,
                                       torch.tensor([False, True]), gamma=0.99, config=config)
    assert torch.allclose(shaped[0], torch.tensor(0.99 * 0.9 - 0.7))
    assert torch.allclose(shaped[1], torch.tensor(-0.6))


def test_bounded_linear_potential_has_slope_at_observed_scratch_distance():
    gamma = 0.99
    before = torch.tensor([0.78, 1.2, 0.01])
    after = torch.tensor([0.77, 1.1, 0.0])
    reward = potential_approach_reward(before, after, torch.zeros(3, dtype=torch.bool), gamma=gamma)
    assert torch.allclose(reward, torch.tensor([gamma * 0.23 - 0.22, 0.0, gamma - 0.99]))
    assert (reward.abs() <= 1).all()
    middle = torch.tensor([0.75])
    end = torch.tensor([0.73])
    first = potential_approach_reward(before[:1], middle, torch.tensor([False]), gamma=gamma)
    second = potential_approach_reward(middle, end, torch.tensor([False]), gamma=gamma)
    assert torch.allclose(first + gamma * second, torch.tensor([gamma**2 * 0.27 - 0.22]))


def test_nonfinite_geometry_fails_before_reward_is_added():
    with pytest.raises(FloatingPointError, match="finite"):
        sampled_surface_gap(torch.tensor([[[float("nan"), 0., 0.]]]),
                            torch.zeros(1, 1, 3))
    with pytest.raises(FloatingPointError, match="nonnegative"):
        potential_approach_reward(torch.tensor([0.02]), torch.tensor([-0.01]),
                                  torch.tensor([False]), gamma=0.99)


def test_both_rank_bootstraps_accept_same_explicit_approach_coefficient(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    off = _load("approach_off_bootstrap", "dexplore_cm_off_rank_bootstrap.py")
    on = _load("approach_on_bootstrap", "dexplore_cm_reward_rank_bootstrap.py")
    checkpoint = tmp_path / "calibrated.pt"
    checkpoint.write_bytes(b"fake-test-checkpoint")
    off_args, off_pass = off.parse_cm_off_args([
        "--cm-distill-coef", "0", "--actual-epochs", "1",
        "--approach-reward-coef", "2", "--task", "Dexplore_Inspire"])
    on_args, on_pass = on.parse_cm_reward_args([
        "--cm-reward-coef", "0.05", "--cmv2-checkpoint", str(checkpoint),
        "--cmv2-sha256", "a" * 64, "--actual-epochs", "1",
        "--approach-reward-coef", "2", "--task", "Dexplore_Inspire"])
    assert off_args.approach_reward_coef == on_args.approach_reward_coef == 2
    assert off_pass == on_pass == ["--task", "Dexplore_Inspire"]
    for module, argv in ((off, ["--cm-distill-coef", "0", "--actual-epochs", "1"]),):
        with pytest.raises(ValueError, match="nonnegative"):
            module.parse_cm_off_args(argv + ["--approach-reward-coef", "nan"])


def test_cm_off_installs_only_geometry_agent_when_approach_enabled(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    off = _load("approach_off_dispatch", "dexplore_cm_off_rank_bootstrap.py")
    monkeypatch.setattr(off, "_assert_cm_not_imported", lambda: None)
    received = {}
    monkeypatch.setattr(off.dexplore_ddp_rank_bootstrap, "main",
                        lambda args, **kw: received.update({"args": args, "kw": kw}))
    off.main(["--cm-distill-coef", "0", "--actual-epochs", "1",
              "--approach-reward-coef", "2", "--task", "Dexplore_Inspire"])
    assert received == {"args": ["--task", "Dexplore_Inspire"], "kw": {
        "agent_class": "src.task.CmResidual.dexplore_approach_agent:DExploreApproachAgent"}}
    assert off.os.environ["REF2DEX_APPROACH_REWARD_COEF"] == "2.0"


def test_launcher_limits_gpu_count_to_two_and_forwards_matched_shaping(tmp_path, monkeypatch):
    launcher = _load("approach_launcher", "run_dexplore_v120_ddp.py")
    assert launcher.parse_gpus("5,6") == (5, 6)
    with pytest.raises(ValueError, match="exactly two"):
        launcher.parse_gpus("5,6,7")
    captured = {}
    monkeypatch.setattr(launcher, "torchrun_command", lambda **kwargs: captured.update(kwargs) or ["echo"])
    launcher.main(["--gpus", "5,6", "--dry-run", "--rank-bootstrap",
                   str(TOOLS / "dexplore_cm_off_rank_bootstrap.py"),
                   "--cm-distill-coef", "0", "--approach-reward-coef", "2",
                   "--actual-epochs", "1"])
    assert captured["bootstrap_args"] == [
        "--cm-distill-coef", "0", "--actual-epochs", "1", "--approach-reward-coef", "2.0"]
