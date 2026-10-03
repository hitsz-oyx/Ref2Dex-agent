"""Regression tests for the episode-local Gate 1 assembler contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from assemble_gate1_dataset_v2 import (  # noqa: E402
    _canonicalize_quaternion_sequence,
    assemble as assemble_reference,
)
from assemble_gate1_dataset_v2_fast import assemble as assemble_vectorized  # noqa: E402


def _write_interleaved_run(path: Path, physical_timing: str | None) -> None:
    path.mkdir()
    steps = 20
    rows = steps * 2
    episode = torch.tensor([ep for step in range(steps) for ep in range(2)])
    step = torch.tensor([step for step in range(steps) for _ in range(2)])
    action = torch.zeros(rows, 2)
    action[:, 0] = episode * 100 + step
    identity = torch.zeros(rows, 4)
    identity[:, 3] = 1
    payload = {
        "state": torch.zeros(rows, 3),
        "context": torch.zeros(rows, 2),
        "previous_action": action - 1,
        "action": action,
        "reward": torch.arange(rows, dtype=torch.float32),
        "reward_components": torch.zeros(rows, 1),
        "done": torch.zeros(rows, dtype=torch.bool),
        "episode_id": episode,
        "step": step,
        "progress": step,
        "motion_id": torch.zeros(rows, dtype=torch.long),
        "noise_std": torch.zeros(rows),
        "object_root": torch.cat((torch.zeros(rows, 3), identity,
                                   torch.zeros(rows, 6)), dim=1),
        "hand_body_position": torch.zeros(rows, 5, 3),
        "hand_body_quaternion": identity[:, None, :].expand(rows, 5, 4).clone(),
        "hand_force": torch.zeros(rows, 5, 3),
        "object_force": torch.zeros(rows, 3),
        "schema": "test",
        "gamma": 0.99,
        "control_dt": 1.0,
        "source_sha256": "test",
        "effect_definition": "test",
        "interaction_definition": "test",
    }
    if physical_timing is not None:
        payload["physical_timing"] = physical_timing
    payload["done"][-1] = True
    torch.save(payload, path / "transitions_000.pt")
    (path / "results.json").write_text(json.dumps({
        "per_episode": [
            {"episode_id": ep, "stable_success": False,
             "drop_after_success": False, "max_hold_seconds": 0.0,
             "mean_lift_meters": 0.0, "contact_fraction": 0.0}
            for ep in range(2)
        ]
    }))


@pytest.mark.parametrize("physical_timing", ["pre_env_step", "post_env_step_legacy"])
def test_reference_and_vectorized_preserve_episode_local_future_actions(
    tmp_path: Path, physical_timing: str,
) -> None:
    run = tmp_path / physical_timing
    _write_interleaved_run(run, physical_timing)
    reference, _ = assemble_reference([run], horizon=3, history_length=10)
    vectorized, _ = assemble_vectorized([run], horizon=3, history_length=10)

    for key, expected in reference.items():
        if isinstance(expected, torch.Tensor):
            assert torch.equal(expected, vectorized[key]), key

    # Each held-out window must see a_{t+1:t+H} from its own episode.
    future = reference["future_action"][..., 0]
    for ep in (0, 1):
        mask = reference["episode_id"] == ep
        expected = (reference["step"][mask, None]
                    + torch.arange(1, 4, dtype=torch.long)[None, :]
                    + ep * 100).to(future.dtype)
        assert torch.equal(future[mask], expected)


def test_quaternion_sign_is_stable_when_window_starts_with_equivalent_sign() -> None:
    q = torch.tensor([
        [0.0, 0.0, 0.6, 0.8],
        [0.0, 0.0, 0.7, 0.71414286],
        [0.0, 0.0, 0.8, 0.6],
    ])
    flipped = q.clone()
    flipped[0] *= -1
    assert torch.allclose(
        _canonicalize_quaternion_sequence(q),
        _canonicalize_quaternion_sequence(flipped),
        atol=1e-6,
    )


def test_missing_physical_timing_fails_closed(tmp_path: Path) -> None:
    run = tmp_path / "missing_timing"
    _write_interleaved_run(run, None)
    with pytest.raises(ValueError, match="physical_timing missing"):
        assemble_reference([run], horizon=3, history_length=10)
    with pytest.raises(ValueError, match="physical_timing missing"):
        assemble_vectorized([run], horizon=3, history_length=10)


def test_mixed_physical_timing_shards_fail_closed(tmp_path: Path) -> None:
    run = tmp_path / "mixed_timing"
    _write_interleaved_run(run, "pre_env_step")
    first = torch.load(run / "transitions_000.pt", map_location="cpu")
    first.pop("physical_timing")
    torch.save(first, run / "transitions_001.pt")
    for assemble in (assemble_reference, assemble_vectorized):
        with pytest.raises(ValueError, match="physical_timing.*every shard"):
            assemble([run], horizon=3, history_length=10)
