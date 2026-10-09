import numpy as np
import torch
import importlib.util
from pathlib import Path

from consequence_evaluator.hand_action_retargeter import (
    ACTION_DIM, CONTEXT_DIM, CONTEXT_SCHEMA, HORIZON, ContextHandActionRetargeter,
    HandActionRetargeter, Standardizer, chunk_offset, hand_object_context,
    trajectory_input)
from consequence_evaluator.retarget_collection import (
    MODE_NAMES, PHASE_NAMES, phase_code, sample_structured_residual,
    validate_residual_family)
from consequence_evaluator.trajectory_utility import (
    OBJECT_EFFECT_DIM, SCHEMA as TRAJECTORY_SCHEMA, TRAJECTORY_DIM,
    TrajectoryUtility)

_runner_spec = importlib.util.spec_from_file_location(
    "hand_action_retarget_runner",
    Path(__file__).resolve().parents[1] / "tools/run/run_hand_action_retargeter.py")
_runner = importlib.util.module_from_spec(_runner_spec)
_runner_spec.loader.exec_module(_runner)


def test_structured_residual_family_is_bounded_and_covers_modes():
    rng = np.random.default_rng(22)
    rows = []
    modes = []
    for phase in range(3):
        value, mode = sample_structured_residual(rng, 256, phase)
        rows.append(value); modes.append(mode)
    value = np.concatenate(rows, axis=0)
    mode = np.concatenate(modes)
    assert validate_residual_family(value[None])
    assert set(mode.tolist()) == set(range(len(MODE_NAMES)))
    assert phase_code(0) == 0 and phase_code(119) == 0
    assert phase_code(120) == 1 and phase_code(239) == 1 and phase_code(240) == 2
    assert len(PHASE_NAMES) == 3


def test_full_action_model_and_horizon_action_normalization_contract():
    rng = np.random.default_rng(23)
    current = rng.normal(size=(5, 11, 3)).astype("float32")
    future = current[:, None] + rng.normal(size=(5, HORIZON, 11, 3)).astype("float32")
    hand = trajectory_input(current, future)
    target = rng.uniform(-1, 1, size=(5, HORIZON, ACTION_DIM)).astype("float32")
    action_stats = Standardizer.fit(target[:3])
    hand_stats = Standardizer.fit(hand[:3])
    np.testing.assert_allclose(action_stats.decode(action_stats.encode(target)), target, atol=2e-6)
    assert action_stats.mean.shape == (HORIZON, ACTION_DIM)
    model = HandActionRetargeter(32)
    output = model(torch.from_numpy(hand_stats.encode(hand)), torch.zeros(5, 36))
    assert output.shape == (5, HORIZON, ACTION_DIM)
    assert torch.isfinite(output).all()


def test_receding_chunk_offset_is_relative_to_latest_query():
    assert chunk_offset(0, 0) == 0
    assert chunk_offset(23, 0) == 23
    assert chunk_offset(24, 24) == 0
    with np.testing.assert_raises(ValueError):
        chunk_offset(24, 0)


def test_context_contract_keeps_query_geometry_and_previous_command_local():
    rng = np.random.default_rng(24)
    current = rng.normal(size=(5, 11, 3)).astype("float32")
    pose = np.broadcast_to(np.eye(4, dtype="float32"), (5, 4, 4)).copy()
    pose[:, :3, 3] = rng.normal(size=(5, 3)).astype("float32")
    previous = rng.uniform(-1, 1, size=(5, ACTION_DIM)).astype("float32")
    context = hand_object_context(current, pose, previous)
    assert CONTEXT_SCHEMA.endswith("v2") and context.shape == (5, CONTEXT_DIM)
    model = ContextHandActionRetargeter(32)
    hand = torch.zeros(5, HORIZON, 11, 3)
    state = torch.zeros(5, 36)
    output = model(hand, state, torch.from_numpy(context))
    assert output.shape == (5, HORIZON, ACTION_DIM)
    assert torch.isfinite(output).all()


def test_broadcast_and_per_env_future_anchor_shapes_are_distinct_and_causal():
    rng = np.random.default_rng(26)
    source_hands = rng.normal(size=(543, 4, 11, 3)).astype("float32")
    source_hand = source_hands[:, 0]
    live_current = rng.normal(size=(4, 11, 3)).astype("float32")
    source_future, broadcast = _runner.anchored_future_batch(
        source_hands, source_hand, live_current, 48, False)
    assert source_future.shape == (HORIZON, 11, 3)
    assert broadcast.shape == (4, HORIZON, 11, 3)
    np.testing.assert_allclose(
        broadcast[2], live_current[2] + source_future - source_hand[48], atol=1e-6)
    per_env_source, per_env = _runner.anchored_future_batch(
        source_hands, source_hand, live_current, 48, True)
    assert per_env_source.shape == (4, HORIZON, 11, 3)
    assert per_env.shape == (4, HORIZON, 11, 3)
    np.testing.assert_allclose(
        per_env[2], live_current[2] + per_env_source[2] - source_hands[48, 2], atol=1e-6)


def test_trajectory_utility_separates_tau_and_object_effect_arms():
    rng = np.random.default_rng(25)
    history = torch.zeros(4, 1442)
    tau = torch.from_numpy(rng.normal(size=(4, HORIZON, TRAJECTORY_DIM)).astype("float32"))
    effect = torch.from_numpy(rng.normal(size=(4, HORIZON, OBJECT_EFFECT_DIM)).astype("float32"))
    model = TrajectoryUtility(width=32, layers=1)
    c0 = model(history, tau, effect, False)
    c1 = model(history, tau, effect, True)
    assert TRAJECTORY_SCHEMA.endswith("trajectory-utility.v1")
    assert c0.shape == c1.shape == (4,)
    assert torch.isfinite(c0).all() and torch.isfinite(c1).all()
    assert not torch.allclose(c0, c1)
