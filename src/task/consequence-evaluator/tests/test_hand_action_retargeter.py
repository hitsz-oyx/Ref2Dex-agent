import json
import numpy as np
import torch
import importlib.util
from pathlib import Path

from consequence_evaluator.hand_action_retargeter import (
    ACTION_DIM, CONTACT_CONTEXT_DIM, CONTACT_CONTEXT_FIELDS, CONTACT_CONTEXT_SCHEMA,
    CONTEXT_DIM, CONTEXT_SCHEMA, HORIZON, ContextHandActionRetargeter,
    HandActionRetargeter, Standardizer, chunk_offset, contact_context_features,
    hand_object_context, trajectory_input)
from consequence_evaluator.fixed_wrist_decoder import (
    pd_inverse_wrist_action, recover_wrist_sequence, replace_wrist_action,
    root_template)
from consequence_evaluator.retarget_collection import (
    ACTIVE_FINGERS, MODE_NAMES, PHASE_NAMES, finger_pulse_residual,
    phase_code, sample_structured_residual, validate_residual_family)
from consequence_evaluator.trajectory_utility import (
    OBJECT_EFFECT_DIM, SCHEMA as TRAJECTORY_SCHEMA, TRAJECTORY_DIM,
    TrajectoryUtility)

_runner_spec = importlib.util.spec_from_file_location(
    "hand_action_retarget_runner",
    Path(__file__).resolve().parents[1] / "tools/run/run_hand_action_retargeter.py")
_runner = importlib.util.module_from_spec(_runner_spec)
_runner_spec.loader.exec_module(_runner)

_train_spec = importlib.util.spec_from_file_location(
    "hand_action_retarget_fit", Path(__file__).resolve().parents[1] /
    "tools/run/train_hand_action_retargeter.py")
_train = importlib.util.module_from_spec(_train_spec)
_train_spec.loader.exec_module(_train)


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


def test_serial_finger_pulse_is_one_independent_bounded_channel():
    residual = finger_pulse_residual(3, int(ACTIVE_FINGERS[0]), .08)
    assert residual.shape == (3, 18)
    assert np.flatnonzero(residual[0]).tolist() == [int(ACTIVE_FINGERS[0])]
    np.testing.assert_array_equal(residual[:, int(ACTIVE_FINGERS[0])],
                                  np.full(3, .08, dtype="float32"))
    assert validate_residual_family(residual[None])
    with np.testing.assert_raises(ValueError):
        finger_pulse_residual(1, 7, .08)
    with np.testing.assert_raises(ValueError):
        finger_pulse_residual(1, int(ACTIVE_FINGERS[0]), .13)


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


def test_fixed_wrist_decoder_uses_only_future_geometry_and_live_state():
    rng = np.random.default_rng(242)
    source_q = rng.normal(size=18).astype("float32")
    source_hand = rng.normal(size=(11, 3)).astype("float32")
    template = root_template(source_hand, source_q)
    current_q = rng.normal(size=(3, 18)).astype("float32")
    current_dq = rng.normal(size=(3, 18)).astype("float32")
    future = rng.normal(size=(3, HORIZON, 11, 3)).astype("float32")
    goals = recover_wrist_sequence(future, template, current_q)
    assert goals.shape == (3, HORIZON, 18)
    coefficients = np.ones((6, 2), dtype="float32")
    action = pd_inverse_wrist_action(current_q, current_dq, goals[:, 0], coefficients)
    assert action.shape == (3, ACTION_DIM)
    assert np.isfinite(action).all()
    np.testing.assert_array_equal(action[:, 6:], np.zeros((3, ACTION_DIM - 6), dtype="float32"))
    with np.testing.assert_raises(ValueError):
        recover_wrist_sequence(future, template, current_q[:, :6])

    model_action = rng.uniform(-1., 1., size=(3, ACTION_DIM)).astype("float32")
    hybrid = replace_wrist_action(model_action, current_q, current_dq, goals[:, 0], coefficients)
    np.testing.assert_allclose(hybrid[:, 6:], model_action[:, 6:], atol=0, rtol=0)
    changed_model = model_action.copy(); changed_model[:, :6] = rng.normal(size=(3, 6))
    np.testing.assert_allclose(
        replace_wrist_action(changed_model, current_q, current_dq, goals[:, 0], coefficients)[:, :6],
        hybrid[:, :6], atol=0, rtol=0)
    changed_fingers = model_action.copy(); changed_fingers[:, 6:] = rng.normal(size=(3, ACTION_DIM - 6))
    assert not np.array_equal(
        replace_wrist_action(changed_fingers, current_q, current_dq, goals[:, 0], coefficients)[:, 6:],
        hybrid[:, 6:])


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


def test_contact_context_is_query_time_only_and_matches_v2_zero_branch():
    rng = np.random.default_rng(241)
    pair = np.asarray([True, False, True, False, True])
    footprint = np.asarray([False, True, False, True, False])
    gap = rng.uniform(0., 1., size=5).astype("float32")
    support = rng.uniform(0., 1., size=5).astype("float32")
    velocity = rng.normal(size=(5, 6)).astype("float32")
    contact = contact_context_features(pair, gap, support, velocity, footprint)
    assert contact.shape == (5, CONTACT_CONTEXT_DIM)
    assert CONTACT_CONTEXT_SCHEMA.endswith("v3")
    assert CONTACT_CONTEXT_FIELDS[0] == "pair" and CONTACT_CONTEXT_FIELDS[-1] == "table_footprint"
    np.testing.assert_array_equal(contact[:, 0], pair.astype("float32"))
    np.testing.assert_array_equal(contact[:, -1], footprint.astype("float32"))

    base = ContextHandActionRetargeter(32)
    augmented = ContextHandActionRetargeter(32, contact_dim=CONTACT_CONTEXT_DIM)
    missing, unexpected = augmented.load_state_dict(base.state_dict(), strict=False)
    assert set(missing) == {"contact.weight", "contact.bias"} and not unexpected
    hand = torch.zeros(5, HORIZON, 11, 3)
    state = torch.zeros(5, 36)
    context = torch.zeros(5, CONTEXT_DIM)
    base_output = base(hand, state, context)
    augmented_output = augmented(hand, state, context, torch.zeros(5, CONTACT_CONTEXT_DIM))
    torch.testing.assert_close(base_output, augmented_output, rtol=0, atol=0)

    with np.testing.assert_raises(ValueError):
        contact_context_features(pair.astype("float32"), gap, support, velocity, footprint)
    with np.testing.assert_raises(ValueError):
        augmented(hand, state, context)


def test_contact_loader_uses_query_tick_and_skips_reset_frame(tmp_path):
    root = tmp_path / "structured"
    root.mkdir()
    frames, envs = 30, 1
    action = np.zeros((frames - 1, envs, ACTION_DIM), dtype="float32")
    for tick in range(frames - 1):
        action[tick, 0] = tick / 40.
    pair = np.zeros((frames, envs), dtype=bool)
    pair[1, 0] = True
    pair[2, 0] = False
    arrays = dict(
        hand_keypoints=np.zeros((frames, envs, 11, 3), dtype="float32"),
        object_pose=np.tile(np.eye(4, dtype="float32"), (frames, envs, 1, 1)),
        dof_position=np.zeros((frames, envs, ACTION_DIM), dtype="float32"),
        dof_velocity=np.zeros((frames, envs, ACTION_DIM), dtype="float32"),
        action=action, done=np.zeros((frames - 1, envs), dtype=bool),
        length=np.asarray([frames - 1], dtype=np.int64),
        structured_residual=np.zeros_like(action), actor_action=np.zeros_like(action),
        structured_mode=np.zeros((frames - 1, envs), dtype=np.int8),
        structured_phase=np.zeros(frames - 1, dtype=np.int8),
        pair=pair, surface_gap=np.arange(frames, dtype="float32")[:, None],
        support_gap=np.zeros((frames, envs), dtype="float32"),
        object_velocity=np.zeros((frames, envs, 6), dtype="float32"),
        table_footprint=np.zeros((frames, envs), dtype=bool),
    )
    np.savez_compressed(root / "trajectory.npz", **arrays)
    (root / "manifest.json").write_text(json.dumps(dict(
        status="COMPLETED", mode="retarget",
        structured_residual_schema="ref2dex.structured-residual.v1", seed=777)))
    windows, _, _ = _train.load_windows(root, stride=1, include_contact=True, skip_reset=True)
    assert windows["tick"][0] == 1
    assert windows["contact"][0, 0] == 1.
    assert windows["contact"][0, 1] == 1.
    np.testing.assert_array_equal(windows["action"][0, 0], action[1, 0])
    np.testing.assert_array_equal(windows["action"][0, 1], action[2, 0])
    assert not np.any(windows["tick"] == 0)


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


def test_teacher_finger_reference_only_replaces_test_env_fingers():
    command = np.arange(4 * 18, dtype="float32").reshape(4, 18)
    teacher = np.linspace(-1., 1., 18, dtype="float32")
    result = _runner.replace_with_teacher_fingers(command, teacher)
    np.testing.assert_array_equal(result[0], command[0])
    np.testing.assert_array_equal(result[:, :6], command[:, :6])
    np.testing.assert_array_equal(result[1:, 6:], np.broadcast_to(teacher[6:], (3, 12)))
    np.testing.assert_array_equal(command, np.arange(4 * 18, dtype="float32").reshape(4, 18))
    with np.testing.assert_raises(ValueError):
        _runner.replace_with_teacher_fingers(command, teacher[:17])


def test_teacher_action_reference_replaces_only_test_env_rows():
    command = np.arange(4 * 18, dtype="float32").reshape(4, 18)
    teacher = np.linspace(-1., 1., 18, dtype="float32")
    result = _runner.replace_with_teacher_action(command, teacher)
    np.testing.assert_array_equal(result[0], command[0])
    np.testing.assert_array_equal(result[1:], np.broadcast_to(teacher, (3, 18)))
    np.testing.assert_array_equal(command, np.arange(4 * 18, dtype="float32").reshape(4, 18))
    with np.testing.assert_raises(ValueError):
        _runner.replace_with_teacher_action(command, teacher[:17])


def test_runtime_source_identity_reads_backend_and_direct_row_layout():
    class Physx:
        use_gpu = True

    class Params:
        physx = Physx()

    class Gym:
        @staticmethod
        def get_sim_params(sim):
            assert sim == "sim"
            return Params()

    class Task:
        gym = Gym()
        sim = "sim"
        device = "cpu"

    backend, actor = _runner.runtime_source_identity(Task(), 4)
    assert backend["tensor_device"] == "cpu"
    for key in _runner.BACKEND_CONTRACT_KEYS:
        assert backend[key] == _runner.SOURCE_BACKEND[key]
    assert actor == dict(_runner.SOURCE_ACTOR_EXECUTION, total_rows=4)


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
