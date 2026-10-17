import pytest
import torch

from trajectory_policy.tau_executor import (
    ACTION_DIM,
    HAND_PLAN_DIM,
    STATE_DIM,
    MeasuredStateTauPolicy,
    expand_active_action,
    validate_batch,
)


def test_policy_emits_bounded_active_action_and_expands_once():
    policy = MeasuredStateTauPolicy(horizon=8, width=32)
    state = torch.zeros(3, STATE_DIM)
    tau = torch.zeros(3, 8, HAND_PLAN_DIM)
    previous = torch.zeros(3, ACTION_DIM)

    active = policy(state, tau, previous)
    native = expand_active_action(active)

    assert active.shape == (3, ACTION_DIM)
    assert native.shape == (3, 18)
    assert torch.all(active.abs() <= 1)
    assert torch.count_nonzero(native[:, [7, 9, 11, 13, 16, 17]]) == 0


def test_contract_rejects_nonfinite_or_wrong_horizon():
    policy = MeasuredStateTauPolicy(horizon=8, width=32)
    state = torch.zeros(2, STATE_DIM)
    previous = torch.zeros(2, ACTION_DIM)
    with pytest.raises(ValueError):
        policy(state, torch.zeros(2, 7, HAND_PLAN_DIM), previous)
    with pytest.raises(ValueError):
        validate_batch(state, torch.full((2, 8, HAND_PLAN_DIM), float("nan")), previous)

