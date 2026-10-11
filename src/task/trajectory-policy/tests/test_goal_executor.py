import pytest
import torch

from consequence_evaluator.reference_tracking import native_action, wrist_feedforward
from trajectory_policy.goal_executor import (
    EXECUTE_STEPS,
    GoalConditionedExecutor,
    receding_goal,
    validate_plan,
)


def _constants():
    offset = torch.zeros(18)
    scale = torch.ones(18)
    scale[6:] = .5
    return offset, scale, torch.full((6,), .1)


def test_receding_goal_consumes_future_prefix_and_rejects_open_loop_slot():
    q = torch.arange(2 * 25 * 18, dtype=torch.float32).reshape(2, 25, 18)
    velocity = q + 1000
    validate_plan(q, velocity)
    selected, selected_velocity = receding_goal(q, velocity, 3)
    torch.testing.assert_close(selected, q[:, 4])
    torch.testing.assert_close(selected_velocity, velocity[:, 4])
    with pytest.raises(ValueError):
        receding_goal(q, velocity, EXECUTE_STEPS)
    with pytest.raises(ValueError):
        receding_goal(q, velocity, 0, execute_steps=0)
    selected, _ = receding_goal(q, velocity, 3, execute_steps=4)
    torch.testing.assert_close(selected, q[:, 4])


def test_closed_loop_step_matches_existing_reference_tracker_contract():
    offset, scale, ratio = _constants()
    executor = GoalConditionedExecutor(offset, scale, ratio)
    current = torch.zeros(2, 18)
    reference = torch.randn(2, 18) * .02
    velocity = torch.randn(2, 18) * .01
    latent = torch.randn(2, 12)
    result = executor.step(current, reference, velocity, latent)

    base = wrist_feedforward(reference, velocity, ratio)
    target = base.clone()
    limits = torch.tensor((.06, .06, .06, .6, .6, .6, .5, .5, .5, .5, .35, .35))
    active = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15)
    target[:, list(active)] += torch.tanh(latent) * limits
    from consequence_evaluator.reference_tracking import apply_coupling
    target = apply_coupling(target)
    expected = native_action(target, current, offset, scale)
    torch.testing.assert_close(result['target'], target)
    torch.testing.assert_close(result['intended'], expected)
    torch.testing.assert_close(result['command'], expected.clamp(-1, 1))
    torch.testing.assert_close(result['clipped'], (expected.abs().amax(-1) > 1))


def test_receding_step_reads_current_q_each_time():
    offset, scale, ratio = _constants()
    executor = GoalConditionedExecutor(offset, scale, ratio)
    plan = torch.zeros(1, 25, 18)
    plan[:, 1:, 0] = .02
    velocity = torch.zeros_like(plan)
    first = executor.receding_step(torch.zeros(1, 18), plan, velocity, torch.zeros(1, 12), 0)
    second = executor.receding_step(torch.full((1, 18), .01), plan, velocity, torch.zeros(1, 12), 0)
    assert first['command'][0, 0] > second['command'][0, 0]
