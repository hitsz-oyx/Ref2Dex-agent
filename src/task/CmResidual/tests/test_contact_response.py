import pytest
import torch

from src.task.CmResidual.contact_response import pulse_action, sample_response, trigger_steps


def test_pulse_has_native_units_and_preserves_reference():
    reference = torch.zeros(3, 18)
    result = pulse_action(reference, torch.tensor([4, 5, 4]), 4, .01)
    assert torch.equal(reference, torch.zeros_like(reference))
    assert torch.allclose(result[:, 2], torch.tensor([.01, 0., .01]))
    assert result[:, torch.arange(18) != 2].count_nonzero() == 0


def test_pulse_refuses_silent_clipping():
    with pytest.raises(ValueError, match='clips'):
        pulse_action(torch.ones(1, 18), torch.tensor([0]), 0, .01)


def test_trigger_uses_next_step_and_requires_both_proxies():
    state = torch.zeros(40, 2, 55)
    state[1, 0, 49] = 1
    state[3, 0, 49:51] = 1
    state[5, 1, 49:51] = 1
    assert torch.equal(trigger_steps(state, torch.ones(40, 2, dtype=torch.bool)), torch.tensor([4, 6]))


def test_missing_contact_is_not_silently_frame_zero():
    with pytest.raises(ValueError, match='without'):
        trigger_steps(torch.zeros(40, 1, 55), torch.ones(40, 1, dtype=torch.bool))


def test_horizon_alignment_and_pre_state_subtraction():
    before = torch.zeros(40, 2, 55)
    after = torch.zeros_like(before)
    before[2, 0, 38] = .2
    before[4, 1, 38] = .4
    after[2, 0, 38] = .3
    after[4, 1, 38] = .7
    after[3, 0, 38] = .5
    after[5, 1, 38] = .9
    triggers = torch.tensor([2, 4])
    assert torch.allclose(sample_response(before, after, triggers, 1)[:, 2], torch.tensor([.1, .3]))
    assert torch.allclose(sample_response(before, after, triggers, 2)[:, 2], torch.tensor([.3, .5]))
