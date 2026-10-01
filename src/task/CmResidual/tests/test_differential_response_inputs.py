import torch

from scripts.train_differential_response import window_features


def example():
    return dict(triggers=torch.tensor([5]), state_before=torch.zeros(20, 1, 55),
                state_after=torch.zeros(20, 1, 55), action=torch.zeros(20, 1, 18))


def test_future_physical_observations_never_enter_model_features():
    data = example()
    nominal = torch.zeros(1, 180)
    before = window_features(data, nominal)
    data['state_before'][6:] = 1000
    data['state_after'][:] = 1000
    after = window_features(data, nominal)
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_no_pulse_control_hides_only_the_assigned_pulse():
    data = example()
    nominal = torch.zeros(1, 180)
    full_before, blind_before = window_features(data, nominal)
    data['action'][5, 0, 2] = .01
    full_after, blind_after = window_features(data, nominal)
    assert not torch.equal(full_before, full_after)
    assert torch.equal(blind_before, blind_after)
    data['state_before'][5, 0, 38] = 2
    _, blind_with_history = window_features(data, nominal)
    assert not torch.equal(blind_before, blind_with_history)
