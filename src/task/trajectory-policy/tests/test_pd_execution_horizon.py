import pytest
import torch

from trajectory_policy.pd_retargeter import PDRetargeter, SCHEMA, load_model


@pytest.mark.parametrize('horizon', [1, 4, 8, 16, 24])
def test_horizon_controls_information_and_gradient(horizon):
    torch.manual_seed(1)
    model = PDRetargeter(16, execution_horizon=horizon)
    torch.nn.init.normal_(model.net[-1].weight)
    model.tau_mean.fill_(.3)
    model.tau_scale.fill_(.2)
    state = torch.randn(3, 87)
    tau = torch.randn(3, 24, 33, requires_grad=True)
    baseline = model(state, tau)
    changed = tau.detach().clone()
    changed[:, horizon:] += 100
    assert torch.equal(baseline, model(state, changed))
    baseline.sum().backward()
    assert tau.grad[:, :horizon].abs().sum() > 0
    assert tau.grad[:, horizon:].abs().sum() == 0
    packet = dict(schema=SCHEMA, width=16, use_tau=True,
                  execution_horizon=horizon, model=model.state_dict())
    loaded = load_model(packet, 'cpu')
    assert loaded.execution_horizon == horizon
    assert torch.equal(baseline, loaded(state, tau))
    packet.pop('execution_horizon')
    assert load_model(packet, 'cpu').execution_horizon == 24


@pytest.mark.parametrize('value', [0, 25, 4.5, True])
def test_invalid_horizon_rejected(value):
    with pytest.raises(ValueError):
        PDRetargeter(execution_horizon=value)
