import torch
from wam_rl_policy.bc_mlp import CurrentMLP


def test_current_reference_normalization_and_roundtrip():
    model=CurrentMLP()
    with torch.no_grad():
        model.x_mean.fill_(.3);model.x_scale.fill_(2.)
        model.y_mean.copy_(torch.linspace(-.2,.2,24));model.y_scale.fill_(.4)
    x=torch.randn(3,300)*10
    torch.testing.assert_close(model(x),model.y_mean.expand(3,-1))
    loss=model(x).square().mean();loss.backward()
    assert model.net[-1].weight.grad.abs().sum()>0
    assert model.x_mean.grad is None and model.y_scale.grad is None
    restored=CurrentMLP();restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(x),restored(x))
