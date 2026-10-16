import torch
from wam_rl_policy.bc_mlp import CurrentMLP,MLPController


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


def test_current_controller_exposes_hidden_without_changing_bc_command():
    torch.manual_seed(11);model=CurrentMLP().eval().requires_grad_(False)
    with torch.no_grad():model.net[-1].weight.normal_(std=.02);model.net[-1].bias.normal_(std=.02)
    controller=MLPController.__new__(MLPController);controller.model=model;controller.hidden_model=model.net[:-1].eval()
    x=torch.randn(5,300)*4
    torch.testing.assert_close(controller.act({},x,1),model(x).clamp(-1,1),rtol=0,atol=0)
    assert controller.last_features.shape==(5,512) and not controller.last_features.requires_grad
