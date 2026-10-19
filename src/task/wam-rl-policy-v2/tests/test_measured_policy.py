import torch
from wam_rl_policy_v2.measured_policy import MeasuredInspirePolicy


def test_shape_encoder_learns_jointly_without_changing_shadow_actor_capacity():
    torch.set_num_threads(2);model=MeasuredInspirePolicy()
    assert model.actor.hidden_dims==(1024,1024,512,512)
    with torch.no_grad():model.actor.net[-1].weight.normal_(std=.01)
    measured=torch.randn(2,347);task=torch.randn(2,10);points=torch.randn(2,32,3)*.03
    actions=model(measured,task,points);assert actions.shape==(2,12)
    actions.square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.shape.parameters())
