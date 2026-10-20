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


def test_scheduled_expansion_preserves_trained_current_for_arbitrary_tau():
    torch.manual_seed(13)
    parent = MeasuredInspirePolicy().eval()
    with torch.no_grad():
        parent.actor.net[-1].weight.normal_(std=.02)
        parent.actor.x_mean.normal_(std=.1)
        parent.actor.x_scale.uniform_(.5, 2)
        parent.actor.y_mean.normal_(std=.1)
    scheduled = MeasuredInspirePolicy(with_tau=True).eval()
    scheduled.initialize_current(dict(schema='ref2dex.inspire-measured-current-bc.v1', model=parent.state_dict()))
    assert torch.count_nonzero(scheduled.actor.net[0].weight[:, 421:]) == 0
    measured, task, points = torch.randn(3,347), torch.randn(3,10), torch.randn(3,32,3)
    with torch.no_grad():
        expected = parent(measured, task, points)
        for valid in (torch.ones(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool)):
            actual = scheduled(measured, task, points, tau=torch.randn(3,8,11,3), tau_valid=valid)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
