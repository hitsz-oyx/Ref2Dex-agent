import torch
from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy


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


def test_shared_cad_encoding_preserves_actions_and_all_parameter_gradients():
    import copy
    torch.manual_seed(17)
    expanded = MeasuredInspirePolicy(with_tau=True)
    with torch.no_grad():
        expanded.actor.net[-1].weight.normal_(std=.01)
    shared = copy.deepcopy(expanded)
    measured, task = torch.randn(8,347), torch.randn(8,10)
    points = torch.randn(3,32,3)*.03
    indices = torch.tensor([1,2,1,0,2,0,1,2])
    tau = torch.randn(8,8,11,3)*.03
    valid = torch.ones(8,dtype=torch.bool)
    a = expanded(measured,task,points[indices],tau=tau,tau_valid=valid)
    b = shared(measured,task,points,tau=tau,tau_valid=valid,shape_index=indices)
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    target = torch.randn_like(a)
    (a-target).square().mean().backward()
    (b-target).square().mean().backward()
    for old,new in zip(expanded.parameters(),shared.parameters()):
        torch.testing.assert_close(old.grad,new.grad,rtol=1e-4,atol=1e-6)
