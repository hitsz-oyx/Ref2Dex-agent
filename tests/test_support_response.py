import torch
from src.task.CmResidual.support_response import assignments,placement_offsets,primitive_target,support_features

def test_private_balanced_assignment_and_independent_placement():
    motion=torch.arange(768)%3;a=assignments(motion,519)
    for m in range(3):assert torch.equal(torch.bincount(a[motion==m],minlength=8),torch.full((8,),32))
    before=torch.get_rng_state().clone();xy=placement_offsets(768,519)
    assert torch.equal(before,torch.get_rng_state()) and torch.all(xy.abs()<=.01)
    assert torch.equal(a,assignments(motion,519))

def test_primitive_changes_only_requested_native_targets():
    base=torch.ones(2,18)*.5;lo=torch.zeros(18);hi=torch.ones(18)*1.6
    parameters=torch.tensor([[.01,0.,0.],[0.,0.,-.30]])
    goal=primitive_target(base,parameters,lo,hi)
    torch.testing.assert_close(goal[0,:6],torch.tensor([.51,.5,.5,.5,.5,.5]))
    torch.testing.assert_close(goal[1,6],torch.tensor(.2));torch.testing.assert_close(goal[1,7],torch.tensor(.21))
    torch.testing.assert_close(goal[:,14],base[:,14])

def test_current_features_preserve_relative_velocity_and_plan_without_actions():
    context=torch.zeros(1,70);context[:,0]=1;context[:,18]=.1;context[:,36]=1.02;context[:,43]=.2;context[:,51]=1.01
    result=support_features(context,torch.zeros(1),torch.zeros(1))
    assert result.shape==(1,69)
    torch.testing.assert_close(result[:,33],torch.tensor([.02]));torch.testing.assert_close(result[:,40],torch.tensor([.1]))
    torch.testing.assert_close(result[:,48],torch.tensor([.01]))
