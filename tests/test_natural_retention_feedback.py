import torch
from src.task.CmResidual.natural_retention_feedback import NaturalRetentionFeedback,groups


def test_slip_action_needs_observed_acquisition_and_latches_actual_wrist():
    f = NaturalRetentionFeedback(torch.arange(4),torch.zeros(4),torch.full((4,),2),torch.zeros(4,dtype=torch.long))
    lower,upper = torch.full((18,),-5.),torch.full((18,),5.)
    base,q,dq = torch.zeros((4,18)),torch.zeros((4,18)),torch.zeros((4,18))
    obj = torch.zeros((4,13));obj[:,2] = .05;obj[:,9] = -.04
    clear = torch.full((4,),.04)
    for tick in range(6):
        q[:,:3] = tick*.001
        goal = f.target(base,q,dq,obj,clear,torch.full((4,),tick),lower,upper)
        assert (f.event_tick == -1).all()
        assert goal[2,6] == 0 and torch.equal(goal[3,:3],base[3,:3])
    q[:,:3] = .006
    goal = f.target(base,q,dq,obj,clear,torch.full((4,),6),lower,upper)
    assert torch.equal(f.event_tick,torch.full((4,),6))
    assert abs(float(goal[2,6])-.15) < 1e-6
    assert torch.equal(goal[3,:3],q[3,:3])
    # Later loss and wrist movement must neither erase nor relatch the first event.
    q[:,:3] = .02;obj[:,2] = 0;obj[:,9] = .1
    goal = f.target(base,q,dq,obj,clear,torch.full((4,),7),lower,upper)
    assert torch.equal(f.event_tick,torch.full((4,),6))
    assert torch.equal(goal[3,:3],torch.full((3,),.006))
    assert torch.equal(goal[0],base[0])


def test_assignment_is_balanced_and_preserves_global_rng():
    motion = torch.arange(3).repeat_interleave(256)
    rng = torch.get_rng_state().clone()
    first = groups(motion,543)
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.equal(first,groups(motion,543))
    assert not torch.equal(first,groups(motion,544))
    for m in range(3):
        assert torch.equal(torch.bincount(first[motion==m]),torch.full((4,),64))
