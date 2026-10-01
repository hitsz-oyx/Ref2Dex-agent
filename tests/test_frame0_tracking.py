import torch
from src.task.CmResidual.frame0_tracking import tracking_target,assign_tracking_arms


def test_reference_q_not_mutated_and_ramp_keeps_native_coupled_bounds():
    q=torch.zeros(3,18);q[:,5]=-4.59;q[:,14]=1.3;q[:,15]=.54
    saved=q.clone();lo=torch.zeros(18);hi=torch.ones(18)*1.6;hi[14]=1.15;hi[15]=.55
    target,dose=tracking_target(q,torch.tensor([48,55,63]),torch.zeros(3,dtype=torch.long),torch.ones(3),torch.tensor([63,44,55]),lo,hi)
    assert torch.equal(q,saved) and torch.equal(target[:,:6],q[:,:6])
    assert torch.allclose(dose,torch.tensor([0.,.14,.30]))
    assert torch.allclose(target[:,15],torch.tensor([.54,.55,.55]))
    assert (target[:,14]<=1.15).all()


def test_two_arm_private_assignment_keeps_every_motion_matched():
    motion=torch.arange(3).repeat_interleave(32);assignment=assign_tracking_arms(motion,506)
    for m in range(3):assert torch.equal(torch.bincount(assignment[motion==m]),torch.tensor([16,16]))
