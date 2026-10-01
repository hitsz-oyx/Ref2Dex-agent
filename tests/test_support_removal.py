import torch
from src.task.CmResidual.support_removal import witness_target,assign_witness_arms


def test_removal_only_after_phase_and_only_randomized_hand_arm():
    q=torch.zeros(2,18);q[:,2]=1.;q[:,3:6]=torch.tensor([.1,.2,-4.59]);q[:,14]=.4
    original=q.clone();lo=torch.zeros(18);hi=torch.ones(18)*1.6;hi[14]=1.15;hi[15]=.55
    args=(q,torch.tensor([140,140]),torch.tensor([140,140]),torch.ones(2,dtype=torch.long),torch.tensor([0,1]),torch.tensor([63,44,55]),torch.tensor([162,140,152]),lo,hi)
    before,mask=witness_target(*args)
    after_args=list(args);after_args[2]=torch.tensor([141,141])
    after,released=witness_target(*after_args)
    assert not mask.any() and torch.equal(released,torch.tensor([False,True]))
    assert torch.equal(before[0],before[1]) and torch.equal(after[0],before[0])
    assert torch.allclose(after[1,2],before[1,2]+.2)
    assert torch.equal(after[1,3:6],before[1,3:6]) and not after[1,6:].any()
    assert torch.equal(q,original)


def test_private_witness_assignments_balanced_before_any_outcome():
    motion=torch.arange(3).repeat_interleave(32);a=assign_witness_arms(motion,508)
    assert torch.equal(a,assign_witness_arms(motion,508))
    assert not torch.equal(a,assign_witness_arms(motion,509))
    for m in range(3):assert torch.equal(torch.bincount(a[motion==m]),torch.tensor([16,16]))
