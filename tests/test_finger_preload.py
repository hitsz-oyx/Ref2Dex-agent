import torch
from src.task.CmResidual.finger_preload import assign_doses,preload_target


def test_dependent_limit_constrains_parent_and_wrist_is_continuous():
    base=torch.zeros(2,18);base[:,5]=-4.59;base[:,6]=.4;base[:,7]=.42
    lo=torch.zeros(18);hi=torch.ones(18)*2;hi[7]=.525
    lo[:6]=-3.14;hi[:6]=3.14
    target=preload_target(base,torch.tensor([.3,.05]),lo,hi)
    assert torch.allclose(target[:,6],torch.tensor([.5,.45]))
    assert torch.allclose(target[:,7],torch.tensor([.525,.4725]))
    assert torch.equal(target[:,:6],base[:,:6])


def test_random_assignment_balanced_in_each_motion_and_reproducible():
    motion=torch.arange(3).repeat_interleave(32)
    a=assign_doses(motion,504);b=assign_doses(motion,505)
    assert torch.equal(a,assign_doses(motion,504)) and not torch.equal(a,b)
    for m in range(3):assert torch.equal(torch.bincount(a[motion==m]),torch.full((4,),8))
