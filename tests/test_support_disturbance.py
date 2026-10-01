import torch
from src.task.CmResidual.support_disturbance import cells, recovery_parameters


def test_complete_cells_and_directional_recovery():
    motion = torch.arange(768)%3
    assigned = cells(motion,527)
    assert torch.equal(assigned,cells(motion,527))
    assert not torch.equal(assigned,cells(motion,528))
    for mo in range(3):
        assert torch.equal(torch.bincount(assigned[motion==mo],minlength=32),torch.full((32,),8))
    direction = ((assigned//4)%2)*2-1
    arm = assigned%4
    p = recovery_parameters(arm,direction)
    assert torch.all(p[arm==0]==0)
    assert torch.allclose(p[arm==1,0],direction[arm==1]*.02)
    assert torch.allclose(p[arm==2,0],-direction[arm==2]*.02)
    assert torch.all(p[:,1]==0)
    assert torch.allclose(p[arm==3,2],torch.full_like(p[arm==3,2],.15))
    assert torch.all(p[arm!=3,2]==0)
