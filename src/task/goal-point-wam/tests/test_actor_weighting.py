import numpy as np
import torch
from goal_point_wam.actor_weighting import target_mismatch,weighted_trajectory_loss


def test_downweight_preserves_unmarked_gradients_and_does_not_renormalize():
    p=torch.full((2,24,2,11,3),.02,requires_grad=True)
    b=dict(trajectory_target=torch.zeros_like(p),trajectory_valid=torch.ones(p.shape[:-1],dtype=torch.bool))
    full=weighted_trajectory_loss(p,b,.05,torch.ones(2))
    g,=torch.autograd.grad(full,p,retain_graph=True)
    reduced=weighted_trajectory_loss(p,b,.05,torch.tensor([.1,1.]))
    h,=torch.autograd.grad(reduced,p)
    torch.testing.assert_close(h[0],g[0]*.1)
    torch.testing.assert_close(h[1],g[1])


def test_static_goal_alone_is_not_mismatch():
    rows=np.array([[1,1,20,10,150,140,140,20,10],
                   [1,1,20,10,20,10,10,10,10],
                   [1,1,1,1,150,140,140,20,10],
                   [1,8,20,10,150,140,140,20,10]],dtype=float)
    assert target_mismatch(rows).tolist()==[True,False,False,False]
