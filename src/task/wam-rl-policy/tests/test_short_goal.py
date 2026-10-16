import torch
from wam_rl_policy.short_goal import (ShortGoal,rotation_from_columns,point_displacement,
                                     normalized_inputs)


def test_rotation_is_proper_even_for_degenerate_columns():
    torch.manual_seed(3)
    c=torch.cat((torch.randn(50,6),torch.zeros(1,6),torch.tensor([[1.,0,0,1,0,0]])))
    r=rotation_from_columns(c)
    torch.testing.assert_close(r.transpose(-1,-2)@r,torch.eye(3).expand_as(r),atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(torch.linalg.det(r),torch.ones(len(r)),atol=1e-5,rtol=1e-5)


def test_rigid_point_convention_and_identity_initialization():
    r=torch.tensor([[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]])
    points=torch.tensor([[[1.,0.,0.],[0.,1.,0.]]]);t=torch.tensor([[.1,.2,.3]])
    torch.testing.assert_close(point_displacement(points,r,t),torch.tensor([[[-.9,1.2,.3],[-.9,-.8,.3]]]))
    m=ShortGoal(8,24);state=torch.randn(2,8);tokens=torch.randn(2,12,24);valid=torch.ones(2,12,dtype=torch.bool)
    out=m(state,tokens,valid,torch.randn(2,32,3))
    torch.testing.assert_close(out['delta'],torch.zeros(2,32,3))
    out['delta'].sum().backward()
    assert torch.isfinite(m.pose.weight.grad).all() and m.pose.weight.grad.abs().sum()>0


def test_statistics_ignore_heldout_objects_and_padding():
    fit=torch.tensor([True,True,False]);valid=torch.tensor([[True,False],[True,True],[True,True]])
    state=torch.randn(3,8);token=torch.randn(3,2,24)
    _,_,a=normalized_inputs(state,token,fit,mean=False,valid=valid)
    state[-1]=10000;token[-1]=10000;token[0,1]=10000
    _,_,b=normalized_inputs(state,token,fit,mean=False,valid=valid)
    for key in a:torch.testing.assert_close(a[key],b[key])
