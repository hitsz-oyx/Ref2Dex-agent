import math
import torch
from src.task.CmResidual.executable_contact_options import hold_target
from src.task.CmResidual.orientation_anchored_options import orientation_anchored_action
from tests.test_executable_contact_options import native_targets


def test_rotation_goal_survives_motion_without_stopping_lift_or_fingers():
    scale=torch.ones(18);scale[3:6]=math.pi;offset=torch.zeros(18)
    initial=torch.rand(4,18)*.2;anchor=hold_target(initial,offset,scale)
    moved=initial.clone();moved[:,:6]+=.05
    feedback=torch.full_like(moved,.3);feedback[:,2]=.01
    raw=orientation_anchored_action(feedback,anchor,moved,offset,scale)
    target=native_targets(raw,moved,offset,scale)
    assert torch.equal(raw[:,:3],feedback[:,:3])
    assert torch.equal(raw[:,6:],feedback[:,6:])
    assert torch.allclose(target[:,3:6],anchor[:,3:6],atol=1e-6)
    assert (target[:,2]>moved[:,2]).all()
