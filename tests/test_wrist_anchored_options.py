import math
import torch
from src.task.CmResidual.executable_contact_options import hold_target
from src.task.CmResidual.wrist_anchored_options import wrist_anchored_action
from tests.test_executable_contact_options import native_targets


def test_anchored_wrist_keeps_goal_while_feedback_fingers_change():
    scale=torch.ones(18);scale[3:6]=math.pi;offset=torch.zeros(18)
    initial=torch.rand(4,18)*.2;anchor=hold_target(initial,offset,scale)
    later=initial.clone();later[:,:6]+=.05
    expert=torch.full_like(initial,-.4)
    raw=wrist_anchored_action(expert,anchor,later,offset,scale)
    target=native_targets(raw,later,offset,scale)
    assert torch.allclose(target[:,:6],anchor[:,:6],atol=1e-6)
    assert torch.equal(raw[:,6:],expert[:,6:])
    next_expert=expert.clone();next_expert[:,6:]=.6
    next_raw=wrist_anchored_action(next_expert,anchor,later,offset,scale)
    next_target=native_targets(next_raw,later,offset,scale)
    assert torch.allclose(next_target[:,:6],anchor[:,:6],atol=1e-6)
    assert (next_target[:,[6,8,10,12,14,15]]>target[:,[6,8,10,12,14,15]]).all()
