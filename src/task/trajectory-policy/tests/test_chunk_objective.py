import numpy as np
import torch

from trajectory_policy.chunk_objective import control_aware_loss,independent_PD_scale,development_key
from trajectory_policy.pd_retargeter import pd_from_action
from trajectory_policy.retargeter_data import ACTIVE


def test_physical_error_matches_native_targets_including_finger_half_scale():
    torch.manual_seed(17)
    current_q=torch.randn(4,18);offset=torch.randn(18);scale=torch.rand(18)+.1
    a,b=torch.randn(4,18),torch.randn(4,18)
    actual=(pd_from_action(a,current_q,offset,scale)-pd_from_action(b,current_q,offset,scale))[:,list(ACTIVE)]
    expected=(a-b)[:,list(ACTIVE)]*independent_PD_scale(scale[list(ACTIVE)])
    torch.testing.assert_close(actual,expected)


def test_executed_first_error_is_more_costly_and_raw_out_of_bounds_has_gradient():
    target=torch.zeros(2,8,12);prediction=target.clone().requires_grad_(True)
    with torch.no_grad():prediction[0,0,0]=2;prediction[1,1,0]=2
    action_scale=torch.ones(8,12);pd_scale=torch.ones(12)
    first,_=control_aware_loss(prediction[:1],target[:1],action_scale,pd_scale)
    later,_=control_aware_loss(prediction[1:],target[1:],action_scale,pd_scale)
    assert first>later*4
    first.backward()
    assert prediction.grad[0,0,0]>0  # clipping in the loss would kill this correction


def test_checkpoint_selection_prioritizes_first_PD_over_chunk_average():
    target=np.zeros((20,8,12),np.float32)
    first_precise=target.copy();first_precise[:,1:]=.5
    chunk_precise=target.copy();chunk_precise[:,0,:3]=.1
    norm=np.ones((8,12),np.float32);physical=np.ones(12,np.float32)
    a=development_key(first_precise,target,norm,physical)
    b=development_key(chunk_precise,target,norm,physical)
    assert a[1]>b[1] and a<b
