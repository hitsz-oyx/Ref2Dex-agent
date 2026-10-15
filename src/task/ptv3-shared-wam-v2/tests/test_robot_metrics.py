import torch
import pytest
from ptv3_shared_wam_v2.robot_metrics import physical_errors,summarize_errors


def test_known_first_control_xyz_rotation_fingers_and_mask():
    target=torch.zeros(2,8,15);action=target.clone()
    action[:,0,0]=.01;action[:,0,5]=.2;action[:,0,6:12]=.3
    p=dict(action=action,trajectory=torch.ones(2,24,2,11,3)*.01,scene=torch.zeros(2,1,24,3,3))
    b=dict(action_target=target,trajectory_valid=torch.ones(2,24,2,11,dtype=torch.bool),
        start_hand_valid=torch.ones(2,2,11,dtype=torch.bool),scene_valid=torch.ones(2,1,24,3,dtype=torch.bool),
        object_valid=torch.ones(2,1,dtype=torch.bool),trajectory_target=torch.zeros_like(p['trajectory']),
        scene_target=torch.zeros_like(p['scene']))
    e=physical_errors(p,b);r=summarize_errors({key:[value] for key,value in e.items()})
    assert r['PD_xyz_mm']['mean']==pytest.approx(10.)
    assert r['PD_rotation_rad']['p95']==pytest.approx(.2)
    assert r['PD_finger_rad']['mean']==pytest.approx(.3)
    assert r['hand_all24_mm']['mean']==pytest.approx(3**.5*10.)
    assert r['PD_xyz_mm']['count']==2


def test_native_and_geometry_labels_have_same_physical_first_target():
    import numpy as np
    from ptv3_shared_wam.action_targets import NATIVE,GEOMETRY_LIFTED,encode_targets
    from trajectory_policy.retargeter_data import ACTIVE
    from trajectory_policy.pd_retargeter import pd_from_action
    torch.manual_seed(6)
    q=torch.randn(2,18)*.1
    label=torch.randn(2,8,12)*.1
    controller=dict(offset=[0.]*18,scale=[.2]*18)
    full=torch.zeros(2,8,18);full[:,:,list(ACTIVE)]=label
    pd=pd_from_action(full,q[:,None].expand(-1,8,-1),q.new_zeros(18),q.new_full((18,),.2))
    native=label.clone();native[:,0,0]+=.01
    a=full.clone();a[:,0,0]+=.01
    desired=pd_from_action(a,q[:,None].expand(-1,8,-1),q.new_zeros(18),q.new_full((18,),.2))
    geometry=torch.from_numpy(encode_targets(desired.numpy(),q.numpy(),GEOMETRY_LIFTED))
    p=dict(action=native,trajectory=torch.zeros(2,24,2,11,3),scene=torch.zeros(2,1,24,3,3))
    b=dict(q_real=q[:,None].expand(-1,8,-1),q_pd=pd,native_action=label,
        action_target=torch.from_numpy(encode_targets(pd.numpy(),q.numpy(),GEOMETRY_LIFTED)),
        trajectory_valid=torch.ones(2,24,2,11,dtype=torch.bool),start_hand_valid=torch.ones(2,2,11,dtype=torch.bool),
        scene_valid=torch.ones(2,1,24,3,dtype=torch.bool),object_valid=torch.ones(2,1,dtype=torch.bool),
        trajectory_target=p['trajectory'],scene_target=p['scene'])
    first=physical_errors(p,b,controller,NATIVE)
    second=physical_errors(dict(p,action=geometry),b,controller,GEOMETRY_LIFTED)
    for key in ('PD_xyz_mm','PD_rotation_rad','PD_finger_rad','PD_euler_rad','command_first_mae','applied_first_mae'):
        np.testing.assert_allclose(first[key],second[key],atol=1e-6)
    np.testing.assert_allclose(first['PD_xyz_mm'],2.,atol=1e-5)
    # Independent rotvec/lift heads need not agree. Compare decoded Euler,
    # rather than calling the raw lift-head residual a physical joint error.
    from ptv3_shared_wam.action_targets import freeze_chunk,consume_target
    mismatch=geometry.clone();mismatch[:,0,3]+=.2;mismatch[:,0,12:15]=0
    q8=q[:,None].expand(-1,8,-1)
    intended,_,_=consume_target(freeze_chunk(mismatch,q,GEOMETRY_LIFTED),q8,GEOMETRY_LIFTED,controller)
    eg=physical_errors(dict(p,action=mismatch),b,controller,GEOMETRY_LIFTED)
    en=physical_errors(dict(p,action=intended[:,:,list(ACTIVE)]),b,controller,NATIVE)
    np.testing.assert_allclose(eg['PD_euler_rad'],en['PD_euler_rad'],atol=1e-6)
