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
