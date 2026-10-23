"""Independent finite-difference and deliberate-corruption checks for the audit."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from wam_rl_policy_v3.object_tracking import hand_setpoint

spec=importlib.util.spec_from_file_location('tracking_error_audit',Path(__file__).resolve().parents[1]/'tools/audit/audit_tracking_error.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

def test_signed_components_do_not_add_norms():
    desired=np.eye(4)[None];relative=np.eye(4);relative[0,3]=.1
    hand=np.eye(4)[None];hand[0,:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix()
    hand[0,:3,3]=[-.08,.02,.03]
    obj=hand@relative;obj[0,:3,3]+=[.001,-.002,.003]
    parts=audit.decompose(hand,obj,desired,relative)
    np.testing.assert_allclose(parts['hand_translation'],[[.02,.02,.03]],atol=1e-15)
    np.testing.assert_allclose(parts['rotation_lever'],[[-.1,.1,0]],atol=1e-15)
    np.testing.assert_allclose(parts['grasp'],[[.001,-.002,.003]],atol=1e-15)
    np.testing.assert_allclose(parts['closure'],0,atol=1e-15)

@pytest.mark.parametrize('mutant',['inverse_transform','one_frame','lever_velocity','scorer_shift'])
def test_deliberate_errors_are_detected(mutant):
    relative=np.eye(4);relative[:3,3]=[.13,.02,-.04]
    def pose(t):
        p=np.eye(4);p[:3,:3]=Rotation.from_rotvec([0,0,1.2*t]).as_matrix();p[:3,3]=[.1*t,0,1];return p
    t=.4;target,velocity,_=hand_setpoint(pose(t),[.1,0,0],[0,0,1.2],relative)
    eps=1e-6
    independent_velocity=((pose(t+eps)@np.linalg.inv(relative))[:3,3]-(pose(t-eps)@np.linalg.inv(relative))[:3,3])/(2*eps)
    audit.check_recorded(velocity,independent_velocity,1e-9,'valid derivative')
    audit.check_recorded(target@relative,pose(t),1e-12,'valid frame')
    if mutant=='inverse_transform':bad=pose(t)@relative;expected=target
    elif mutant=='one_frame':bad=pose(t+1/30)@np.linalg.inv(relative);expected=target
    elif mutant=='lever_velocity':bad=np.array([.1,0,0]);expected=independent_velocity
    else:bad=np.linalg.norm((pose(t+1/30)@np.linalg.inv(relative))[:3,3]);expected=np.linalg.norm(target[:3,3])
    with pytest.raises(AssertionError):audit.check_recorded(bad,expected,1e-9,mutant)

def test_gravity_fraction_uses_energy_not_mean_sign():
    result=audit.stats(np.array([[3.,4.,0.],[-3.,-4.,0.]]))
    assert result['vertical_power_fraction']==0
    assert result['rms_mm']==5000
    assert result['mean_xyz_mm']==[0.,0.,0.]
