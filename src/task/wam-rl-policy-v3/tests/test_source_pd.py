import numpy as np
import pytest
from wam_rl_policy_v3.source_pd import source_pd_action
from wam_rl_policy_v3.inspire_contract import expand_action


def fixture():
    scale=np.array([1,1,1,np.pi,np.pi,np.pi,1.6,3.14,1.6,3.14,1.6,3.14,1.6,3.14,1.15,.55,3.14,3.14])
    q=np.zeros((1,18));q[0,3:6]=[-.2,.3,.4]
    a=np.linspace(-.5,.5,12)[None];expanded=expand_action(a)
    pd=(expanded+1)/2*scale;pd[:,:6]=q[:,:6]+expanded[:,:6]*scale[:6]
    pd[:,[7,9,11,13]]=pd[:,[6,8,10,12]]*1.05
    pd[:,16]=pd[:,15]*.6;pd[:,17]=pd[:,15]*.8
    return q,pd,scale,a


def test_source_pd_inverse_uses_nearest_equivalent_continuous_target():
    q,pd,scale,a=fixture();pd[:,3:6]+=np.array([2,-3,4])*2*np.pi
    actual,target,roundoff=source_pd_action(pd,q,np.zeros(18),scale)
    np.testing.assert_allclose(actual,a,atol=2e-7)
    assert roundoff==0 and np.abs(target[:,3:6]-q[:,3:6]).max()<np.pi


def test_source_pd_refuses_saturation_and_nonrepresentable_coupled_fingers():
    q,pd,scale,a=fixture();pd[0,1]=q[0,1]+1.1
    with pytest.raises(ValueError,match='saturation'):source_pd_action(pd,q,np.zeros(18),scale)
    q,pd,scale,a=fixture();pd[0,7]+=.1
    with pytest.raises(ValueError,match='coupling'):source_pd_action(pd,q,np.zeros(18),scale)


def test_positive_control_duration_uses_seconds_not_thirty_state_occupancy():
    import importlib.util
    from pathlib import Path
    file=Path(__file__).resolve().parents[1]/'tools/audit/score_source_pd_positive_control.py'
    loader=importlib.util.spec_from_file_location('source_pd_scoring',file);module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    assert module.longest_seconds(np.ones(30,bool),np.arange(30)/30)<1.
    assert module.longest_seconds(np.ones(31,bool),np.arange(31)/30)==1.
    assert module.longest_seconds([True,True,False,True],[0.,.2,.4,.9])==.2
