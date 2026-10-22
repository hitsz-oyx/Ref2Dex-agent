from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from wam_rl_policy_v3.evaluation_sources import continuous_hold,absent_tau,FixedGraspTemplate,template_intent,gt_future
from wam_rl_policy_v3.pose_consumer import pose7


def test_disjoint_progress_does_not_add_to_success_and_current_state_is_not_future():
    times=np.arange(43)/30;held=np.r_[np.ones(20,bool),False,np.ones(22,bool)]
    assert held.sum()/30>1
    result=continuous_hold(times,held)
    assert not result['ever_success'] and not result['final_success']
    assert abs(result['longest_hold_s']-.7)<1e-10
    assert continuous_hold(np.arange(31)/30,np.ones(31,bool))['final_success']
    assert not continuous_hold(np.arange(30)/30,np.ones(30,bool))['ever_success']


def test_gap_or_final_drop_cannot_certify_continuous_final_hold():
    r=continuous_hold(np.r_[np.arange(31)/30,2.],np.ones(32,bool))
    assert r['ever_success'] and not r['final_success']
    r=continuous_hold(np.arange(32)/30,np.r_[np.ones(31,bool),False])
    assert r['ever_success'] and not r['final_success']
    with pytest.raises(ValueError):continuous_hold([0,0],[True,True])


def test_zero_content_and_masked_future_are_explicit_distinct_arms():
    for available in (True,False):
        zero,valid=absent_tau('zero-valid',available)
        masked,mvalid=absent_tau('no-tau',available)
        assert not zero.any() and not masked.any() and valid==available and not mvalid


def test_template_goal_only_depends_on_fixed_geometry_live_measurement_and_task():
    pose=np.eye(4);pose[:3,3]=[.1,.2,.3];q=np.zeros(18);palm=np.eye(4)
    template=FixedGraspTemplate(np.eye(4),np.eye(4),np.ones(6)*.2,np.ones(6)*.4)
    goal=np.eye(4);goal[2,3]=.8
    intent=template_intent(template,q,palm,pose,goal,'close')
    np.testing.assert_allclose(intent.grasp_pose_object,np.eye(4))
    np.testing.assert_allclose(intent.postgrasp_object_pose,goal)
    np.testing.assert_allclose(intent.preshape_q,.4)
    assert intent.mode=='fixed-template' and not intent.tau_valid


def test_gt_reuses_s2d_24step_mapping_and_roundtrip_matches_first_eight_world_deltas():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/audit'))
    from measure_s2d_controls import object_anchored_targets
    ref=dict(hand=np.arange(50*33).reshape(50,1,11,3)*.001,objects=np.zeros((50,1,13)))
    ref['objects'][:,:,6]=1;live=np.zeros(13);live[6]=1;live[:3]=[.4,-.2,.1]
    calls=[]
    def actual(*args,**kwargs):calls.append(kwargs['horizon']);return object_anchored_targets(*args,**kwargs)
    future,valid=gt_future(actual,ref,3,live,True)
    assert valid and calls==[24]
    np.testing.assert_allclose(future[0],ref['hand'][4,0]+live[:3])
    hand=np.ones((11,3))*.3;tau=(future[:8]-hand).astype(np.float32)
    np.testing.assert_allclose(tau+hand,future[:8],atol=1e-7)
    assert not gt_future(actual,ref,26,live,True)[1]
