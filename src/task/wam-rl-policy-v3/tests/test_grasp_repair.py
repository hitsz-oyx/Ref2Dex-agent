"""Regressions at the real R/intention/servo seams, with injected old failures."""
import numpy as np
from types import SimpleNamespace
from wam_rl_policy_v3.grasp_repair import clean_tau_future, GoalServo, ReachabilityFilter


def test_clean_ablation_changes_only_future_content_and_preserves_mask():
    hand=np.arange(33,dtype=np.float32).reshape(11,3)/100
    future=hand[None]+np.ones((24,11,3),np.float32)*.02
    original=future.copy()
    a,va=clean_tau_future(future,hand,True,zero=False)
    b,vb=clean_tau_future(future,hand,True,zero=True)
    assert va is vb is True
    np.testing.assert_array_equal(a,original)
    np.testing.assert_array_equal(b,np.broadcast_to(hand,(24,11,3)))
    np.testing.assert_array_equal(future,original)


def test_goal_servo_overcomes_saved_v9_saturated_drift_and_never_uses_reported_velocity():
    # V9 p006: goal-x=-.06083, x=1.1733, command=-.004m but hand drifts +.002m/tick.
    s=GoalServo();p=np.array([1.17,-1.83,1.03]);g=np.array([-.06,-.74,1.04])
    a=s.step(g,p,np.zeros(3),0.)
    assert a[0]<-.015 and a[1]>.015
    s1,s2=GoalServo(),GoalServo()
    for i in range(90):
        now=i/30
        r1=s1.step(np.array([.01,0.,0.]),np.zeros(3),np.zeros(3),now)
        r2=s2.step(np.array([.01,0.,0.]),np.zeros(3),np.zeros(3),now)
    assert r1[0]>.012 and np.array_equal(r1,r2)
    # Integral does not persist through a measured loss/reacquisition.
    s1.reset();assert s1.integral.max()==0


def test_reachability_rate_limits_collision_into_support_without_relaxing_gate():
    f=ReachabilityFilter();palm=np.eye(4);palm[2,3]=1.2
    obj=np.eye(4);obj[2,3]=1.
    q=np.zeros(18);q[6:]=.2;dq=np.zeros(18)
    target=palm.copy();target[2,3]=.2
    out,fingers,info=f.project(palm,target,obj,q,dq,np.ones(6),table_z=.9)
    assert np.linalg.norm(out[:3,3]-palm[:3,3])<=.025+1e-10
    assert info['candidate_count']>=3
    assert np.all(np.abs(fingers-q[[6,8,10,12,14,15]])<=.08+1e-10)
    # Object rotation changes candidate directions, not an initial-state label.
    obj[:3,:3]=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    other=ReachabilityFilter().project(palm,target,obj,q,dq,np.ones(6),table_z=.9)[2]
    assert not np.array_equal(info['directions_world'],other['directions_world'])


def test_healthy_R_PD_proposal_is_not_mistaken_for_instantaneous_kinematic_motion():
    # Learned PD offsets can extend below the table while the measured hand is
    # high and collision-free. Limiting every such offset destroyed v10-r1.
    palm=np.eye(4);palm[2,3]=1.4;target=palm.copy();target[2,3]=.7
    q=np.zeros(18);q[6:]=.2
    out,fingers,info=ReachabilityFilter().project(palm,target,np.eye(4),q,np.zeros(18),np.ones(6)*.7,
        table_z=.9,hand_local=np.array([[0,0,0],[0,0,-.1]]))
    assert not info['filter_active']
    np.testing.assert_array_equal(out,target)
    np.testing.assert_array_equal(fingers,np.ones(6)*.7)


def test_geometric_candidate_follows_object_and_is_not_selected_by_object_identity():
    from wam_rl_policy_v3.grasp_repair import geometric_grasp_candidate
    palm=np.eye(4);palm[2,3]=1.2
    hand=np.array([[0,0,0],[0,0,-.05],[.05,0,-.10],[0,0,-.05],[-.05,0,-.10],[0,.01,-.05],[-.05,.01,-.10],[0,.02,-.05],[-.05,.02,-.10],[0,.03,-.05],[-.05,.03,-.10]])+palm[:3,3]
    obj=np.eye(4);obj[:3,3]=[.3,0,1.1];points=np.array([[x,y,z] for x in (-.02,.02) for y in (-.02,.02) for z in (-.02,.02)])
    a,info=geometric_grasp_candidate(palm,hand,obj,points,np.array([0,0,1]),.9,palm)
    shift=np.array([.2,-.4,0]);p2=palm.copy();p2[:3,3]+=shift;o2=obj.copy();o2[:3,3]+=shift
    b,_=geometric_grasp_candidate(p2,hand+shift,o2,points,np.array([0,0,1]),.9,p2)
    np.testing.assert_allclose(b[:3,3],a[:3,3]+shift,atol=1e-8)
    assert info['count']>=40 and np.linalg.norm(a[:3,3]-obj[:3,3])<.2


def test_same_state_ablation_keeps_every_non_tau_observation_byte_and_valid_mask():
    physical_task=np.arange(421,dtype=np.float32);hand=np.arange(33,dtype=np.float32).reshape(11,3)/100
    future=np.broadcast_to(hand,(24,11,3)).copy()+.05
    observed=[]
    for zero in (False,True):
        f,valid=clean_tau_future(future,hand,True,zero=zero)
        observed.append(np.r_[physical_task,(f[:8]-hand).ravel(),float(valid)].astype(np.float32))
    assert observed[0][:421].tobytes()==observed[1][:421].tobytes()
    assert observed[0][-1:].tobytes()==observed[1][-1:].tobytes()
    assert np.abs(observed[1][421:-1]).max()==0
    assert np.abs(observed[0][421:-1]).max()>.04


def test_actual_worker_saturation_counter_is_a_serializable_integer():
    import importlib.util,json
    from pathlib import Path
    p=Path(__file__).resolve().parents[1]/'tools/run/run_grasp_repair_worker.py'
    spec=importlib.util.spec_from_file_location('repair_worker_counter',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    info=[dict(phase='hold',reachability=None)]
    result=module.control_counts(info,np.array([[.025,0,0]],np.float32))
    assert result['goal_servo_saturated_controls']==1
    assert all(type(v) is int for v in result.values())
    assert json.loads(json.dumps(result))==result
