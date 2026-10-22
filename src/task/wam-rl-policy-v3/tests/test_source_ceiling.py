"""Approach/close regressions at the production consumer step boundary."""
from types import SimpleNamespace
import numpy as np
from wam_rl_policy_v3.source_ceiling import SourceCeilingConsumer, MarginAdmission
from wam_rl_policy_v3.pose_consumer import GraspIntent, FINGER_ACTIVE
from wam_rl_policy_v3.tau_replay_contract import pd_targets


def approaching(cls=SourceCeilingConsumer, force=0., valid=True):
    q=np.ones(18)*.4;q[:6]=[0,0,1.2,0,0,0]
    palm=np.eye(4);palm[2,3]=1.2;obj=palm.copy();obj[0,3]=.08
    points=np.array([[x,y,z] for x in (-.02,.02) for y in (-.02,.02) for z in (-.02,.02)])
    bounds=dict(has_limits=[True]*3+[False]*3+[True]*12,
                lower=[-5,-5,0]+[0]*15, upper=[5,5,3]+[3.2]*15)
    c=cls(q,palm,np.zeros(18),np.ones(18),plane=(np.array([0.,0,1]),.9),
          palm_vertices=np.zeros((1,3)),object_points=points,joint_limits=bounds)
    c.phase='close';c.close_pose=palm.copy();c.fingers=np.ones(6)*.4
    local=np.linalg.inv(obj)@palm
    intent=GraspIntent(local.copy(),local.copy(),np.ones(6)*.8,np.ones(6)*.8,obj.copy(),'source',False)
    forces=np.zeros((5,3));forces[:,0]=-force
    feedback=SimpleNamespace(force_valid=valid,contact_observed=force>0,grasp_lost=False,grasp_established=False)
    args=dict(hand=np.tile(palm[:3,3],(11,1)),q=q,dq=np.zeros(18),palm=palm,
              object_pose=obj,object_velocity=np.zeros(6),forces=forces,feedback=feedback,
              near_fingers=2,fingertip_gap_m=0.,intent=intent,now=0.)
    return c,args


def test_measured_pressure_unloads_at_actual_approach_close_step():
    c,args=approaching(force=30.)
    command,info=c.step(**args)
    target=pd_targets(command,args['q'],c.offset,c.scale)
    assert np.all(target[FINGER_ACTIVE][:4]<args['q'][FINGER_ACTIVE][:4]), 'Compressed fingers must unload, not continue closing'
    assert command[0]<0, 'Hand must yield along measured force, away from the obstruction'


def test_approach_finger_target_does_not_jump_to_reference():
    c,args=approaching()
    command,info=c.step(**args)
    target=pd_targets(command,args['q'],c.offset,c.scale)
    assert np.max(np.abs(target[FINGER_ACTIVE]-args['q'][FINGER_ACTIVE]))<=.04+1e-6


def test_invalid_force_packet_cannot_initiate_retreat():
    c,args=approaching(force=30.,valid=False)
    _,info=c.step(**args)
    assert not info['source_ceiling']['unloading']


def test_nominally_legal_near_limit_target_is_rejected_before_candidate_ranking():
    from wam_rl_policy_v3.grasp_geometry import JointAdmission,grasp_candidates
    c,args=approaching()
    limits=dict(has_limits=c.outer_admission.mask.tolist(),lower=c.outer_admission.lower.tolist(),upper=c.outer_admission.upper.tolist())
    outer=JointAdmission(limits,c.offset,c.scale);inner=MarginAdmission(limits,c.offset,c.scale)
    raw=np.r_[np.zeros(6),np.full(6,-.98)] # .01 rad, legal but without 40mrad margin
    assert outer.check(raw,args['q'])[0]
    assert not inner.check(raw,args['q'])[0]
    candidate,info=grasp_candidates(args['palm'],args['hand'],args['object_pose'],c.points,*c.plane,args['palm'],np.ones(6)*.01,args['q'],inner)
    assert candidate is None and info['rejected_limits']==info['count']


def test_lower_joint_warning_recovers_inward_instead_of_opening_further():
    c,args=approaching();args['q'][6]=.01;args['dq'][6]=-1.
    command,info=c.step(**args)
    target=pd_targets(command,args['q'],c.offset,c.scale)
    assert info['source_ceiling']['unloading']
    assert target[6]>=.04-1e-6


def test_postgrasp_command_is_identical_to_existing_controller():
    from wam_rl_policy_v3.grasp_geometry import GeometryGraspConsumer
    a,args=approaching();b,_=approaching(GeometryGraspConsumer)
    for c in (a,b):c.phase='lift';c.contact_since=-1.
    args['feedback'].grasp_established=True;args['feedback'].contact_observed=True
    x,_=a.step(**args);y,_=b.step(**args)
    np.testing.assert_array_equal(x,y)
