"""Approach/close regressions at the production consumer step boundary."""
from types import SimpleNamespace
import numpy as np
from wam_rl_policy_v3.source_ceiling import SourceCeilingConsumer, MarginAdmission,object_anchored_source_target
from wam_rl_policy_v3.pose_consumer import GraspIntent, FINGER_ACTIVE
from wam_rl_policy_v3.tau_replay_contract import pd_targets
from wam_rl_policy_v3.source_pd import source_pd_action
from wam_rl_policy_v3.pose_consumer import grasp_intent_from_retargeter


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
    c,args=approaching();args['q'][6]=-.04;args['dq'][6]=-1.
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


def test_close_to_lift_handover_cannot_jump_to_future_closure_target():
    c,args=approaching(force=.5);c.contact_since=-1.
    args['feedback'].grasp_established=True;args['feedback'].contact_observed=True
    command,info=c.step(**args)
    assert c.phase in ('lift','hold')
    target=pd_targets(command,args['q'],c.offset,c.scale)
    assert np.max(np.abs(target[FINGER_ACTIVE]-args['q'][FINGER_ACTIVE]))<=.04+1e-6


def test_executed_target_telemetry_matches_final_admitted_command():
    c,args=approaching(force=30.)
    args['q'][16]=-.01 # Geometry selection and measured unloading both active.
    command,info=c.step(**args)
    target=pd_targets(command,args['q'],c.offset,c.scale)
    np.testing.assert_allclose(info['executed_finger_target'],target[FINGER_ACTIVE],atol=1e-6,rtol=0)
    np.testing.assert_allclose(np.asarray(info['wrist_target'])[:3,3],args['palm'][:3,3]+target[:3]-args['q'][:3],atol=1e-6,rtol=0)


def source_target_intent(reference_object,live_object):
    q=np.zeros(18);q[2]=1.2;q[6:]=.2
    palm=np.eye(4);palm[2,3]=1.2
    source=pd_targets(np.r_[np.zeros(6),np.full(6,-.6)],q,np.zeros(18),np.ones(18))
    source=object_anchored_source_target(source,q,palm,reference_object,q,palm,live_object)
    action=source_pd_action(source,q,np.zeros(18),np.ones(18))[0]
    intent=grasp_intent_from_retargeter(q,palm,live_object,live_object,action,np.zeros(18),np.ones(18))
    return live_object@intent.approach_pose_object,source,intent


def test_source_approach_follows_live_object_translation_not_recording_world():
    reference=np.eye(4);reference[2,3]=1.1;live=reference.copy();live[0,3]=.05
    original,_,_=source_target_intent(reference,reference)
    moved,_,_=source_target_intent(reference,live)
    np.testing.assert_allclose(moved[:3,3]-original[:3,3],[.05,0,0],atol=1e-6,rtol=0)


def test_object_anchor_maps_rotation_and_preserves_finger_targets():
    from scipy.spatial.transform import Rotation
    reference=np.eye(4);reference[2,3]=1.1
    live=reference.copy();live[:3,:3]=Rotation.from_euler('y',.4).as_matrix()
    original,source,_=source_target_intent(reference,reference)
    moved,target,_=source_target_intent(reference,live)
    np.testing.assert_allclose(moved,live@np.linalg.inv(reference)@original,atol=1e-6,rtol=0)
    np.testing.assert_array_equal(source[6:],target[6:])


def test_force_bearing_candidate_gets_bounded_probe_before_motion_coupling():
    c,args=approaching(force=.5);args['feedback'].grasp_candidate=True
    command,info=c.step(**args)
    assert 0<command[2]*c.scale[2]<=.002+1e-6
    assert c.phase=='close', 'A probe must not invent acquired-grasp evidence'


def test_acquisition_probe_is_not_cumulative_and_requires_valid_contact():
    c,args=approaching(force=.5);args['feedback'].grasp_candidate=True
    start=args['palm'][2,3]
    for i in range(12):
        args['now']=i/30;command,info=c.step(**args)
        args['palm'][2,3]+=command[2]*c.scale[2];args['q'][2]=args['palm'][2,3]
        args['hand'][:]=args['palm'][:3,3]
        assert args['palm'][2,3]<=start+.01+1e-6
        assert c.phase=='close'
    args['feedback'].force_valid=False;args['now']=12/30
    _,info=c.step(**args)
    assert not info['source_ceiling']['acquisition_probe'] and c.acquire_anchor is None


def clock_reference(n=30,contact_frame=None):
    palm=np.zeros((n,1,7));palm[:,0,0]=np.arange(n)*.04;palm[:,0,2]=1.2;palm[:,0,6]=1
    objects=np.zeros((n,1,13));objects[:,0,2]=1.1;objects[:,0,6]=1
    force=np.zeros((n,1,5,3))
    if contact_frame is not None:force[contact_frame:,0,:2,2]=1.
    return dict(palm_pose=palm,objects=objects,action=np.zeros((n-1,12)),contact_forces=force,contact_force_valid=np.ones((n,1),bool))


def test_source_cursor_waits_for_measured_wrist_and_is_object_frame_invariant():
    from wam_rl_policy_v3.source_ceiling import SourceReferenceClock
    from wam_rl_policy_v3.pose_consumer import pose7
    ref=clock_reference();c=SourceReferenceClock(ref,.04)
    obj=pose7(ref['objects'][0,0]);palm=pose7(ref['palm_pose'][0,0])
    assert not c.advance(obj,palm)['advanced'] and c.cursor==0
    palm[0,3]=.03;assert c.advance(obj,palm)['advanced'] and c.cursor==1
    obj[1,3]+=.5;palm[1,3]+=.5
    assert not c.advance(obj,palm)['advanced'] and c.cursor==1
    palm[0,3]=.07;assert not c.advance(obj,palm,paused=True)['advanced'] and c.cursor==1


def test_reference_contact_caps_approach_but_missing_contact_is_not_removed():
    from wam_rl_policy_v3.source_ceiling import SourceReferenceClock
    from wam_rl_policy_v3.pose_consumer import pose7
    ref=clock_reference(contact_frame=1);c=SourceReferenceClock(ref,.04)
    obj=pose7(ref['objects'][0,0])
    for i in range(30):c.advance(obj,pose7(ref['palm_pose'][min(i,29),0]))
    assert c.cursor==16 and c.last_info['reference_contact_frame']==1
    missing=SourceReferenceClock(clock_reference(),.04)
    assert missing.limit==28 and missing.contact_frame is None


def test_measured_wrist_arrival_starts_closure_with_open_fingertips():
    c,args=approaching();c.phase='approach'
    args['near_fingers']=0;args['fingertip_gap_m']=1.
    args['intent'].preshape_q[:]=.2;args['intent'].closure_q[:]=.8
    command,info=c.step(**args)
    assert c.phase=='close'
    target=pd_targets(command,args['q'],c.offset,c.scale)
    assert np.all(target[FINGER_ACTIVE]>args['q'][FINGER_ACTIVE])
    assert not info['measured_contact'], 'Wrist arrival authorizes closing, not a grasp label'


def key_reference():
    ref=clock_reference(contact_frame=4);n=len(ref['objects'])
    points=np.array([[x,y,z] for x in (-.02,.02) for y in (-.02,.02) for z in (-.02,.02)])
    ref['hand']=np.tile(ref['objects'][:,:,None,:3]+points[0],(1,1,11,1))
    ref['q']=np.ones((n,1,18))*.2;ref['q'][8:,0,6:]=.8
    ref['action'][:,6:]=-.6;ref['action'][7:,6:]=.6
    return ref,points


def test_source_keygrasp_preserves_preshape_closure_and_live_SE3_without_clock():
    from wam_rl_policy_v3.source_ceiling import SourceGraspReference
    from wam_rl_policy_v3.pose_consumer import pose7
    ref,points=key_reference();key=SourceGraspReference(ref,points,np.zeros(18),np.ones(18))
    assert key.contact_frame==4 and key.grasp_frame==8
    intent=key.intent(np.eye(4));np.testing.assert_array_equal(intent.preshape_q,np.full(6,.2))
    np.testing.assert_allclose(intent.closure_q,np.full(6,.8),atol=1e-7,rtol=0)
    live=pose7(ref['objects'][8,0]);live[1,3]+=.15
    expected=pose7(ref['palm_pose'][8,0]);expected[1,3]+=.15
    np.testing.assert_allclose(live@intent.grasp_pose_object,expected,atol=1e-8)
    assert not key.manifest()['consumes_R'] and key.manifest()['source_action_mode']=='one_predecessor_closure_command_no_replay'


def test_source_contact_force_without_object_proximity_is_not_a_contact_label():
    from wam_rl_policy_v3.source_ceiling import SourceGraspReference
    ref,points=key_reference();ref['hand']+=10.
    key=SourceGraspReference(ref,points,np.zeros(18),np.ones(18))
    assert key.contact_frame is None and key.raw_force_contact_frames>0 and key.object_contact_frames==0
    assert key.manifest()['evidence']=='geometric_candidate_no_source_contact'


def test_force_bearing_closure_uses_preceding_pd_target_not_deflected_joint_state():
    from wam_rl_policy_v3.source_ceiling import SourceGraspReference
    ref,points=key_reference();ref['q'][8,0,6:]=.05
    ref['action'][7,6:]=.4 # Target .7 created the measured state at frame8.
    key=SourceGraspReference(ref,points,np.zeros(18),np.ones(18))
    np.testing.assert_allclose(key.closure,np.full(6,.7),atol=1e-6)
    assert key.manifest()['closure_command_frame']==7


def test_free_approach_is_fast_and_near_object_command_is_slow():
    results=[]
    for near in (False,True):
        c,args=approaching();c.phase='approach';args['near_fingers']=0
        args['intent'].approach_pose_object[0,3]+=.4;args['intent'].grasp_pose_object[0,3]+=.4
        args['fingertip_gap_m']=.066
        if near:
            args['hand'][4]=args['object_pose'][:3,3]+c.points[0]
            args['fingertip_gap_m']=0.
        command,info=c.step(**args);results.append(np.linalg.norm(command[:3]*c.scale[:3]))
    np.testing.assert_allclose(results,[.025,.01],atol=1e-6,rtol=0)


def one_contact_args(force_slot):
    c,args=approaching();c.phase='approach';args['near_fingers']=1;args['fingertip_gap_m']=0.
    args['intent'].approach_pose_object[0,3]+=.04;args['intent'].grasp_pose_object[0,3]+=.04
    args['hand'][4]=args['object_pose'][:3,3]+c.points[0] # Native index fingertip.
    args['hand'][8]+=np.array([.1,0,0]);args['hand'][10]-=np.array([.1,0,0])
    args['forces'][force_slot,2]=1.
    return c,args


def test_same_finger_contact_authorizes_closure_before_tight_wrist_goal():
    c,args=one_contact_args(0) # Force slot0 is index, native tip slot1.
    _,info=c.step(**args)
    assert c.phase=='close' and info['source_ceiling']['contact_ready']
    assert not info['measured_contact']


def test_thumb_force_does_not_pair_with_index_geometry():
    c,args=one_contact_args(4) # Thumb force, but only index is near the object.
    _,info=c.step(**args)
    assert c.phase=='approach' and not info['source_ceiling']['matched_contact']


def test_command_slew_accumulates_preload_when_measured_joint_is_blocked():
    c,args=approaching(force=.5);targets=[]
    for i in range(12):
        args['now']=i/30
        command,_=c.step(**args)
        targets.append(pd_targets(command,args['q'],c.offset,c.scale)[FINGER_ACTIVE])
    np.testing.assert_allclose(targets[-1][:4],np.full(4,.8),atol=1e-6)
    assert np.max(np.abs(np.diff(targets,axis=0)))<=.04+1e-6


def test_target_margin_does_not_trigger_measured_unload_on_ordinary_deflection():
    c,args=approaching();args['q'][6]=.01;args['dq'][6]=-.3
    command,info=c.step(**args)
    assert not info['source_ceiling']['unloading']
    assert c.admission.check(command,args['q'])[0]


def test_unloading_interlocks_lift_without_falsifying_feedback():
    c,args=approaching(force=30.);c.contact_since=-1.
    args['feedback'].grasp_established=True
    _,info=c.step(**args)
    assert c.phase=='close' and info['lift_interlocked_for_unloading']
    assert args['feedback'].grasp_established is True


def test_intermediate_link_contact_can_start_closure_before_tip_contact():
    c,args=one_contact_args(0)
    center=args['object_pose'][:3,3]
    args['hand'][3]=center+[-.04,.02,.02]
    args['hand'][4]=center+[.04,.02,.02]
    _,info=c.step(**args)
    assert c.phase=='close' and info['source_ceiling']['matched_contact']


def test_source_key_prefers_completed_coupled_lift_over_initial_touch():
    from wam_rl_policy_v3.source_ceiling import SourceGraspReference
    ref,points=key_reference()
    for t in range(len(ref['objects'])):
        ref['objects'][t,0,2]+=min(max(t-4,0)*.01,.15)
        ref['hand'][t,0]=ref['objects'][t,0,:3]+points[0]
        ref['palm_pose'][t,0,:3]=ref['objects'][t,0,:3]+[-.15,0,.05]
        ref['hand'][t,0,0]=ref['palm_pose'][t,0,:3]
    key=SourceGraspReference(ref,points,np.zeros(18),np.ones(18),plane=(np.array([0.,0,1.]),1.08))
    assert key.grasp_frame==9 and key.manifest()['evidence']=='measured_source_coupled_lift'
    assert key.manifest()['source_contact_is_live_success'] is False


def test_closing_reference_does_not_chase_object_after_real_contact():
    c,args=one_contact_args(0)
    c.step(**args) # Capture the grasp geometry at the measured first contact.
    args['now']=1/30;args['palm'][0,3]+=.04;args['q'][0]+=.04;args['hand'][:,0]+=.04
    args['object_pose'][0,3]+=.05 # Object slips, but the same index still touches.
    command,_=c.step(**args)
    assert abs(command[0])<1e-6, 'The wrist reached the captured grasp; do not chase the pushed object'


def test_contact_loss_releases_captured_goal_and_returns_to_approach():
    c,args=one_contact_args(0);c.step(**args)
    args['object_pose'][0,3]+=.2;args['forces'][:]=0
    args['intent'].preshape_q[:]=.2
    for t in range(1,6):
        args['now']=t/30;_,info=c.step(**args)
    assert c.phase=='approach' and c.closing_goal is None and info['source_ceiling']['regrasp_requested']
