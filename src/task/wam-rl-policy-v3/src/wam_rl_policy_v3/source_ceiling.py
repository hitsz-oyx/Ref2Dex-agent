"""Measured pre-grasp unloading; deliberately separate from the v13 consumer.

No object IDs, sequence IDs or task labels enter this controller. The held-object
servo and physical failure/scoring contracts are inherited without modification.
"""
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from .grasp_geometry import GeometryGraspConsumer, JointAdmission, COUPLINGS
from .pose_consumer import FINGER_ACTIVE, closest_wrist_angles,pose7,GraspIntent,TIPS
from .inspire_contract import ACTIVE,NATIVE_TO_WAM
from .tau_replay_contract import pd_targets
from .interaction_state import InteractionConfig
from .task_feedback import FeedbackObserver

SOURCE_CEILING_CONTRACT = 'source.measured-motion.close-before-probe.v12'


def object_anchored_source_target(target,reference_q,reference_palm,reference_object,
                                  live_q,live_palm,live_object):
    """Retain source finger commands while attaching its wrist intent to live SE3.

    Source remains a direct PD chain, not a GT-tau/R qualification. The reference
    frame and target are sampled together; no future object state is fed to WAM.
    """
    ref_mount=Rotation.from_euler('XYZ',reference_q[3:6]).as_matrix().T@reference_palm[:3,:3]
    desired=reference_palm.copy();desired[:3,3]+=target[:3]-reference_q[:3]
    desired[:3,:3]=Rotation.from_euler('XYZ',target[3:6]).as_matrix()@ref_mount
    desired=live_object@np.linalg.inv(reference_object)@desired
    mount=Rotation.from_euler('XYZ',live_q[3:6]).as_matrix().T@live_palm[:3,:3]
    result=np.asarray(target).copy();result[:3]=live_q[:3]+desired[:3,3]-live_palm[:3,3]
    result[3:6]=closest_wrist_angles(desired[:3,:3],live_q[3:6],mount)
    return result


class SourceReferenceClock:
    """Advance only when the measured wrist is closer to the next source state.

    This follows recorded STATES, not potentially unreachable PD targets. Source
    contact limits the approach horizon, never changes TaskSpec or success labels.
    No-contact references remain in the denominator with an explicit missing flag.
    """
    def __init__(self,reference,length_scale_m):
        self.local=np.array([np.linalg.inv(pose7(o[0]))@pose7(p[0]) for o,p in zip(reference['objects'],reference['palm_pose'])])
        count=(np.linalg.norm(reference['contact_forces'][:,0],axis=-1)>=.1).sum(-1)
        contact=np.flatnonzero((count>=2)&reference['contact_force_valid'][:,0])
        self.contact_frame=int(contact[0]) if len(contact) else None
        self.limit=len(reference['action'])-1
        if self.contact_frame is not None:self.limit=min(self.limit,self.contact_frame+15)
        self.scale=float(length_scale_m)
        if self.scale<=0:raise ValueError('positive measured geometry scale required')
        self.cursor=0;self.last_info=None

    def advance(self,object_pose,palm,*,paused=False):
        live=np.linalg.inv(object_pose)@palm;next_frame=min(self.cursor+1,self.limit)
        def error(frame):
            ref=self.local[frame]
            return float(np.linalg.norm(ref[:3,3]-live[:3,3])/self.scale+
                         np.linalg.norm(Rotation.from_matrix(ref[:3,:3]@live[:3,:3].T).as_rotvec())/np.pi)
        before=self.cursor;current_error=error(before);next_error=error(next_frame)
        if not paused and next_error<=current_error+1e-9:self.cursor=next_frame
        self.last_info=dict(cursor_before=before,cursor=self.cursor,advanced=self.cursor!=before,
            current_error=current_error,next_error=next_error,paused=bool(paused),
            reference_contact_frame=self.contact_frame,reference_contact_missing=self.contact_frame is None,
            approach_frame_limit=self.limit)
        return self.last_info


class SourceGraspReference:
    """Three source intentions, without replaying its unstable PD/state clock.

    Contact is evidence from the source recording, not a live success label.
    No-contact recordings get an explicit geometric candidate and stay in n40.
    """
    def __init__(self,reference,points,offset,scale,*,plane=None):
        force=np.asarray(reference['contact_forces'])[:,0]
        raw_contact=(np.linalg.norm(force,axis=-1)>=.1).sum(-1)>=2
        raw_contact&=reference['contact_force_valid'][:,0]&np.isfinite(force).all(axis=(1,2))
        hands=np.asarray(reference['hand'])[:,0,TIPS]
        rotations=Rotation.from_quat(reference['objects'][:,0,3:7]).as_matrix()
        local=np.einsum('tki,tij->tkj',hands-reference['objects'][:,0,None,:3],rotations)
        gaps=cKDTree(points).query(local.reshape(-1,3))[0].reshape(len(local),5)
        # Reuse the production observer's measured proximity contract. Native
        # fingertip forces include table/self contacts, not only the object.
        near=(gaps<=InteractionConfig().near_surface_m).sum(-1)>=2
        contact=raw_contact&near
        self.raw_force_contact_frames=int(raw_contact.sum());self.object_contact_frames=int(contact.sum())
        found=np.flatnonzero(contact);self.contact_frame=int(found[0]) if len(found) else None
        observer=FeedbackObserver(points);coupled=np.zeros(len(force),bool)
        if plane is not None:
            normal,height=plane;initial_z=reference['objects'][0,0,2]
            for t in range(len(force)):
                obj=reference['objects'][t,0];pose=pose7(obj)
                gap=float(np.min((points@pose[:3,:3].T+pose[:3,3])@normal)-height)
                feedback=observer.update(pose,reference['hand'][t,0,NATIVE_TO_WAM],obj[7:13],t/30,
                    contact_forces=force[t],force_valid=bool(reference['contact_force_valid'][t,0]),supported=gap<=.002)
                coupled[t]=bool(feedback.grasp_established and obj[2]-initial_z>=.05 and gap>.02 and contact[t])
        positive=np.flatnonzero(coupled);self.source_coupled_lift_frames=int(coupled.sum())
        if len(found):
            stable=[int(i)+4 for i in found if i+4<len(contact) and contact[i:i+5].all()]
            frame=int(positive[0]) if len(positive) else stable[0] if stable else self.contact_frame
            self.evidence='measured_source_coupled_lift' if len(positive) else 'measured_source_force_contact'
        else:
            frame=int(np.argmin(np.partition(gaps,1,axis=1)[:,1]))
            self.evidence='geometric_candidate_no_source_contact'
        self.grasp_frame=frame;self.preshape_frame=max(0,(self.contact_frame if self.contact_frame is not None else frame)-5)
        self.grasp=np.linalg.inv(pose7(reference['objects'][frame,0]))@pose7(reference['palm_pose'][frame,0])
        self.preshape=np.asarray(reference['q'][self.preshape_frame,0,FINGER_ACTIVE]).copy()
        self.command_frame=max(0,frame-1)
        command=reference['action'][self.command_frame]
        if command.shape==(18,):command=command[list(ACTIVE)]
        self.closure=pd_targets(command,reference['q'][self.command_frame,0],offset,scale)[FINGER_ACTIVE]
        self.measured_grasp_q=np.asarray(reference['q'][frame,0,FINGER_ACTIVE]).copy()
        self.motion_cursor=0;self.motion_enabled=False
        self.motion_local=np.array([np.linalg.inv(pose7(o[0]))@pose7(p[0]) for o,p in zip(reference['objects'][:frame+1],reference['palm_pose'][:frame+1])])
        self.motion_pd=[];self.motion_fingers=[]
        for t in range(frame+1):
            ci=min(t,len(reference['action'])-1);action=reference['action'][ci]
            if action.shape==(18,):action=action[list(ACTIVE)]
            desired=pd_targets(action,reference['q'][ci,0],offset,scale)
            palm=pose7(reference['palm_pose'][ci,0]);source_q=reference['q'][ci,0]
            mount=Rotation.from_euler('XYZ',source_q[3:6]).as_matrix().T@palm[:3,:3]
            target=palm.copy();target[:3,3]+=desired[:3]-source_q[:3]
            target[:3,:3]=Rotation.from_euler('XYZ',desired[3:6]).as_matrix()@mount
            self.motion_pd.append(np.linalg.inv(pose7(reference['objects'][ci,0]))@target)
            self.motion_fingers.append(desired[FINGER_ACTIVE].copy())
        # The key state was produced by action[f-1]. Preserve all six wrist PD
        # targets as well as finger preload; do not confuse measured deflection
        # or motion lag with the command. This is not an estimate of contact force.
        qi=reference['q'][frame,0];palm=pose7(reference['palm_pose'][frame,0])
        source_command=reference['action'][self.command_frame]
        if source_command.shape==(18,):source_command=source_command[list(ACTIVE)]
        desired=pd_targets(source_command,reference['q'][self.command_frame,0],offset,scale)
        mount=Rotation.from_euler('XYZ',qi[3:6]).as_matrix().T@palm[:3,:3]
        target=palm.copy();target[:3,3]+=desired[:3]-qi[:3]
        target[:3,:3]=Rotation.from_euler('XYZ',desired[3:6]).as_matrix()@mount
        self.pd_grasp=np.linalg.inv(pose7(reference['objects'][frame,0]))@target

    def intent(self,object_goal,*,live_object=None,live_palm=None):
        if live_object is None or live_palm is None:
            return GraspIntent(self.grasp.copy(),self.grasp.copy(),self.preshape.copy(),self.closure.copy(),
                               np.asarray(object_goal).copy(),'source-keygrasp',False,True)
        self.motion_enabled=True
        live=np.linalg.inv(live_object)@live_palm
        candidates=range(self.motion_cursor,min(self.motion_cursor+9,len(self.motion_local)))
        def error(t):
            reference=self.motion_local[t]
            return float(np.linalg.norm(reference[:3,3]-live[:3,3])+.05*np.linalg.norm(Rotation.from_matrix(reference[:3,:3]@live[:3,:3].T).as_rotvec()))
        self.motion_cursor=min(candidates,key=error)
        return GraspIntent(self.motion_pd[self.motion_cursor].copy(),self.pd_grasp.copy(),
            self.motion_fingers[self.motion_cursor].copy(),self.closure.copy(),
            np.asarray(object_goal).copy(),'source-measured-motion-PD',False,True)

    def manifest(self):
        return dict(kind='source_measured_motion_PD',motion_cursor=self.motion_cursor,evidence=self.evidence,contact_frame=self.contact_frame,
                    grasp_frame=self.grasp_frame,preshape_frame=self.preshape_frame,
                    raw_force_contact_frames=self.raw_force_contact_frames,object_contact_frames=self.object_contact_frames,
                    source_coupled_lift_frames=self.source_coupled_lift_frames,
                    consumes_source_actions=True,source_action_mode='measured_state_PD_path_and_predecessor_grasp_command' if self.motion_enabled else 'one_predecessor_closure_command_no_replay',
                    closure_command_frame=self.command_frame,closure_has_predecessor=self.grasp_frame>0,
                    measured_grasp_q=self.measured_grasp_q.tolist(),closure_pd_target_q=self.closure.tolist(),
                    consumes_R=False,source_contact_is_live_success=False)


class MarginAdmission(JointAdmission):
    """Use an inner target workspace, never relax the physical abort bounds."""
    def __init__(self, limits, offset, scale):
        inner={k:list(v) for k,v in limits.items()}
        for i,limited in enumerate(limits['has_limits']):
            if not limited:continue
            margin=min(.04 if i>=6 else .01, .1*(limits['upper'][i]-limits['lower'][i]))
            inner['lower'][i]+=margin;inner['upper'][i]-=margin
        super().__init__(inner,offset,scale)
        self.measured_lower=np.asarray(limits['lower'],float)
        self.measured_upper=np.asarray(limits['upper'],float)

    def measured_pressure(self,q,dq):
        # Command margins are NOT physical bounds. Ordinary elastic deflection
        # must not replace the reference by another grasp. Match the existing
        # source warning halfway to the unchanged +/- .10 physical abort.
        q=np.asarray(q);dq=np.asarray(dq);projected=q+dq/30
        risk=self.mask&(((projected<self.measured_lower-.05)&(dq<-.2))|
                        ((projected>self.measured_upper+.05)&(dq>.2)))
        return bool(risk.any())


class SourceCeilingConsumer(GeometryGraspConsumer):
    """Retain source intent, yield before collision pressure becomes a limit hit.

    Fingertip force is a measured vector on the hand, not a task reward. Invalid
    force packets cannot initiate unloading. Actual joint coasting is a separate
    warning because fingertip sensors do not cover all hand/self contacts.
    """
    force_limit_n=12.
    finger_rate_radps=1.2
    reaching_linear_rate_mps=.30
    reaching_angular_rate_radps=1.2
    free_linear_rate_mps=.75
    free_angular_rate_radps=2.4
    acquisition_distance_m=.01
    acquisition_rate_mps=.06
    wrist_angle_tolerance_rad=.2
    force_to_native_tip=(1,2,4,3,0)
    measured_warning_margin=.05 # Half the unchanged .10 physical abort margin.
    contact_geometry='segment'
    contact_loss_dwell_s=.12
    probe_closure_dwell_s=InteractionConfig().history_s

    def __init__(self,*args,joint_limits,**kwargs):
        super().__init__(*args,joint_limits=joint_limits,**kwargs)
        self.outer_admission=self.admission
        self.inner_admission=MarginAdmission(joint_limits,self.offset,self.scale)
        self.unload_until=-np.inf
        self.unload_direction=None
        self.source_paused=False
        self.acquire_anchor=None
        self.object_tree=cKDTree(self.points)
        initial=np.asarray(args[0] if args else kwargs['q'])
        self.wrist_command=np.asarray(args[1] if len(args)>1 else kwargs['palm']).copy()
        self.finger_target=self.inner_admission.fingers(initial[FINGER_ACTIVE])
        self.unload_fingers=self.finger_target.copy()
        self.closing_goal=None;self.closing_lost_since=None;self.regrasp_pending=False
        self.closure_ready_since=None

    def closure_ready_for_probe(self,now,closure,candidate):
        completed=np.max(np.abs(self.finger_target-self.admission.fingers(closure)))<=1e-6
        if not (candidate and self.phase=='close' and completed):
            self.closure_ready_since=None
            return False
        if self.closure_ready_since is None:self.closure_ready_since=now
        return now-self.closure_ready_since>=self.probe_closure_dwell_s-1e-10

    def step(self,**args):
        if self.phase not in ('approach','close'):
            # Same post-grasp control as the registered source baseline.
            self.admission=self.outer_admission
            self.source_paused=False
            self.acquire_anchor=None;self.closure_ready_since=None
            self.closing_goal=None;self.closing_lost_since=None
            command,info=super().step(**args)
            applied=pd_targets(command,args['q'],self.offset,self.scale)
            self.finger_target=applied[FINGER_ACTIVE].copy()
            self.wrist_command=np.asarray(info['wrist_target']).copy()
            info['source_ceiling']=dict(contract=SOURCE_CEILING_CONTRACT,active=False,source_paused=False)
            return command,info
        self.admission=self.inner_admission
        q=np.asarray(args['q']);dq=np.asarray(args['dq']);palm=np.asarray(args['palm'])
        now=float(args['now']);intent=args['intent'];force=np.asarray(args['forces'])
        grasp=args['object_pose']@intent.grasp_pose_object
        distances=self.object_tree.query((np.asarray(args['hand'])[TIPS]-args['object_pose'][:3,3])@args['object_pose'][:3,:3])[0]
        local_hand=(np.asarray(args['hand'])-args['object_pose'][:3,3])@args['object_pose'][:3,:3]
        bases=local_hand[TIPS-1];segments=local_hand[TIPS]-bases
        along=np.clip(np.einsum('fpi,fi->fp',self.points[None]-bases[:,None],segments)/np.maximum((segments*segments).sum(-1)[:,None],1e-12),0,1)
        segment_gaps=np.linalg.norm(self.points[None]-(bases[:,None]+along[:,:,None]*segments[:,None]),axis=-1).min(-1)
        force_valid=bool(args['feedback'].force_valid)
        magnitudes=np.linalg.norm(force,axis=-1) if force_valid else np.zeros(5)
        # Native tips: thumb,index,middle,ring,pinky; force/active-driver order:
        # index,middle,pinky,ring,thumb. Never pair sensors by raw slot number.
        contact_gaps=segment_gaps if self.contact_geometry=='segment' else distances
        contact_match=bool(((contact_gaps[list(self.force_to_native_tip)]<=InteractionConfig().near_surface_m)&(magnitudes>=.1)).any())
        tips=np.asarray(args['hand'])[TIPS]
        hand_span=float(np.linalg.norm(tips[:,None]-tips[None],axis=-1).max())
        wrist_distance=float(np.linalg.norm(grasp[:3,3]-palm[:3,3]))
        wrist_angle=float(np.linalg.norm(Rotation.from_matrix(grasp[:3,:3]@palm[:3,:3].T).as_rotvec()))
        wrist_arrived=bool(np.linalg.norm(grasp[:3,3]-palm[:3,3])<=self.geometry.wrist_gap_m and
                           wrist_angle<=self.wrist_angle_tolerance_rad)
        contact_ready=bool(contact_match and wrist_distance<=hand_span*.5+self.geometry.wrist_gap_m and wrist_angle<=.5)
        wrist_arrived|=contact_ready
        recovered_after_unload=bool(self.regrasp_pending and now>=self.unload_until)
        if recovered_after_unload:
            self.phase='approach';self.closing_goal=None;self.closing_lost_since=None
            self.contact_since=None;self.acquire_anchor=None;self.regrasp_pending=False
        if self.phase=='approach' and wrist_arrived:
            # Closing is an attempted action, not a contact/grasp certificate.
            self.phase='close';self.close_pose=palm.copy();self.fingers=q[FINGER_ACTIVE].copy()
        regrasp_requested=False
        if self.phase=='close':
            if self.closing_goal is None and contact_match:self.closing_goal=palm.copy()
            if self.closing_goal is not None:
                if contact_match:self.closing_lost_since=None
                elif force_valid:
                    if self.closing_lost_since is None:self.closing_lost_since=now
                    if now-self.closing_lost_since>=self.contact_loss_dwell_s-1e-10:
                        self.phase='approach';self.closing_goal=None;self.closing_lost_since=None
                        self.contact_since=None;self.acquire_anchor=None;regrasp_requested=True
        near_object=bool(args['fingertip_gap_m']<=InteractionConfig().approach_distance_m or args['feedback'].contact_observed)
        linear_rate=self.reaching_linear_rate_mps if near_object else self.free_linear_rate_mps
        angular_rate=self.reaching_angular_rate_radps if near_object else self.free_angular_rate_radps
        loaded=magnitudes>self.force_limit_n
        projected=q+dq/30
        # Directional prediction: harmless motion back into the workspace must
        # not trigger another retreat. Continuous wrist angles have no bounds.
        # Targets keep the URDF inner margin. Actual compliant contact states
        # warn halfway to the EXISTING physical abort, not at the target margin.
        # The physical_safety function and its .10m/.10rad bound are untouched.
        lower=(self.outer_admission.mask & (projected<self.outer_admission.lower-self.measured_warning_margin) & (dq<-.2))
        upper=(self.outer_admission.mask & (projected>self.outer_admission.upper+self.measured_warning_margin) & (dq>.2))
        risk=lower|upper
        trigger=bool(loaded.any() or risk[6:].any())
        if trigger:
            self.closing_goal=None;self.closing_lost_since=None;self.regrasp_pending=True
            self.unload_until=now+.10
            direction=force[loaded].sum(0) if loaded.any() else palm[:3,3]-args['object_pose'][:3,3]
            normal,height=self.plane
            if np.linalg.norm(direction)<1e-8:direction=normal.copy()
            # Do not turn unloading into a table strike.
            direction=direction-normal*min(float(direction@normal),0.)
            if np.linalg.norm(direction)<1e-8:direction=normal.copy()
            self.unload_direction=direction/np.linalg.norm(direction)
        unloading=now<self.unload_until
        self.source_paused=unloading
        fingers=np.asarray(intent.closure_q if self.phase=='close' else intent.preshape_q).copy()
        if trigger:
            finger_load=np.r_[loaded[:4],loaded[4],loaded[4]]
            fingers=np.where(finger_load,np.minimum(self.finger_target-.04,q[FINGER_ACTIVE]-.04),self.finger_target)
            # An actual lower-bound excursion needs inward recovery, not more
            # opening. Mimic-joint warnings act on their independent driver.
            for i,joint in enumerate(FINGER_ACTIVE):
                group=[(joint,1.)]+[(dst,ratio) for dst,src,ratio in COUPLINGS if src==joint]
                if any(lower[j] for j,_ in group):
                    fingers[i]=max(q[j]/ratio+.04 for j,ratio in group)
                if any(upper[j] for j,_ in group):
                    fingers[i]=min(q[j]/ratio-.04 for j,ratio in group)
            self.unload_fingers=self.admission.fingers(fingers)
        fingers=self.unload_fingers.copy() if unloading else self.finger_target+np.clip(fingers-self.finger_target,-self.finger_rate_radps/30,self.finger_rate_radps/30)
        fingers=self.admission.fingers(fingers)
        closure=self.finger_target+np.clip(intent.closure_q-self.finger_target,-self.finger_rate_radps/30,self.finger_rate_radps/30)
        closure=self.admission.fingers(closure)
        captured_close=bool(self.phase=='close' and self.closing_goal is not None and not unloading)
        proposal=self.closing_goal.copy() if captured_close else args['object_pose']@intent.approach_pose_object
        candidate=bool(force_valid and args['feedback'].contact_observed and getattr(args['feedback'],'grasp_candidate',False))
        closure_ready=self.closure_ready_for_probe(now,intent.closure_q,candidate and not unloading)
        probing=bool(candidate and closure_ready and not args['feedback'].grasp_established and not unloading)
        if not probing:self.acquire_anchor=None
        if probing:
            if self.acquire_anchor is None:self.acquire_anchor=palm.copy()
            proposal=self.acquire_anchor.copy();proposal[:3,3]+=self.plane[0]*self.acquisition_distance_m
            # A contact candidate is not a completed grasp. Keep the planned,
            # slew-limited closure until acquisition; premature hold updates
            # freeze partially open fingers. Overpressure still unloads above.
        if unloading:
            proposal=palm.copy();proposal[:3,3]+=self.unload_direction*self.reaching_linear_rate_mps/30
        # Pre-limit the source too, not just replacement geometry candidates.
        # PD target offset is not velocity. Slew consecutive commanded poses,
        # retaining the tracking/feed-forward offset required by native PD.
        # A pressure retreat or bounded acquisition probe keeps its measured
        # anchor and safety displacement instead of preserving old preload.
        base=palm if unloading or probing else self.wrist_command
        delta=proposal[:3,3]-base[:3,3]
        linear_step=self.acquisition_rate_mps/30 if probing else linear_rate/30
        delta*=min(1.,linear_step/max(np.linalg.norm(delta),1e-12))
        proposal[:3,3]=base[:3,3]+delta
        rotation=Rotation.from_matrix(proposal[:3,:3]@base[:3,:3].T).as_rotvec()
        rotation*=min(1.,angular_rate/30/max(np.linalg.norm(rotation),1e-12))
        proposal[:3,:3]=Rotation.from_rotvec(rotation).as_matrix()@base[:3,:3]
        mount=Rotation.from_euler('XYZ',q[3:6]).as_matrix().T@palm[:3,:3]
        changed=replace(intent,approach_pose_object=np.linalg.inv(args['object_pose'])@proposal,
                        preshape_q=fingers,closure_q=closure,approach_joint_angles=closest_wrist_angles(proposal[:3,:3],q[3:6],mount))
        if captured_close:changed=replace(changed,grasp_pose_object=np.linalg.inv(args['object_pose'])@self.closing_goal)
        # Parent geometry selection still checks all candidate targets against
        # the INNER bounds before ranking. It also retains table feasibility.
        command,info=super().step(**dict(args,intent=changed))
        if unloading and self.phase in ('lift','hold'):
            # Preserve the real feedback, but interlock commanded lift while
            # measured pressure/joint danger is still being relieved.
            self.phase='close';self.servo.reset();self.servo_active=False
            info.update(phase='close',lift_interlocked_for_unloading=True)
        if self.phase in ('approach','close'):
            # Apply slew limits AFTER geometry selection too. Candidate rotation
            # and target admission cannot bypass the approach control budget.
            target=pd_targets(command,q,self.offset,self.scale)
            candidate=palm.copy();candidate[:3,3]+=target[:3]-q[:3]
            candidate[:3,:3]=Rotation.from_euler('XYZ',target[3:6]).as_matrix()@mount
            delta=candidate[:3,3]-base[:3,3]
            delta*=min(1.,linear_rate/30/max(np.linalg.norm(delta),1e-12))
            candidate[:3,3]=base[:3,3]+delta
            turn=Rotation.from_matrix(candidate[:3,:3]@base[:3,:3].T).as_rotvec()
            turn*=min(1.,angular_rate/30/max(np.linalg.norm(turn),1e-12))
            candidate[:3,:3]=Rotation.from_rotvec(turn).as_matrix()@base[:3,:3]
            command[:3]=(candidate[:3,3]-palm[:3,3]-self.offset[:3])/self.scale[:3]
            command[3:6]=(closest_wrist_angles(candidate[:3,:3],q[3:6],mount)-q[3:6]-self.offset[3:6])/self.scale[3:6]
            command=self.admission.bound_action(command,q)
        if unloading and self.phase in ('approach','close'):
            # A new candidate must not overwrite this tick's measured retreat.
            command[:3]=(proposal[:3,3]-palm[:3,3]-self.offset[:3])/self.scale[:3]
            command[3:6]=-self.offset[3:6]/self.scale[3:6]
            command[6:]=2*(fingers-self.offset[FINGER_ACTIVE])/self.scale[FINGER_ACTIVE]-1
            command=self.admission.bound_action(command,q)
        elif (probing or captured_close) and self.phase in ('approach','close'):
            command[:3]=(proposal[:3,3]-palm[:3,3]-self.offset[:3])/self.scale[:3]
            command[3:6]=(closest_wrist_angles(proposal[:3,:3],q[3:6],mount)-q[3:6]-self.offset[3:6])/self.scale[3:6]
            command[6:]=2*(fingers-self.offset[FINGER_ACTIVE])/self.scale[FINGER_ACTIVE]-1
            command=self.admission.bound_action(command,q)
        applied_unloading=bool(unloading and self.phase in ('approach','close'))
        self.source_paused=applied_unloading
        applied=pd_targets(command,q,self.offset,self.scale)
        executed_pose=palm.copy();executed_pose[:3,3]+=applied[:3]-q[:3]
        executed_pose[:3,:3]=Rotation.from_euler('XYZ',applied[3:6]).as_matrix()@mount
        info.update(wrist_target=executed_pose.tolist(),executed_finger_target=applied[FINGER_ACTIVE].tolist())
        finger_step=float(np.max(np.abs(applied[FINGER_ACTIVE]-self.finger_target)))
        finger_error=float(np.max(np.abs(applied[FINGER_ACTIVE]-q[FINGER_ACTIVE])))
        info['source_ceiling']=dict(contract=SOURCE_CEILING_CONTRACT,active=True,
            force_valid=force_valid,force_norm_n=magnitudes.tolist(),force_limit_n=self.force_limit_n,
            wrist_pose_ready=wrist_arrived,
            contact_ready=contact_ready,matched_contact=contact_match,hand_span_m=hand_span,
            closing_reference_captured=captured_close,closing_goal_world=None if self.closing_goal is None else self.closing_goal.tolist(),
            regrasp_requested=regrasp_requested,recovered_after_unload=recovered_after_unload,
            control_contact_geometry=self.contact_geometry,
            near_object=near_object,command_linear_rate_mps=linear_rate,command_angular_rate_radps=angular_rate,
            loaded_fingers=np.flatnonzero(loaded).tolist(),coasting_joint_indices=np.flatnonzero(risk).tolist(),
            unloading=applied_unloading,unload_requested=unloading,source_paused=self.source_paused,inner_target_admitted=self.admission.check(command,q)[0],
            acquisition_probe=probing,closure_ready_for_probe=closure_ready,closure_ready_since=self.closure_ready_since,probe_closure_dwell_s=self.probe_closure_dwell_s,
            acquisition_anchor=None if self.acquire_anchor is None else self.acquire_anchor.tolist(),acquisition_maximum_m=self.acquisition_distance_m,
            finger_target_step_rad=finger_step,finger_tracking_error_rad=finger_error,
            finger_emergency_slew_override=bool(finger_step>self.finger_rate_radps/30+1e-6),
            measured_warning_margin=self.measured_warning_margin,physical_abort_margin_unchanged=.10,
            wrist_target_step_m=float(np.linalg.norm(executed_pose[:3,3]-self.wrist_command[:3,3])),
            wrist_tracking_offset_m=float(np.linalg.norm(executed_pose[:3,3]-palm[:3,3])),
            wrist_slew_reference='previous_command_except_measured_retreat_or_bounded_probe',
            finger_rate_radps=self.finger_rate_radps,retreat_direction=None if not unloading else self.unload_direction.tolist())
        self.last_info=info
        self.finger_target=applied[FINGER_ACTIVE].copy()
        self.wrist_command=executed_pose.copy()
        return command,info


class SourcePDGuard(GeometryGraspConsumer):
    """Minimal guard around the original source-PD intent/phase/servo chain.

    No keypose replacement, wrist-freezing, artificial PD-error rate clamp or
    acquisition probe. The source trajectory supplies the acquisition motion.
    """
    force_limit_n=12.
    contract='source-PD-guard.force-margin-bandwidth.v1'
    finger_rate_radps=1.2
    wrist_joint_rate_radps=2.4
    far_rate_mps=.75
    near_rate_mps=.30
    far_tracking_offset_m=.075
    far_angular_offset_rad=.15

    def __init__(self,*args,joint_limits,**kwargs):
        super().__init__(*args,joint_limits=joint_limits,**kwargs)
        self.outer_admission=self.admission
        self.inner_admission=MarginAdmission(joint_limits,self.offset,self.scale)
        self.unload_until=-np.inf;self.unload_direction=None
        self.unload_fingers=None
        self.last_pd_target=np.asarray(args[0] if args else kwargs['q']).copy()

    def limit_wrist_target(self,target,q,near,hand):
        result=target.copy();previous=self.last_pd_target
        delta=result[:3]-previous[:3];rate=self.near_rate_mps if near else self.far_rate_mps
        delta*=min(1.,rate/30/max(np.linalg.norm(delta),1e-12))
        result[:3]=previous[:3]+delta
        offset=result[:3]-q[:3]
        bound=self.geometry.wrist_gap_m if near else self.far_tracking_offset_m
        offset*=min(1.,bound/max(np.linalg.norm(offset),1e-12));result[:3]=q[:3]+offset
        turn=result[3:6]-previous[3:6];turn-=2*np.pi*np.round(turn/(2*np.pi))
        turn*=min(1.,self.wrist_joint_rate_radps/30/max(np.linalg.norm(turn),1e-12))
        result[3:6]=previous[3:6]+turn
        offset=result[3:6]-q[3:6];offset-=2*np.pi*np.round(offset/(2*np.pi))
        tips=np.asarray(hand)[TIPS];span=float(np.linalg.norm(tips[:,None]-tips[None],axis=-1).max())
        angular_bound=min(self.far_angular_offset_rad,self.geometry.wrist_gap_m/max(span,self.geometry.diameter_m)) if near else self.far_angular_offset_rad
        offset*=min(1.,angular_bound/max(np.linalg.norm(offset),1e-12));result[3:6]=q[3:6]+offset
        return result

    def step(self,**args):
        recovery=self.phase in ('lift','hold') and (not args['feedback'].contact_observed or args['feedback'].grasp_lost)
        pre=self.phase in ('approach','close') or recovery
        self.admission=self.inner_admission if pre else self.outer_admission
        q=np.asarray(args['q']);dq=np.asarray(args['dq']);palm=np.asarray(args['palm'])
        now=float(args['now']);intent=args['intent'];valid=bool(args['feedback'].force_valid)
        force=np.asarray(args['forces']);magnitudes=np.linalg.norm(force,axis=-1) if valid else np.zeros(5)
        projected=q+dq/30;outer=self.outer_admission
        lower=outer.mask&(projected<outer.lower-.05)&(dq<-.2)
        upper=outer.mask&(projected>outer.upper+.05)&(dq>.2)
        loaded=magnitudes>self.force_limit_n
        trigger=bool(pre and (loaded.any() or (lower|upper).any()))
        if trigger:
            self.unload_until=now+.1
            direction=force[loaded].sum(0) if loaded.any() else palm[:3,3]-args['object_pose'][:3,3]
            normal=self.plane[0];direction=direction-normal*min(float(direction@normal),0.)
            if np.linalg.norm(direction)<1e-8:direction=normal.copy()
            self.unload_direction=direction/np.linalg.norm(direction)
            fingers=np.asarray(intent.preshape_q).copy();finger_load=np.r_[loaded[:4],loaded[4],loaded[4]]
            fingers=np.where(finger_load,np.minimum(fingers,q[FINGER_ACTIVE]-.04),fingers)
            for i,joint in enumerate(FINGER_ACTIVE):
                group=[(joint,1.)]+[(dst,ratio) for dst,src,ratio in COUPLINGS if src==joint]
                if any(lower[j] for j,_ in group):fingers[i]=max(q[j]/ratio+.04 for j,ratio in group)
                if any(upper[j] for j,_ in group):fingers[i]=min(q[j]/ratio-.04 for j,ratio in group)
            self.unload_fingers=self.admission.fingers(fingers)
        unloading=bool(pre and now<self.unload_until)
        changed=intent;retreat=None;finger_limited=root_limited=handover_limited=False;rate_projection_override=False
        if pre:
            preshape=self.admission.fingers(intent.preshape_q);closure=self.admission.fingers(intent.closure_q)
            if not unloading:
                previous=self.last_pd_target[FINGER_ACTIVE]
                limited_preshape=previous+np.clip(preshape-previous,-self.finger_rate_radps/30,self.finger_rate_radps/30)
                limited_closure=previous+np.clip(closure-previous,-self.finger_rate_radps/30,self.finger_rate_radps/30)
                finger_limited=bool((np.abs(limited_preshape-preshape)>1e-6).any())
                handover_limited=bool((np.abs(limited_closure-closure)>1e-6).any())
                preshape=self.admission.fingers(limited_preshape);closure=self.admission.fingers(limited_closure)
            changed=replace(intent,preshape_q=preshape,closure_q=closure)
            if unloading:
                retreat=palm.copy();retreat[:3,3]+=self.unload_direction*.01
                local=np.linalg.inv(args['object_pose'])@retreat
                changed=replace(changed,approach_pose_object=local,grasp_pose_object=local,
                    preshape_q=self.unload_fingers.copy(),closure_q=self.unload_fingers.copy(),approach_joint_angles=q[3:6].copy())
        command,info=super().step(**dict(args,intent=changed))
        if pre and not unloading and self.phase in ('approach','close'):
            desired=pd_targets(command,q,self.offset,self.scale)
            near=bool(args['fingertip_gap_m']<=InteractionConfig().approach_distance_m or args['feedback'].contact_observed)
            limited=self.limit_wrist_target(desired,q,near,args['hand'])
            root_limited=bool((np.abs(limited[:6]-desired[:6])>1e-6).any())
            raw=command.copy();raw[:6]=(limited[:6]-q[:6]-self.offset[:6])/self.scale[:6]
            command=self.admission.bound_action(raw,q)
            actual=pd_targets(command,q,self.offset,self.scale)
            rate_projection_override=bool((np.abs(actual[:6]-limited[:6])>1e-6).any())
        if unloading:
            if self.phase in ('lift','hold'):
                self.phase='close';self.servo.reset();self.servo_active=False
                info.update(phase='close',lift_interlocked_for_unloading=True)
            command[:3]=(retreat[:3,3]-palm[:3,3]-self.offset[:3])/self.scale[:3]
            command[3:6]=-self.offset[3:6]/self.scale[3:6]
            command[6:]=2*(self.unload_fingers-self.offset[FINGER_ACTIVE])/self.scale[FINGER_ACTIVE]-1
            command=self.admission.bound_action(command,q)
        target=pd_targets(command,q,self.offset,self.scale)
        mount=Rotation.from_euler('XYZ',q[3:6]).as_matrix().T@palm[:3,:3]
        executed=palm.copy();executed[:3,3]+=target[:3]-q[:3]
        executed[:3,:3]=Rotation.from_euler('XYZ',target[3:6]).as_matrix()@mount
        if pre and not unloading and self.phase in ('approach','close'):
            near=bool(args['fingertip_gap_m']<=InteractionConfig().approach_distance_m or args['feedback'].contact_observed)
            rate=self.near_rate_mps if near else self.far_rate_mps
            # Geometry/compliance/native projection has safety priority. Expose
            # any resulting exception rather than silently claiming a hard cap.
            rate_projection_override|=bool(np.linalg.norm(target[:3]-self.last_pd_target[:3])>rate/30+1e-6 or
                np.linalg.norm(target[3:6]-self.last_pd_target[3:6])>self.wrist_joint_rate_radps/30+1e-6 or
                np.max(np.abs(target[FINGER_ACTIVE]-self.last_pd_target[FINGER_ACTIVE]))>self.finger_rate_radps/30+1e-6)
        projection=bool(pre and (np.abs(self.inner_admission.fingers(intent.preshape_q)-intent.preshape_q)>1e-7).any())
        info.update(wrist_target=executed.tolist(),executed_finger_target=target[FINGER_ACTIVE].tolist())
        info['source_ceiling']=dict(contract=self.contract,active=pre,force_valid=valid,
            force_norm_n=magnitudes.tolist(),force_limit_n=self.force_limit_n,
            unloading=unloading,source_paused=unloading,loaded_fingers=np.flatnonzero(loaded).tolist(),
            coasting_joint_indices=np.flatnonzero(lower|upper).tolist(),margin_target_projection=projection,
            inner_target_admitted=self.admission.check(command,q)[0],
            preserves_source_PD_intent=not (unloading or projection or finger_limited or root_limited or (handover_limited and self.phase in ('lift','hold'))),uses_keypose_replacement=False,uses_acquisition_probe=False,
            finger_bandwidth_limited=finger_limited,closure_handover_limited=handover_limited,root_bandwidth_or_compliance_limited=root_limited,
            bandwidth_projection_override=rate_projection_override,emergency_bandwidth_override=unloading,
            command_step_position_m=float(np.linalg.norm(target[:3]-self.last_pd_target[:3])),
            command_step_wrist_joint_rad=float(np.linalg.norm(target[3:6]-self.last_pd_target[3:6])),
            command_step_finger_rad=float(np.max(np.abs(target[FINGER_ACTIVE]-self.last_pd_target[FINGER_ACTIVE]))),
            compliance_scope='PD target tracking offset; not measured velocity or a contact-force guarantee', 
            wrist_tracking_offset_m=float(np.linalg.norm(executed[:3,3]-palm[:3,3])),
            finger_tracking_error_rad=float(np.max(np.abs(target[FINGER_ACTIVE]-q[FINGER_ACTIVE]))))
        self.last_pd_target=target.copy()
        return command,info
