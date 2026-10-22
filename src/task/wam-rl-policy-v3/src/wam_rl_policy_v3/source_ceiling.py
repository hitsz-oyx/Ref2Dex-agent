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
from .tau_replay_contract import pd_targets
from .interaction_state import InteractionConfig

SOURCE_CEILING_CONTRACT = 'source.keygrasp.measured-close.unload.probe.v5'


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
    def __init__(self,reference,points):
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
        if len(found):
            stable=[int(i)+4 for i in found if i+4<len(contact) and contact[i:i+5].all()]
            frame=stable[0] if stable else self.contact_frame
            self.evidence='measured_source_force_contact'
        else:
            frame=int(np.argmin(np.partition(gaps,1,axis=1)[:,1]))
            self.evidence='geometric_candidate_no_source_contact'
        self.grasp_frame=frame;self.preshape_frame=max(0,(self.contact_frame if self.contact_frame is not None else frame)-5)
        self.grasp=np.linalg.inv(pose7(reference['objects'][frame,0]))@pose7(reference['palm_pose'][frame,0])
        self.preshape=np.asarray(reference['q'][self.preshape_frame,0,FINGER_ACTIVE]).copy()
        self.closure=np.asarray(reference['q'][frame,0,FINGER_ACTIVE]).copy()

    def intent(self,object_goal):
        return GraspIntent(self.grasp.copy(),self.grasp.copy(),self.preshape.copy(),self.closure.copy(),
                           np.asarray(object_goal).copy(),'source-keygrasp',False,True)

    def manifest(self):
        return dict(kind='source_keygrasp',evidence=self.evidence,contact_frame=self.contact_frame,
                    grasp_frame=self.grasp_frame,preshape_frame=self.preshape_frame,
                    raw_force_contact_frames=self.raw_force_contact_frames,object_contact_frames=self.object_contact_frames,
                    consumes_source_actions=False,consumes_R=False,source_contact_is_live_success=False)


class MarginAdmission(JointAdmission):
    """Use an inner target workspace, never relax the physical abort bounds."""
    def __init__(self, limits, offset, scale):
        inner={k:list(v) for k,v in limits.items()}
        for i,limited in enumerate(limits['has_limits']):
            if not limited:continue
            margin=min(.04 if i>=6 else .01, .1*(limits['upper'][i]-limits['lower'][i]))
            inner['lower'][i]+=margin;inner['upper'][i]-=margin
        super().__init__(inner,offset,scale)


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
    acquisition_distance_m=.01
    acquisition_rate_mps=.06
    wrist_angle_tolerance_rad=.2

    def __init__(self,*args,joint_limits,**kwargs):
        super().__init__(*args,joint_limits=joint_limits,**kwargs)
        self.outer_admission=self.admission
        self.inner_admission=MarginAdmission(joint_limits,self.offset,self.scale)
        self.unload_until=-np.inf
        self.unload_direction=None
        self.source_paused=False
        self.acquire_anchor=None

    def step(self,**args):
        if self.phase not in ('approach','close'):
            # Same post-grasp control as the registered source baseline.
            self.admission=self.outer_admission
            self.source_paused=False
            self.acquire_anchor=None
            command,info=super().step(**args)
            info['source_ceiling']=dict(contract=SOURCE_CEILING_CONTRACT,active=False,source_paused=False)
            return command,info
        self.admission=self.inner_admission
        q=np.asarray(args['q']);dq=np.asarray(args['dq']);palm=np.asarray(args['palm'])
        now=float(args['now']);intent=args['intent'];force=np.asarray(args['forces'])
        grasp=args['object_pose']@intent.grasp_pose_object
        wrist_arrived=bool(np.linalg.norm(grasp[:3,3]-palm[:3,3])<=self.geometry.wrist_gap_m and
                           np.linalg.norm(Rotation.from_matrix(grasp[:3,:3]@palm[:3,:3].T).as_rotvec())<=self.wrist_angle_tolerance_rad)
        if self.phase=='approach' and wrist_arrived:
            # Closing is an attempted action, not a contact/grasp certificate.
            self.phase='close';self.close_pose=palm.copy();self.fingers=q[FINGER_ACTIVE].copy()
        force_valid=bool(args['feedback'].force_valid)
        magnitudes=np.linalg.norm(force,axis=-1) if force_valid else np.zeros(5)
        loaded=magnitudes>self.force_limit_n
        projected=q+dq/30
        # Directional prediction: harmless motion back into the workspace must
        # not trigger another retreat. Continuous wrist angles have no bounds.
        lower=(self.admission.mask & (projected<self.admission.lower) & (dq<-.2))
        upper=(self.admission.mask & (projected>self.admission.upper) & (dq>.2))
        risk=lower|upper
        trigger=bool(loaded.any() or risk[6:].any())
        if trigger:
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
        if unloading:
            finger_load=np.r_[loaded[:4],loaded[4],loaded[4]]
            fingers=np.where(finger_load,q[FINGER_ACTIVE]-.04,fingers)
            # An actual lower-bound excursion needs inward recovery, not more
            # opening. Mimic-joint warnings act on their independent driver.
            for i,joint in enumerate(FINGER_ACTIVE):
                group=[(joint,1.)]+[(dst,ratio) for dst,src,ratio in COUPLINGS if src==joint]
                if any(lower[j] for j,_ in group):
                    fingers[i]=max(q[j]/ratio+.04 for j,ratio in group)
                if any(upper[j] for j,_ in group):
                    fingers[i]=min(q[j]/ratio-.04 for j,ratio in group)
        fingers=q[FINGER_ACTIVE]+np.clip(fingers-q[FINGER_ACTIVE],-self.finger_rate_radps/30,self.finger_rate_radps/30)
        fingers=self.admission.fingers(fingers)
        closure=q[FINGER_ACTIVE]+np.clip(intent.closure_q-q[FINGER_ACTIVE],-self.finger_rate_radps/30,self.finger_rate_radps/30)
        closure=self.admission.fingers(closure)
        proposal=args['object_pose']@intent.approach_pose_object
        candidate=bool(force_valid and args['feedback'].contact_observed and getattr(args['feedback'],'grasp_candidate',False))
        probing=bool(candidate and not args['feedback'].grasp_established and not unloading)
        if not probing:self.acquire_anchor=None
        if probing:
            if self.acquire_anchor is None:self.acquire_anchor=palm.copy()
            proposal=self.acquire_anchor.copy();proposal[:3,3]+=self.plane[0]*self.acquisition_distance_m
            # Preserve the actual force-bearing pre-shape rather than following
            # a later reference opening command while waiting for acquisition.
            mapped_force=np.r_[magnitudes[:4],magnitudes[4],magnitudes[4]]
            fingers=self.admission.fingers(q[FINGER_ACTIVE]+np.where(mapped_force<.25,.005,0.))
        if unloading:
            proposal=palm.copy();proposal[:3,3]+=self.unload_direction*self.reaching_linear_rate_mps/30
        # Pre-limit the source too, not just replacement geometry candidates.
        delta=proposal[:3,3]-palm[:3,3]
        linear_step=self.acquisition_rate_mps/30 if probing else self.reaching_linear_rate_mps/30
        delta*=min(1.,linear_step/max(np.linalg.norm(delta),1e-12))
        proposal[:3,3]=palm[:3,3]+delta
        rotation=Rotation.from_matrix(proposal[:3,:3]@palm[:3,:3].T).as_rotvec()
        rotation*=min(1.,self.reaching_angular_rate_radps/30/max(np.linalg.norm(rotation),1e-12))
        proposal[:3,:3]=Rotation.from_rotvec(rotation).as_matrix()@palm[:3,:3]
        mount=Rotation.from_euler('XYZ',q[3:6]).as_matrix().T@palm[:3,:3]
        changed=replace(intent,approach_pose_object=np.linalg.inv(args['object_pose'])@proposal,
                        preshape_q=fingers,closure_q=closure,approach_joint_angles=closest_wrist_angles(proposal[:3,:3],q[3:6],mount))
        # Parent geometry selection still checks all candidate targets against
        # the INNER bounds before ranking. It also retains table feasibility.
        command,info=super().step(**dict(args,intent=changed))
        if self.phase in ('approach','close'):
            # Apply slew limits AFTER geometry selection too. Candidate rotation
            # and target admission cannot bypass the approach control budget.
            command[:3]*=min(1.,self.reaching_linear_rate_mps/30/max(np.linalg.norm(command[:3]*self.scale[:3]),1e-12))
            command[3:6]*=min(1.,self.reaching_angular_rate_radps/30/max(np.linalg.norm(command[3:6]*self.scale[3:6]),1e-12))
        if unloading and self.phase in ('approach','close'):
            # A new candidate must not overwrite this tick's measured retreat.
            command[:3]=(proposal[:3,3]-palm[:3,3]-self.offset[:3])/self.scale[:3]
            command[3:6]=-self.offset[3:6]/self.scale[3:6]
            command[6:]=2*(fingers-self.offset[FINGER_ACTIVE])/self.scale[FINGER_ACTIVE]-1
            command=self.admission.bound_action(command,q)
        elif probing and self.phase in ('approach','close'):
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
        finger_delta=float(np.max(np.abs(applied[FINGER_ACTIVE]-q[FINGER_ACTIVE])))
        info['source_ceiling']=dict(contract=SOURCE_CEILING_CONTRACT,active=True,
            force_valid=force_valid,force_norm_n=magnitudes.tolist(),force_limit_n=self.force_limit_n,
            wrist_pose_ready=wrist_arrived,
            loaded_fingers=np.flatnonzero(loaded).tolist(),coasting_joint_indices=np.flatnonzero(risk).tolist(),
            unloading=applied_unloading,unload_requested=unloading,source_paused=self.source_paused,inner_target_admitted=self.admission.check(command,q)[0],
            acquisition_probe=probing,acquisition_anchor=None if self.acquire_anchor is None else self.acquire_anchor.tolist(),acquisition_maximum_m=self.acquisition_distance_m,
            finger_max_applied_delta_rad=finger_delta,finger_rate_override_for_admission=bool(finger_delta>self.finger_rate_radps/30+1e-6),
            finger_rate_radps=self.finger_rate_radps,retreat_direction=None if not unloading else self.unload_direction.tolist())
        self.last_info=info
        return command,info
