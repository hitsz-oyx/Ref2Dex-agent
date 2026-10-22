"""Measured pre-grasp unloading; deliberately separate from the v13 consumer.

No object IDs, sequence IDs or task labels enter this controller. The held-object
servo and physical failure/scoring contracts are inherited without modification.
"""
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
from .grasp_geometry import GeometryGraspConsumer, JointAdmission, COUPLINGS
from .pose_consumer import FINGER_ACTIVE, closest_wrist_angles

SOURCE_CEILING_CONTRACT = 'source.measured-unload.margin-admission.slew.v1'


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

    def __init__(self,*args,joint_limits,**kwargs):
        super().__init__(*args,joint_limits=joint_limits,**kwargs)
        self.outer_admission=self.admission
        self.inner_admission=MarginAdmission(joint_limits,self.offset,self.scale)
        self.unload_until=-np.inf
        self.unload_direction=None
        self.source_paused=False

    def step(self,**args):
        if self.phase not in ('approach','close'):
            # Same post-grasp control as the registered source baseline.
            self.admission=self.outer_admission
            self.source_paused=False
            command,info=super().step(**args)
            info['source_ceiling']=dict(contract=SOURCE_CEILING_CONTRACT,active=False,source_paused=False)
            return command,info
        self.admission=self.inner_admission
        q=np.asarray(args['q']);dq=np.asarray(args['dq']);palm=np.asarray(args['palm'])
        now=float(args['now']);intent=args['intent'];force=np.asarray(args['forces'])
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
        fingers=np.asarray(intent.preshape_q).copy()
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
        proposal=args['object_pose']@intent.approach_pose_object
        if unloading:
            proposal=palm.copy();proposal[:3,3]+=self.unload_direction*self.reaching_linear_rate_mps/30
        # Pre-limit the source too, not just replacement geometry candidates.
        delta=proposal[:3,3]-palm[:3,3]
        delta*=min(1.,self.reaching_linear_rate_mps/30/max(np.linalg.norm(delta),1e-12))
        proposal[:3,3]=palm[:3,3]+delta
        rotation=Rotation.from_matrix(proposal[:3,:3]@palm[:3,:3].T).as_rotvec()
        rotation*=min(1.,self.reaching_angular_rate_radps/30/max(np.linalg.norm(rotation),1e-12))
        proposal[:3,:3]=Rotation.from_rotvec(rotation).as_matrix()@palm[:3,:3]
        mount=Rotation.from_euler('XYZ',q[3:6]).as_matrix().T@palm[:3,:3]
        changed=replace(intent,approach_pose_object=np.linalg.inv(args['object_pose'])@proposal,
                        preshape_q=fingers,approach_joint_angles=closest_wrist_angles(proposal[:3,:3],q[3:6],mount))
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
        info['source_ceiling']=dict(contract=SOURCE_CEILING_CONTRACT,active=True,
            force_valid=force_valid,force_norm_n=magnitudes.tolist(),force_limit_n=self.force_limit_n,
            loaded_fingers=np.flatnonzero(loaded).tolist(),coasting_joint_indices=np.flatnonzero(risk).tolist(),
            unloading=unloading,source_paused=self.source_paused,inner_target_admitted=self.admission.check(command,q)[0],
            finger_rate_radps=self.finger_rate_radps,retreat_direction=None if not unloading else self.unload_direction.tolist())
        self.last_info=info
        return command,info
