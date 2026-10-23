"""Pre-command feasibility for Inspire's floating XYZ wrist and coupled fingers.

This is NOT a serial-arm IK or a collision/force-closure certificate. Nominal
URDF limits remain unchanged; an inner envelope constrains reference targets.
"""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from .inspire_contract import ACTIVE
from .pose_consumer import FINGER_ACTIVE, closest_wrist_angles
from .tau_replay_contract import pd_targets

COUPLINGS=((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8))


class UnreachableReference(ValueError):
    pass


def limit_categories(q,target,limits):
    q,target=np.asarray(q),np.asarray(target)
    mask=np.asarray(limits['has_limits'],bool);lo=np.asarray(limits['lower']);hi=np.asarray(limits['upper'])
    outside=mask&((q<lo)|(q>hi))
    near=mask&~outside&(np.minimum(q-lo,hi-q)<=.10)
    command=mask&((target<lo)|(target>hi))
    return dict(true_outside=np.flatnonzero(outside).tolist(),near_inside=np.flatnonzero(near).tolist(),
                command_outside_state_inside=np.flatnonzero(command&~outside).tolist(),
                physical_gate_outside=np.flatnonzero(mask&((q<lo-.10)|(q>hi+.10))).tolist(),
                continuous_wrist_q_rad=q[3:6].tolist())


class JointEnvelope:
    fingers=FINGER_ACTIVE

    def __init__(self,limits,offset,scale):
        self.limits=limits;self.mask=np.asarray(limits['has_limits'],bool)
        self.nominal_lower=np.asarray(limits['lower'],float);self.nominal_upper=np.asarray(limits['upper'],float)
        self.offset,self.scale=np.asarray(offset,float),np.asarray(scale,float)
        if any(x.shape!=(18,) for x in (self.mask,self.nominal_lower,self.nominal_upper,self.offset,self.scale)):
            raise ValueError('18 native DOFs required')
        if not np.isfinite(np.r_[self.nominal_lower,self.nominal_upper,self.offset,self.scale]).all() or (self.scale<=0).any():
            raise ValueError('finite valid native controller and bounds required')
        if not self.mask[:3].all() or self.mask[3:6].any():
            raise ValueError('only floating XYZ translation + continuous XYZ wrist is certified')
        if (self.nominal_upper[self.mask]<=self.nominal_lower[self.mask]).any():raise ValueError('nonempty joint intervals required')
        margin=np.minimum(np.where(np.arange(18)<3,.02,.02),.10*(self.nominal_upper-self.nominal_lower))
        self.lower=self.nominal_lower+np.where(self.mask,margin,0.)
        self.upper=self.nominal_upper-np.where(self.mask,margin,0.)

    def check(self,target):
        target=np.asarray(target)
        if target.shape!=(18,) or not np.isfinite(target).all():raise UnreachableReference('nonfinite/native-shape configuration')
        return np.flatnonzero(self.mask&((target<self.lower-1e-7)|(target>self.upper+1e-7))).tolist()

    def finger_intervals(self):
        low=np.maximum(self.lower[self.fingers],self.offset[self.fingers])
        high=np.minimum(self.upper[self.fingers],self.offset[self.fingers]+self.scale[self.fingers])
        for dst,src,ratio in COUPLINGS:
            if self.mask[dst]:
                k=int(np.flatnonzero(self.fingers==src)[0])
                low[k]=max(low[k],self.lower[dst]/ratio);high[k]=min(high[k],self.upper[dst]/ratio)
        if (low>high).any():raise UnreachableReference('empty coupled driver interval')
        return low,high

    def pose_configuration(self,target,q,palm,fingers):
        q=np.asarray(q,float);target=np.asarray(target,float);palm=np.asarray(palm,float)
        if target.shape!=(4,4) or not np.isfinite(target).all():raise UnreachableReference('finite SE3 required')
        if not np.allclose(target[3],[0,0,0,1]) or not np.allclose(target[:3,:3].T@target[:3,:3],np.eye(3),atol=1e-6) or np.linalg.det(target[:3,:3])<.999999:
            raise UnreachableReference('proper homogeneous pose required')
        # This local chain has palm origin at the virtual XYZ translation.
        # Refuse a different robot/offset instead of pretending this is arm IK.
        if not np.allclose(palm[:3,3],q[:3],atol=1e-4,rtol=0):raise UnreachableReference('floating-palm calibration mismatch')
        mount=Rotation.from_euler('XYZ',q[3:6]).as_matrix().T@palm[:3,:3]
        goal=q.copy();goal[:3]=q[:3]+target[:3,3]-palm[:3,3]
        goal[3:6]=closest_wrist_angles(target[:3,:3],q[3:6],mount)
        goal[self.fingers]=fingers
        for dst,src,ratio in COUPLINGS:goal[dst]=ratio*goal[src]
        bad=self.check(goal)
        low,high=self.finger_intervals()
        if bad or ((goal[self.fingers]<low-1e-7)|(goal[self.fingers]>high+1e-7)).any():
            raise UnreachableReference('outside inner/coupled workspace: joints='+str(bad))
        return goal

    def admit_trajectory(self,curve,hand_object,q,palm,fingers):
        from .object_tracking import hand_setpoint
        ticks=np.minimum(np.arange(int(np.ceil(curve.times[-1]*30))+1)/30,curve.times[-1])
        times=np.unique(np.r_[curve.times,ticks]);minimum=np.inf;minimum_joint=None
        for t in times:
            desired,v,w=curve.evaluate(t);target,_,_=hand_setpoint(desired,v,w,hand_object)
            try:goal=self.pose_configuration(target,q,palm,fingers)
            except UnreachableReference as error:return dict(admitted=False,time_s=float(t),reason=str(error),checked_points=len(times))
            margins=np.minimum(goal-self.nominal_lower,self.nominal_upper-goal);margins[~self.mask]=np.inf
            j=int(np.argmin(margins))
            if margins[j]<minimum:minimum=float(margins[j]);minimum_joint=j
        return dict(admitted=True,checked_points=len(times),minimum_planned_nominal_margin=minimum,minimum_joint=minimum_joint,
                    scope='native control ticks + original spline knots; no swept collision or dynamic guarantee')

    def protect(self,action,q,dq):
        q,dq=np.asarray(q,float),np.asarray(dq,float);action=np.asarray(action,float)
        if action.shape!=(12,) or not np.isfinite(np.r_[action,q,dq]).all():raise UnreachableReference('nonfinite command/state')
        # pd_targets is the SAME clipped canonical conversion as production.
        proposed=pd_targets(action,q,self.offset,self.scale);target=proposed.copy()
        predicted=q+.10*dq
        risk=self.mask&(((predicted<self.lower)&(dq<-.01))|((predicted>self.upper)&(dq>.01)))
        if risk.any():
            # Stop approaching the obstruction; retain the grip, then drive the
            # affected independent/mimic group inward instead of opening blindly.
            target[:6]=q[:6]
        low,high=self.finger_intervals()
        desired=target[self.fingers].copy()
        for k,src in enumerate(self.fingers):
            for j,ratio in [(int(src),1.)]+[(d,r) for d,s,r in COUPLINGS if s==src]:
                if risk[j] and dq[j]<0:desired[k]=max(desired[k],(self.lower[j]+.02)/ratio)
                if risk[j] and dq[j]>0:desired[k]=min(desired[k],(self.upper[j]-.02)/ratio)
        target[self.fingers]=np.clip(desired,low,high)
        target[:3]=np.clip(target[:3],self.lower[:3],self.upper[:3])
        for dst,src,ratio in COUPLINGS:target[dst]=ratio*target[src]
        raw=np.r_[(target[:6]-q[:6]-self.offset[:6])/self.scale[:6],
                  2*(target[self.fingers]-self.offset[self.fingers])/self.scale[self.fingers]-1]
        if (np.abs(raw)>1+1e-6).any():raise UnreachableReference('safe native command is not representable')
        result=np.clip(raw,-1,1).astype(np.float32)
        actual=pd_targets(result,q,self.offset,self.scale)
        if self.check(actual):raise UnreachableReference('post-conversion target failed admission')
        return result,dict(braking=bool(risk.any()),risk_joints=np.flatnonzero(risk).tolist(),
            target_projected=bool(np.max(np.abs(actual-proposed))>2e-6),canonical_proposal_outside=bool((np.abs(action)>1).any()),
            proposed_pd=proposed.tolist(),executed_pd=actual.tolist(),categories=limit_categories(q,actual,self.limits))


class ProgressWatchdog:
    """One-second real-time window, never count time spent waiting as progress."""
    def __init__(self):self.samples=deque();self.latched=None
    def update(self,now,measured,target):
        if self.latched:return self.latched
        self.samples.append((float(now),np.asarray(measured).copy(),np.asarray(target).copy()))
        while len(self.samples)>1 and now-self.samples[1][0]>=1.-1e-9:self.samples.popleft()
        start,old,old_target=self.samples[0]
        e=float(np.linalg.norm(target[:3,3]-measured[:3,3]));a=float(Rotation.from_matrix(target[:3,:3]@measured[:3,:3].T).magnitude())
        if now-start<1.-1e-9 or (e<=.015 and a<=np.deg2rad(5)):return None
        motion=float(np.linalg.norm(measured[:3,3]-old[:3,3]));turn=float(Rotation.from_matrix(measured[:3,:3]@old[:3,:3].T).magnitude())
        ref_motion=float(np.linalg.norm(target[:3,3]-old_target[:3,3]));ref_turn=float(Rotation.from_matrix(target[:3,:3]@old_target[:3,:3].T).magnitude())
        stagnant=motion<.002 and turn<np.deg2rad(.5)
        if ref_motion<.001 and ref_turn<np.deg2rad(.5):
            old_e=np.linalg.norm(old_target[:3,3]-old[:3,3]);old_a=Rotation.from_matrix(old_target[:3,:3]@old[:3,:3].T).magnitude()
            stagnant=stagnant or (old_e-e<.001 and old_a-a<np.deg2rad(.5))
        if stagnant:self.latched='UNREACHABLE_STALL'
        return self.latched
