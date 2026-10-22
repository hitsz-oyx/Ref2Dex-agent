"""Size-derived grasp geometry, measured SE3 servo and native target admission.

All lengths use the same canonical object points (metres). Constants are global
fractions/rate limits; no object names, source frames or initial-state IDs enter.
"""
from dataclasses import dataclass,asdict,replace
import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from .pose_consumer import MeasuredGraspConsumer,FINGER_ACTIVE,closest_wrist_angles
from .grasp_repair import GoalServo
from .tau_replay_contract import pd_targets

GEOMETRY_CONTRACT='grasp.object-extent.relative-SE3.native-limit-admission.v1'
COUPLINGS=((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8))


@dataclass(frozen=True)
class GraspGeometry:
    minimum_extent_m:float
    diameter_m:float
    position_tolerance_m:float
    palm_gap_m:float
    tip_gap_m:float
    wrist_gap_m:float
    angle_tolerance_rad:float
    point_count:int

    @classmethod
    def from_points(cls,points):
        points=np.asarray(points,float)
        if points.ndim!=2 or points.shape[1]!=3 or len(points)<4 or not np.isfinite(points).all():raise ValueError('finite 3D object geometry required')
        # PCA gives a geometry frame independent of world orientation. Sparse
        # canonical surface coverage is a documented approximation to true CAD.
        hull=ConvexHull(points);normals=hull.equations[:,:3]
        width=float(np.ptp(points@normals.T,axis=0).min());diameter=float(np.linalg.norm(points[:,None]-points[None],axis=-1).max())
        if width<=1e-6 or diameter<=1e-6:raise ValueError('nondegenerate volumetric object required')
        tol=.10*width
        return cls(width,diameter,tol,.50*diameter,.25*width,.25*width,float(2*np.arcsin(min(1.,tol/diameter))),len(points))

    def arrived(self,position_error,angle_error=0.,position_only=True):
        return bool(position_error<=self.position_tolerance_m and (position_only or angle_error<=self.angle_tolerance_rad))

    def paired_tip_gap(self,hand_wam,object_pose,points):
        world=np.asarray(points)@object_pose[:3,:3].T+object_pose[:3,3]
        tips=np.asarray(hand_wam)[6:]
        gaps=np.linalg.norm(tips[:,None]-world[None],axis=-1).min(-1)
        return float(np.partition(gaps,1)[1])

    def manifest(self):return dict(contract=GEOMETRY_CONTRACT,tip_measurement='second-closest of five native fingertips in WAM order; two-finger grasp',**asdict(self))


class JointAdmission:
    """Check all 18 native targets, including passive coupling, before ranking."""
    def __init__(self,limits,offset,scale):
        self.mask=np.asarray(limits['has_limits'],bool);self.lower=np.asarray(limits['lower'],float);self.upper=np.asarray(limits['upper'],float)
        self.offset=np.asarray(offset);self.scale=np.asarray(scale)
        if any(x.shape!=(18,) for x in (self.mask,self.lower,self.upper,self.offset,self.scale)):raise ValueError('native 18-DOF bounds/controller required')
        self.active_lower=self.lower[FINGER_ACTIVE].copy();self.active_upper=self.upper[FINGER_ACTIVE].copy()
        for dst,src,ratio in COUPLINGS:
            i=np.flatnonzero(FINGER_ACTIVE==src)[0]
            if self.mask[dst]:
                self.active_lower[i]=max(self.active_lower[i],self.lower[dst]/ratio)
                self.active_upper[i]=min(self.active_upper[i],self.upper[dst]/ratio)

    def check(self,action,q):
        action=np.asarray(action)
        if action.shape!=(12,) or not np.isfinite(action).all() or (np.abs(action)>1+1e-6).any():return False,["canonical_action_range"]
        target=pd_targets(action,q,self.offset,self.scale)
        bad=np.flatnonzero(self.mask&((target<self.lower-1e-7)|(target>self.upper+1e-7)))
        return not len(bad),bad.tolist()

    def fingers(self,target):return np.clip(target,self.active_lower,self.active_upper)

    def bound_action(self,action,q):
        # Final containment protects force-regulated closure and pose servo too;
        # candidate feasibility is separately checked BEFORE candidate selection.
        target=pd_targets(action,q,self.offset,self.scale);target[:3]=np.clip(target[:3],self.lower[:3],self.upper[:3])
        target[FINGER_ACTIVE]=self.fingers(target[FINGER_ACTIVE])
        result=np.asarray(action,float).copy();result[:3]=(target[:3]-q[:3]-self.offset[:3])/self.scale[:3]
        result[6:]=2*(target[FINGER_ACTIVE]-self.offset[FINGER_ACTIVE])/self.scale[FINGER_ACTIVE]-1
        result=np.clip(result,-1.,1.)
        if not self.check(result,q)[0]:raise ValueError('no feasible native bounded target')
        return result.astype(np.float32)


def action_for_pose(palm,target,q,fingers,admission,mount):
    angles=closest_wrist_angles(target[:3,:3],q[3:6],mount)
    action=np.r_[(target[:3,3]-palm[:3,3]-admission.offset[:3])/admission.scale[:3],
                 (angles-q[3:6]-admission.offset[3:6])/admission.scale[3:6],
                 2*(fingers-admission.offset[FINGER_ACTIVE])/admission.scale[FINGER_ACTIVE]-1]
    return action,angles


def grasp_candidates(palm,hand,obj,points,normal,height,hint,fingers,q,admission):
    """PCA object axes × proper signed permutations × opposing fingertip pairs.

    Rank only candidates with feasible native PD targets (including mimic DOFs).
    This is a point/pose approximation, not articulated collision or force closure.
    """
    local=(hand-palm[:3,3])@palm[:3,:3];points=np.asarray(points)
    _,axes=np.linalg.eigh(np.cov(points.T))
    if np.linalg.det(axes)<0:axes[:,0]*=-1
    world=points@obj[:3,:3].T+obj[:3,3];center=world.mean(0)
    mount=Rotation.from_euler('XYZ',q[3:6]).as_matrix().T@palm[:3,:3]
    rotations=[palm[:3,:3],hint[:3,:3]]
    # Map measured thumb/opposing-tip frame onto geometry axes, rather than only
    # rotating the current wrist by half turns (which misses side approaches).
    for tip in (4,6,8,10):
        x=local[tip]-local[2];x/=max(np.linalg.norm(x),1e-12)
        z=np.array([0.,0.,1.]);z-=x*(x@z)
        if np.linalg.norm(z)<1e-6:z=np.array([0.,1.,0.])-x*x[1]
        z/=np.linalg.norm(z);frame=np.stack((x,np.cross(z,x),z),axis=1)
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((-1.,1.),repeat=3):
                basis=axes[:,perm]*np.array(signs)
                if np.linalg.det(basis)>.5:rotations.append(obj[:3,:3]@basis@frame.T)
    candidates=[];rejected=0;count=0
    for r in rotations:
        for tip in (4,6,8,10):
            position=center-r@((local[2]+local[tip])*.5)
            position+=normal*max(height-float(np.min((local@r.T+position)@normal)),0.)
            target=np.eye(4);target[:3,:3]=r;target[:3,3]=position
            raw,angles=action_for_pose(palm,target,q,fingers,admission,mount);count+=1
            feasible,bad=admission.check(raw,q)
            if not feasible or (np.abs(raw)>1+1e-6).any():rejected+=1;continue
            proposed=local@r.T+position;gap=np.linalg.norm(proposed[[2,tip],None]-world[None],axis=-1).min(-1)
            turn=np.linalg.norm(Rotation.from_matrix(r@palm[:3,:3].T).as_rotvec())
            score=float(gap.sum()+.015*turn+.05*np.linalg.norm(position-palm[:3,3]))
            candidates.append((score,target,angles,tip))
    if not candidates:return None,dict(count=count,admitted=0,rejected_limits=rejected,status='NO_ADMISSIBLE_CANDIDATE')
    score,target,angles,tip=min(candidates,key=lambda x:x[0])
    return target,dict(count=count,admitted=len(candidates),rejected_limits=rejected,status='ADMITTED',score=score,opposing_tip=tip)


class RelativePoseServo:
    def __init__(self,*,linear_rate_mps=.75,angular_rate_radps=2.4):
        self.linear_rate=linear_rate_mps;self.angular_rate=angular_rate_radps;self.reset()
    def reset(self):
        self.translation=GoalServo();self.grasp_relative=None;self.object_rotation=None;self.previous_time=None
    def applied_translation(self,delta):
        # Tracking anti-windup sees the FINAL admitted command, including the
        # norm limiter, canonical saturation and native position projection.
        self.translation.integral=np.clip(self.translation.integral+np.asarray(delta)-self.requested_delta,-.025,.025)

    def step(self,goal,obj,palm,q,dq,mount,now,position_only):
        if self.grasp_relative is None:
            self.grasp_relative=np.linalg.inv(obj)@palm;self.object_rotation=obj[:3,:3].copy()
        effective=np.asarray(goal).copy()
        if position_only:effective[:3,:3]=self.object_rotation
        measured_relative=np.linalg.inv(obj)@palm
        desired=effective@measured_relative
        # Desired wrist is refreshed from the measured object-relative transform, including
        # the lever-arm translation when the object goal rotates.
        dt=1/30 if self.previous_time is None else now-self.previous_time
        if dt<=0 or dt>1/30+1e-8:raise ValueError('continuous pose-servo clock required')
        delta=self.translation.step(desired[:3,3],palm[:3,3],dq[:3],now)
        self.requested_delta=delta.copy()
        delta*=min(1.,self.linear_rate*dt/max(np.linalg.norm(delta),1e-12))
        rotvec=Rotation.from_matrix(desired[:3,:3]@palm[:3,:3].T).as_rotvec()
        angular=.5*rotvec-.006*np.asarray(dq[3:6]);angular*=min(1.,self.angular_rate*dt/max(np.linalg.norm(angular),1e-12))
        target=palm.copy();target[:3,3]+=delta;target[:3,:3]=Rotation.from_rotvec(angular).as_matrix()@palm[:3,:3]
        angles=closest_wrist_angles(target[:3,:3],q[3:6],mount)
        self.previous_time=now
        return target,angles,dict(grasp_relative=self.grasp_relative.tolist(),measured_grasp_relative=measured_relative.tolist(),relative_slip_m=float(np.linalg.norm(measured_relative[:3,3]-self.grasp_relative[:3,3])),desired_wrist=desired.tolist(),linear_step_m=float(np.linalg.norm(delta)),angular_step_rad=float(np.linalg.norm(angular)),linear_rate_limit_mps=self.linear_rate,angular_rate_limit_radps=self.angular_rate,
            object_angle_error_rad=float(np.linalg.norm(Rotation.from_matrix(effective[:3,:3]@obj[:3,:3].T).as_rotvec())))


class GeometryGraspConsumer(MeasuredGraspConsumer):
    def __init__(self,*args,plane,palm_vertices,object_points,joint_limits,table_geometry=None,**kwargs):
        super().__init__(*args,**kwargs);self.geometry=GraspGeometry.from_points(object_points)
        self.points=np.asarray(object_points);self.plane=plane;self.palm_vertices=np.asarray(palm_vertices)
        self.admission=JointAdmission(joint_limits,self.offset,self.scale);self.servo=RelativePoseServo();self.servo_active=False
        self.candidate=None;self.table_geometry=table_geometry
    def step(self,**args):
        hand=np.asarray(args.pop('hand'));intent=args['intent'];q=np.asarray(args['q']);palm=args['palm'];obj=args['object_pose'];reach_info=None
        mount=Rotation.from_euler('XYZ',q[3:6]).as_matrix().T@palm[:3,:3]
        if self.phase in ('approach','close'):
            proposal=obj@intent.approach_pose_object
            raw,_=action_for_pose(palm,proposal,q,intent.preshape_q,self.admission,mount)
            safe,bad=self.admission.check(raw,q)
            local=(hand-palm[:3,3])@palm[:3,:3];normal,height=self.plane
            measured=np.r_[hand,self.palm_vertices@palm[:3,:3].T+palm[:3,3]]
            mask=np.ones(len(measured),bool)
            if self.table_geometry is not None:
                table_pose,bounds=self.table_geometry;axes=np.argsort(bounds[1]-bounds[0])[1:]
                table_local=(measured-table_pose[:3,3])@table_pose[:3,:3]
                mask=((table_local[:,axes]>=bounds[0,axes])&(table_local[:,axes]<=bounds[1,axes])).all(-1)
            measured_gap=float(np.min(measured[mask]@normal)-height) if mask.any() else np.inf
            # Coasting beyond a true bound is an admission warning, independent
            # of the more permissive unchanged physical abort margin.
            projected=q+np.asarray(args['dq'])/30
            pressure=bool((self.admission.mask&((projected<self.admission.lower)|(projected>self.admission.upper))).any())
            geometry_needed=not safe or pressure or measured_gap<0
            choice=dict(status='R_ADMITTED',rejected_joint_indices=bad,measured_support_gap_m=measured_gap if np.isfinite(measured_gap) else None)
            if geometry_needed:
                fingers=self.admission.fingers(intent.preshape_q)
                desired,choice=grasp_candidates(palm,hand,obj,self.points,normal,height,obj@intent.grasp_pose_object,fingers,q,self.admission)
                if desired is None:
                    # No feasible grasp: bounded retreat, recorded as no candidate,
                    # never a success or silent dropped episode.
                    desired=palm.copy();desired[:3,3]+=normal*self.geometry.minimum_extent_m*.25
                self.candidate=desired
                target=desired.copy();delta=target[:3,3]-palm[:3,3];delta*=min(1.,.75/30/max(np.linalg.norm(delta),1e-12));target[:3,3]=palm[:3,3]+delta
                rv=Rotation.from_matrix(target[:3,:3]@palm[:3,:3].T).as_rotvec();rv*=min(1.,2.4/30/max(np.linalg.norm(rv),1e-12));target[:3,:3]=Rotation.from_rotvec(rv).as_matrix()@palm[:3,:3]
                raw,angles=action_for_pose(palm,target,q,fingers,self.admission,mount);raw=self.admission.bound_action(raw,q)
                # Actual wrist target and IK branch follow the admitted command.
                target[:3,3]=palm[:3,3]+self.offset[:3]+self.scale[:3]*raw[:3]
                intent=replace(intent,approach_pose_object=np.linalg.inv(obj)@target,grasp_pose_object=np.linalg.inv(obj)@desired,preshape_q=fingers,approach_joint_angles=angles)
            reach_info=dict(filter_active=geometry_needed,candidate=choice,proposal_admitted=safe,coasting_limit_risk=pressure)
        args['intent']=intent;command,info=super().step(**args)
        if self.phase in ('lift','hold'):
            target,angles,servo_info=self.servo.step(intent.postgrasp_object_pose,obj,palm,q,np.asarray(args['dq']),mount,args['now'],intent.position_only)
            command[:3]=(target[:3,3]-palm[:3,3]-self.offset[:3])/self.scale[:3]
            command[3:6]=(angles-q[3:6]-self.offset[3:6])/self.scale[3:6]
            error=np.linalg.norm(intent.postgrasp_object_pose[:3,3]-obj[:3,3])
            self.phase='hold' if self.geometry.arrived(error,servo_info['object_angle_error_rad'],intent.position_only) else 'lift'
            info.update(phase=self.phase,wrist_target=target.tolist(),relative_pose_servo=servo_info);self.servo_active=True
        elif self.servo_active:self.servo.reset();self.servo_active=False
        command=self.admission.bound_action(command,q)
        if self.phase in ('lift','hold'):
            self.servo.applied_translation(self.offset[:3]+self.scale[:3]*command[:3])
            info['relative_pose_servo']['integral_m']=self.servo.translation.integral.tolist()
        info.update(reachability=reach_info,repair_contract=GEOMETRY_CONTRACT,geometry=self.geometry.manifest(),native_target_admitted=self.admission.check(command,q)[0])
        self.last_info=info
        return command,info
