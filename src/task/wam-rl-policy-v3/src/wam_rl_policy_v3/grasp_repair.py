"""Shared, bounded control fixes; no object/sequence identity or learned updates."""
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
from .pose_consumer import MeasuredGraspConsumer,FINGER_ACTIVE,FINGER_UPPER,closest_wrist_angles

REPAIR_CONTRACT='grasp.sharedR.zero-future.pose-reachability.measured-PI.v1'


def clean_tau_future(future,hand,valid,*,zero):
    """The only arm intervention: replace displacement content, preserve mask.

    All 24 frames are replaced, including the keypose reducer's input. Leaving
    WAM keyposes behind a zero R input would leak the ablated information.
    """
    if future is None:return None,bool(valid)
    result=np.broadcast_to(np.asarray(hand,dtype=np.float32),(24,11,3)).copy() if zero else np.asarray(future,dtype=np.float32).copy()
    return result,bool(valid)


class GoalServo:
    """Object-position PI with pose-difference velocity and bounded anti-windup.

    Native contact solver velocities can disagree with finite pose changes. Use
    measured position history at the actual control clock for this outer loop.
    Native PD still receives real measured q/dq; no state or force is fabricated.
    """
    def __init__(self):self.reset()
    def reset(self):
        self.integral=np.zeros(3);self.previous=None;self.time=None;self.velocity=np.zeros(3)
    def step(self,goal,position,wrist_velocity,now):
        position=np.asarray(position);error=np.asarray(goal)-position
        dt=0. if self.time is None else now-self.time
        if dt<0 or dt>1/30+1e-8:raise ValueError('continuous measured servo clock required')
        if dt:
            self.velocity=.8*self.velocity+.2*(position-self.previous)/dt
            # Stop integrating a saturated coordinate in the same direction.
            free=np.abs(.7*error+self.integral)<.025
            self.integral=np.clip(self.integral+np.where(free,.30*error*dt,0.),-.025,.025)
        command=np.clip(.7*error+self.integral-.025*self.velocity-.010*np.asarray(wrist_velocity),-.025,.025)
        self.previous=position.copy();self.time=float(now)
        return command


def support_plane(table_pose,bounds):
    r=Rotation.from_quat(np.asarray(table_pose)[3:7]).as_matrix();bounds=np.asarray(bounds)
    axis=int(np.argmin(bounds[1]-bounds[0]));sign=1. if r[2,axis]>=0 else -1.
    n=sign*r[:,axis];local=np.zeros(3);local[axis]=bounds[1,axis] if sign>0 else bounds[0,axis]
    if n[2]<.5:raise ValueError('upward CAD support normal required')
    return n,float(n@(np.asarray(table_pose)[:3]+r@local))


class ReachabilityFilter:
    """Choose safe approach increments from object-frame candidates, not IDs.

    The R proposal is retained when safe. On a support collision it competes
    with six object-axis tangent approach candidates and an upward retreat.
    Actual limited-joint targets are rate limited; the physical abort is intact.
    """
    def project(self,palm,target,object_pose,q,dq,fingers,*,table_z=None,plane=None,hand_local=None,palm_vertices=None,table_geometry=None):
        normal,height=(np.array([0.,0.,1.]),float(table_z)) if plane is None else plane
        raw_delta=target[:3,3]-palm[:3,3]
        mount=Rotation.from_euler('XYZ',q[3:6]).as_matrix().T@palm[:3,:3]
        raw_angles=closest_wrist_angles(target[:3,:3],q[3:6],mount)
        local=np.zeros((1,3)) if hand_local is None else np.asarray(hand_local)
        if palm_vertices is not None:local=np.r_[local,palm_vertices]
        proposed_bottom=float(np.min((local@target[:3,:3].T+target[:3,3])@normal))
        measured=local@palm[:3,:3].T+palm[:3,3]
        mask=np.ones(len(measured),bool)
        if table_geometry is not None:
            table_pose,bounds=table_geometry;axes=np.argsort(bounds[1]-bounds[0])[1:]
            table_local=(measured-table_pose[:3,3])@table_pose[:3,:3]
            mask=((table_local[:,axes]>=bounds[0,axes])&(table_local[:,axes]<=bounds[1,axes])).all(-1)
        actual_bottom=float(np.min(measured[mask]@normal)) if mask.any() else np.inf
        risk=bool(actual_bottom<height-.015) if hand_local is not None else bool(proposed_bottom<height-.005 or np.linalg.norm(raw_delta)>.30)
        if not risk:
            directions=np.r_[object_pose[:3,:3].T,-object_pose[:3,:3].T,normal[None]]
            return target.copy(),np.asarray(fingers).copy(),dict(candidate_count=8,selected_candidate=0,directions_world=directions.tolist(),support_required_delta_m=None,filter_active=False,proposal_delta_m=raw_delta.tolist(),filtered_delta_m=raw_delta.tolist(),actual_support_gap_m=float(actual_bottom-height) if np.isfinite(actual_bottom) else None)
        delta=raw_delta.copy()
        if risk:
            length=np.linalg.norm(delta);delta*=min(1.,.025/max(length,1e-12))
            angles=q[3:6]+np.clip(raw_angles-q[3:6]-.010*dq[3:6],-.10,.10)
        else:angles=raw_angles
        rotation=Rotation.from_euler('XYZ',angles).as_matrix()@mount
        directions=np.r_[object_pose[:3,:3].T,-object_pose[:3,:3].T,normal[None]]
        local=np.zeros((1,3)) if hand_local is None else np.asarray(hand_local)
        if palm_vertices is not None:local=np.r_[local,palm_vertices]
        # A few millimetres of planned clearance prevents commanded table plunge;
        # actual safety remains the original 100mrad / 20mm physical gate.
        required=height+.003-float(np.min((local@rotation.T+palm[:3,3])@normal)) if risk else -np.inf
        alternatives=[delta]
        for direction in directions:
            travel=.025*direction
            # Favor displacement toward the object's current center.
            if np.dot(travel,object_pose[:3,3]-palm[:3,3])<0:travel*=.25
            alternatives.append(travel)
        costs=[];projected=[]
        for i,value in enumerate(alternatives):
            if risk:
                value=value+normal*max(required-float(value@normal),0.)
                norm=np.linalg.norm(value);value*=min(1.,.025/max(norm,1e-12))
            deficit=max(required-float(value@normal),0.)
            cost=np.linalg.norm(value-delta)+10*deficit+(0. if i==0 else .005)
            projected.append(value);costs.append(cost)
        chosen=int(np.argmin(costs));out=np.eye(4);out[:3,:3]=rotation;out[:3,3]=palm[:3,3]+projected[chosen]
        desired=np.asarray(fingers).copy()
        if risk:
            desired=np.clip(desired-.010*dq[FINGER_ACTIVE],.06,FINGER_UPPER)
            desired=np.clip(desired,q[FINGER_ACTIVE]-.08,q[FINGER_ACTIVE]+.08)
            desired=np.clip(desired,.06,FINGER_UPPER)
        return out,desired,dict(candidate_count=len(alternatives),selected_candidate=chosen,directions_world=directions.tolist(),support_required_delta_m=float(required) if np.isfinite(required) else None,filter_active=risk,
            proposal_delta_m=(target[:3,3]-palm[:3,3]).tolist(),filtered_delta_m=projected[chosen].tolist())


class RepairedGraspConsumer(MeasuredGraspConsumer):
    """Same measured phases and R for both arms; shared reach filter and PI."""
    def __init__(self,*args,plane,palm_vertices,table_geometry=None,object_points=None,**kwargs):
        super().__init__(*args,**kwargs);self.servo=GoalServo();self.reach=ReachabilityFilter();self.plane=plane
        self.palm_vertices=np.asarray(palm_vertices);self.servo_active=False;self.table_geometry=table_geometry;self.object_points=object_points;self.recovery=False;self.recovery_rotation=None
    def step(self,**args):
        intent=args['intent'];palm=args['palm'];q=np.asarray(args['q']);dq=np.asarray(args['dq']);obj=args['object_pose']
        reach_info=None
        if self.phase in ('approach','close'):
            hand=np.asarray(args.pop('hand'));local=(hand-palm[:3,3])@palm[:3,:3]
            target,fingers,reach_info=self.reach.project(palm,obj@intent.approach_pose_object,obj,q,dq,intent.preshape_q,
                plane=self.plane,hand_local=local,palm_vertices=self.palm_vertices,table_geometry=self.table_geometry)
            mount=Rotation.from_euler('XYZ',q[3:6]).as_matrix().T@palm[:3,:3]
            support_gap=reach_info.get('actual_support_gap_m')
            pushing_support=bool(support_gap is not None and support_gap<.010 and (obj@intent.approach_pose_object-palm)[2,3]<-.15 and self.phase=='approach')
            joint_pressure=bool((q[FINGER_ACTIVE]<-.035).any() or (q[FINGER_ACTIVE]>FINGER_UPPER+.06).any() or q[16]<-.035 or q[17]<-.035)
            if self.object_points is not None and not args['feedback'].contact_observed and (pushing_support or joint_pressure):
                self.recovery=True
            if self.recovery:
                desired,choice=geometric_grasp_candidate(palm,hand,obj,self.object_points,*self.plane,obj@intent.grasp_pose_object)
                # First unload a table-loaded finger chain, then follow a current
                # object candidate. Phase still advances ONLY on actual sensors.
                if joint_pressure or (support_gap is not None and support_gap<-.003):
                    desired=palm.copy();desired[:3,3]+=self.plane[0]*.06
                    choice['unload_support']=True
                angles=closest_wrist_angles(desired[:3,:3],q[3:6],mount)
                angles=q[3:6]+np.clip(.4*(angles-q[3:6])-.010*dq[3:6],-.10,.10)
                target=np.eye(4);target[:3,:3]=Rotation.from_euler('XYZ',angles).as_matrix()@mount
                delta=np.clip(.5*(desired[:3,3]-palm[:3,3])-.010*dq[:3],-.025,.025)
                target[:3,3]=palm[:3,3]+delta
                fingers=np.clip(intent.preshape_q-.010*dq[FINGER_ACTIVE],.06,FINGER_UPPER)
                if self.phase=='close':fingers=np.maximum(fingers,q[FINGER_ACTIVE]+.003)
                fingers=np.clip(np.clip(fingers,q[FINGER_ACTIVE]-.08,q[FINGER_ACTIVE]+.08),.06,FINGER_UPPER)
                intent=replace(intent,approach_pose_object=np.linalg.inv(obj)@target,grasp_pose_object=np.linalg.inv(obj)@desired,
                    preshape_q=fingers,approach_joint_angles=angles)
                reach_info['recovery_candidate']=choice;reach_info['filter_active']=True
            elif reach_info['filter_active']:
                intent=replace(intent,approach_pose_object=np.linalg.inv(obj)@target,preshape_q=fingers,
                    approach_joint_angles=closest_wrist_angles(target[:3,:3],q[3:6],mount))
        else:args.pop('hand')
        args['intent']=intent;command,info=super().step(**args)
        if self.phase in ('lift','hold'):
            delta=self.servo.step(intent.postgrasp_object_pose[:3,3],obj[:3,3],dq[:3],args['now'])
            command[:3]=(delta-self.offset[:3])/self.scale[:3]
            info['wrist_target']=np.asarray(info['wrist_target']).tolist()
            for j in range(3):info['wrist_target'][j][3]=float(palm[j,3]+delta[j])
            info['servo_integral_m']=self.servo.integral.tolist();info['servo_pose_velocity_mps']=self.servo.velocity.tolist()
            self.servo_active=True
        elif self.servo_active:self.servo.reset();self.servo_active=False
        info['reachability']=reach_info;info['repair_contract']=REPAIR_CONTRACT
        return command,info


def geometric_grasp_candidate(palm,hand,object_pose,points,normal,height,hint):
    """Object-pose candidates scored by opposing measured fingertips and support.

    No source commands, initial-state ID, object label, or privileged future.
    The optional hint is the SAME tau-derived keypose for either input arm.
    """
    local=(np.asarray(hand)-palm[:3,3])@palm[:3,:3]
    world=np.asarray(points)@object_pose[:3,:3].T+object_pose[:3,3]
    center=world.mean(0);candidates=[]
    # Current orientation plus the tau wrist hint, with object-frame half turns.
    for base in (palm[:3,:3],hint[:3,:3]):
        for axis in range(3):
            for angle in (0.,np.pi):
                r=Rotation.from_rotvec(object_pose[:3,axis]*angle).as_matrix()@base
                for tip in (4,6,8,10):
                    position=center-r@((local[2]+local[tip])*.5)
                    # Limit the whole hand's planned support penetration.
                    bottom=float(np.min((local@r.T+position)@normal));position+=normal*max(height+.003-bottom,0.)
                    proposed=local@r.T+position;gap=np.linalg.norm(proposed[[2,tip],None]-world[None],axis=-1).min(-1)
                    angle_cost=np.linalg.norm(Rotation.from_matrix(r@palm[:3,:3].T).as_rotvec())
                    cost=float(gap.sum()+.015*angle_cost+.05*np.linalg.norm(position-palm[:3,3]))
                    result=np.eye(4);result[:3,:3]=r;result[:3,3]=position
                    candidates.append((cost,result,axis,angle,tip))
    selected=min(candidates,key=lambda x:x[0])
    return selected[1],dict(count=len(candidates),score=selected[0],object_axis=selected[2],angle_rad=selected[3],opposing_tip=selected[4])
