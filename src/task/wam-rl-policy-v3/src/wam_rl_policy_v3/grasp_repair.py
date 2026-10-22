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
    def project(self,palm,target,object_pose,q,dq,fingers,*,table_z=None,plane=None,hand_local=None,palm_vertices=None):
        normal,height=(np.array([0.,0.,1.]),float(table_z)) if plane is None else plane
        delta=target[:3,3]-palm[:3,3];length=np.linalg.norm(delta)
        delta=delta*min(1.,.025/max(length,1e-12))
        mount=Rotation.from_euler('XYZ',q[3:6]).as_matrix().T@palm[:3,:3]
        angles=closest_wrist_angles(target[:3,:3],q[3:6],mount)
        angles=q[3:6]+np.clip(angles-q[3:6]-.010*dq[3:6],-.10,.10)
        rotation=Rotation.from_euler('XYZ',angles).as_matrix()@mount
        directions=np.r_[object_pose[:3,:3].T,-object_pose[:3,:3].T,normal[None]]
        local=np.zeros((1,3)) if hand_local is None else np.asarray(hand_local)
        if palm_vertices is not None:local=np.r_[local,palm_vertices]
        # A few millimetres of planned clearance prevents commanded table plunge;
        # actual safety remains the original 100mrad / 20mm physical gate.
        required=height+.003-float(np.min((local@rotation.T+palm[:3,3])@normal))
        alternatives=[delta]
        for direction in directions:
            travel=.025*direction
            # Favor displacement toward the object's current center.
            if np.dot(travel,object_pose[:3,3]-palm[:3,3])<0:travel*=.25
            alternatives.append(travel)
        costs=[];projected=[]
        for i,value in enumerate(alternatives):
            value=value+normal*max(required-float(value@normal),0.)
            norm=np.linalg.norm(value);value*=min(1.,.025/max(norm,1e-12))
            deficit=max(required-float(value@normal),0.)
            cost=np.linalg.norm(value-delta)+10*deficit+(0. if i==0 else .005)
            projected.append(value);costs.append(cost)
        chosen=int(np.argmin(costs));out=np.eye(4);out[:3,:3]=rotation;out[:3,3]=palm[:3,3]+projected[chosen]
        desired=np.clip(np.asarray(fingers)-.010*dq[FINGER_ACTIVE],.06,FINGER_UPPER)
        desired=np.clip(desired,q[FINGER_ACTIVE]-.08,q[FINGER_ACTIVE]+.08)
        desired=np.clip(desired,.06,FINGER_UPPER)
        return out,desired,dict(candidate_count=len(alternatives),selected_candidate=chosen,directions_world=directions.tolist(),support_required_delta_m=required,
            proposal_delta_m=(target[:3,3]-palm[:3,3]).tolist(),filtered_delta_m=projected[chosen].tolist())


class RepairedGraspConsumer(MeasuredGraspConsumer):
    """Same measured phases and R for both arms; shared reach filter and PI."""
    def __init__(self,*args,plane,palm_vertices,**kwargs):
        super().__init__(*args,**kwargs);self.servo=GoalServo();self.reach=ReachabilityFilter();self.plane=plane
        self.palm_vertices=np.asarray(palm_vertices);self.servo_active=False
    def step(self,**args):
        intent=args['intent'];palm=args['palm'];q=np.asarray(args['q']);dq=np.asarray(args['dq']);obj=args['object_pose']
        reach_info=None
        if self.phase in ('approach','close'):
            local=(np.asarray(args.pop('hand'))-palm[:3,3])@palm[:3,:3]
            target,fingers,reach_info=self.reach.project(palm,obj@intent.approach_pose_object,obj,q,dq,intent.preshape_q,
                plane=self.plane,hand_local=local,palm_vertices=self.palm_vertices)
            mount=Rotation.from_euler('XYZ',q[3:6]).as_matrix().T@palm[:3,:3]
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
