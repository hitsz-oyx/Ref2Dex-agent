"""Standalone48D intent -> bounded rigid hand trajectory, no base policy."""
import numpy as np
import torch
from scipy.special import expit
from scipy.spatial.transform import Rotation, Slerp
from consequence_evaluator.contracts import HAND_LINKS
from consequence_evaluator.reference_motion import NATIVE_DOF_NAMES
from consequence_evaluator.retargeter import wrist_rotation, closest_euler
from consequence_evaluator.reset_kinematics import ResetKinematics
from consequence_evaluator.reference_tracking import future_reference_velocity
from consequence_evaluator.tau_tracking import FINGERS, FINGER_LIMITS

SCHEMA='ref2dex.trajectory-decoder-48.v1'
KNOTS=(1,8,16,24)
HORIZON=24
LATENT_DIM=48
TRANSLATION_BOUND_M=1.


class TrajectoryDecoder:
    def __init__(self,urdf,device):
        self.device=torch.device(device)
        self.fk=ResetKinematics(urdf,NATIVE_DOF_NAMES,HAND_LINKS,self.device)

    @staticmethod
    def encode(future_q,current_q,object_pose):
        """Diagnostic/training labels only: geometry-derived q, never measured future q.

        Deployment uses decode(c,current_q,object_pose) without a future argument.
        """
        future=np.asarray(future_q,dtype=np.float32)
        current=np.asarray(current_q,dtype=np.float32)
        obj=np.asarray(object_pose,dtype=np.float32)
        if future.shape!=(len(current),24,18) or current.shape!=(len(current),18) or obj.shape!=(len(current),4,4):
            raise ValueError('exact24frame geometry label/current shapes required')
        nodes=future[:,np.asarray(KNOTS)-1]
        rotation=obj[:,:3,:3]
        delta=np.einsum('nij,nkj->nki',rotation.transpose(0,2,1),nodes[:,:,:3]-current[:,None,:3])
        if np.max(np.abs(delta))>=TRANSLATION_BOUND_M:
            raise ValueError('label lies outside declared translation action space')
        translation=np.arctanh(delta/TRANSLATION_BOUND_M)
        relative=rotation[:,None].transpose(0,1,3,2)@wrist_rotation(nodes)@wrist_rotation(current)[:,None].transpose(0,1,3,2)@rotation[:,None]
        vector=Rotation.from_matrix(relative.reshape(-1,3,3)).as_rotvec().reshape(len(current),4,3)
        angle=np.linalg.norm(vector,axis=-1,keepdims=True)
        angular=vector*(np.arctanh(np.minimum(angle/np.pi,1-1e-6))/np.maximum(angle,1e-12))
        fraction=nodes[:,:,list(FINGERS)]/np.asarray(FINGER_LIMITS)
        if np.min(fraction)<-1e-5 or np.max(fraction)>1+1e-5:
            raise ValueError('label lies outside native finger bounds')
        fraction=np.clip(fraction,1e-6,1-1e-6)
        finger=np.log(fraction/(1-fraction))
        c=np.concatenate((translation,angular,finger),-1).reshape(len(current),48).astype(np.float32)
        if not np.isfinite(c).all():raise ValueError('nonfinite encoded trajectory')
        return c

    def decode(self,c,current_q,object_pose,dt):
        """Only actor intent and live state; rigid tau and its derived geometry q."""
        c=np.asarray(c,dtype=np.float32);current=np.asarray(current_q,dtype=np.float32)
        obj=np.asarray(object_pose,dtype=np.float32)
        if c.shape!=(len(current),48) or current.shape!=(len(current),18) or obj.shape!=(len(current),4,4):
            raise ValueError('latent/current shapes mismatch')
        if not all(np.isfinite(v).all() for v in (c,current,obj)):raise ValueError('finite action/state required')
        nodes=c.reshape(-1,4,12);rotation=obj[:,:3,:3]
        translation=current[:,None,:3]+np.einsum('nij,nkj->nki',rotation,np.tanh(nodes[:,:,:3])*TRANSLATION_BOUND_M)
        radius=np.linalg.norm(nodes[:,:,3:6],axis=-1,keepdims=True)
        vector=nodes[:,:,3:6]*(np.pi*np.tanh(radius)/np.maximum(radius,1e-12))
        rel=Rotation.from_rotvec(vector.reshape(-1,3)).as_matrix().reshape(len(c),4,3,3)
        orientations=rotation[:,None]@rel@rotation[:,None].transpose(0,1,3,2)@wrist_rotation(current)[:,None]
        fingers=expit(nodes[:,:,6:])*np.asarray(FINGER_LIMITS,dtype=np.float32)
        time=np.arange(25);knot_time=np.asarray((0,)+KNOTS)
        q=np.zeros((len(c),25,18),dtype=np.float32)
        for i in range(len(c)):
            xyz=np.concatenate((current[i:i+1,:3],translation[i]))
            finger=np.concatenate((current[i:i+1,list(FINGERS)],fingers[i]))
            for d in range(3):q[i,:,d]=np.interp(time,knot_time,xyz[:,d])
            for j,d in enumerate(FINGERS):q[i,:,d]=np.interp(time,knot_time,finger[:,j])
            mats=np.concatenate((wrist_rotation(current[i:i+1]),orientations[i]))
            interpolated=Slerp(knot_time,Rotation.from_matrix(mats))(time).as_matrix()
            prior=current[i,3:6]
            for j,matrix in enumerate(interpolated):
                q[i,j,3:6]=closest_euler(matrix,prior);prior=q[i,j,3:6]
        for distal,parent,ratio in ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)):
            q[:,:,distal]=q[:,:,parent]*ratio
        q[:,0]=current
        value=torch.as_tensor(q,device=self.device)
        root=torch.zeros(len(c)*25,13,device=self.device);root[:,6]=1
        points=self.fk.positions(value.reshape(-1,18),root).reshape(len(c),25,11,3)
        velocity=torch.stack([future_reference_velocity(v,dt) for v in value])
        if not torch.isfinite(points).all():raise FloatingPointError('nonfinite decoded points')
        return dict(q=value,hand=points[:,1:],velocity=velocity,c=c)
