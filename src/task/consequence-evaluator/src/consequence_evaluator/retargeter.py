"""Hand-trajectory actions -> current-state-relative Inspire PD targets.

Only twelve controls are independent: wrist xyz/rotation and six finger joints.
The native interface has eighteen coordinates and supplies the finger coupling.
The learned target is a recorded *commanded* PD target, not a future measured q.
"""
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn

SCHEMA = 'ref2dex.hand-trajectory-retargeter.v1'
ACTIVE_FINGERS = np.array([6,8,10,12,14,15])
FINGER_SCALE = np.array([1.6,1.6,1.6,1.6,1.15,.55],dtype=np.float32)
DOF_NAMES = ('joint1','joint2','joint3','joint4','joint5','joint6',
             'index_proximal_joint','index_intermediate_joint',
             'middle_proximal_joint','middle_intermediate_joint',
             'pinky_proximal_joint','pinky_intermediate_joint',
             'ring_proximal_joint','ring_intermediate_joint',
             'thumb_proximal_yaw_joint','thumb_proximal_pitch_joint',
             'thumb_intermediate_joint','thumb_distal_joint')


def wrist_rotation(q):
    """URDF chain is intrinsic XYZ: Rx(q3) Ry(q4) Rz(q5)."""
    value = np.asarray(q)
    return Rotation.from_euler('XYZ',value[...,3:6].reshape(-1,3)).as_matrix().reshape(*value.shape[:-1],3,3)


def commanded_targets(q, control):
    q, control = np.asarray(q), np.asarray(control)
    if q.shape != control.shape or q.shape[-1] != 18:
        raise ValueError('native q/control shape mismatch')
    out = np.zeros_like(q)
    out[...,:3] = q[...,:3]+control[...,:3]
    out[...,3:6] = q[...,3:6]+np.pi*control[...,3:6]
    out[...,ACTIVE_FINGERS] = (1+control[...,ACTIVE_FINGERS])*.5*FINGER_SCALE
    for dst,src,ratio in [(7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)]:
        out[...,dst] = out[...,src]*ratio
    return out.astype('float32')


def relative_targets(anchor, desired):
    """Recorded future commanded pose/fingers relative to query-time q."""
    anchor,desired = np.asarray(anchor),np.asarray(desired)
    rel = np.swapaxes(wrist_rotation(anchor),-1,-2)[...,None,:,:] @ wrist_rotation(desired)
    rv = Rotation.from_matrix(rel.reshape(-1,3,3)).as_rotvec().reshape(*desired.shape[:-1],3)
    return np.concatenate((desired[...,:3]-anchor[...,None,:3],rv,
                           desired[...,ACTIVE_FINGERS]-anchor[...,None,ACTIVE_FINGERS]),-1).astype('float32')


def closest_euler(matrix, reference):
    first = Rotation.from_matrix(np.asarray(matrix).reshape(-1,3,3)).as_euler('XYZ').reshape(np.asarray(reference).shape)
    second = np.stack((first[...,0]+np.pi,np.pi-first[...,1],first[...,2]+np.pi),-1)
    candidates = np.stack((first,second),-2)
    candidates += 2*np.pi*np.round((np.asarray(reference)[...,None,:]-candidates)/(2*np.pi))
    index = np.argmin(np.sum((candidates-np.asarray(reference)[...,None,:])**2,-1),-1)
    return np.take_along_axis(candidates,index[...,None,None],axis=-2)[...,0,:]


def absolute_targets(anchor, relative):
    anchor,relative = np.asarray(anchor),np.asarray(relative)
    out = np.zeros((*relative.shape[:-1],18),dtype='float32')
    out[...,:3] = anchor[...,None,:3]+relative[...,:3]
    rv = Rotation.from_rotvec(relative[...,3:6].reshape(-1,3)).as_matrix().reshape(*relative.shape[:-1],3,3)
    matrix = wrist_rotation(anchor)[...,None,:,:] @ rv
    ref = np.broadcast_to(anchor[...,None,3:6],relative[...,3:6].shape)
    out[...,3:6] = closest_euler(matrix,ref)
    out[...,ACTIVE_FINGERS] = anchor[...,None,ACTIVE_FINGERS]+relative[...,6:]
    for dst,src,ratio in [(7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)]:
        out[...,dst] = out[...,src]*ratio
    return out


def native_control(desired, current):
    """Convert frozen absolute targets using mechanical joint feedback only.

    No trajectory/model requery occurs here. Native wrist actions are increments,
    whereas finger controls are absolute scaled targets. The physical native
    controller supplies the six dependent joints; unused outputs are zero.
    """
    desired,current = np.asarray(desired),np.asarray(current)
    out = np.zeros_like(current,dtype='float32')
    out[...,:3] = desired[...,:3]-current[...,:3]
    target_rot = closest_euler(wrist_rotation(desired),current[...,3:6])
    out[...,3:6] = (target_rot-current[...,3:6])/np.pi
    out[...,ACTIVE_FINGERS] = desired[...,ACTIVE_FINGERS]*2/FINGER_SCALE-1
    return out


def trajectory_input(current_hand, future_hand):
    current_hand,future_hand = np.asarray(current_hand),np.asarray(future_hand)
    return (future_hand-current_hand[...,None,:,:]).astype('float32')


class Standardizer:
    def __init__(self, mean, scale):
        self.mean = np.asarray(mean,dtype='float32')
        self.scale = np.asarray(scale,dtype='float32')
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or (self.scale<=0).any():
            raise ValueError('invalid retargeter statistics')

    @classmethod
    def fit(cls, value, floor=1e-4):
        value = np.asarray(value)
        return cls(value.mean(0,dtype=np.float64),np.maximum(value.std(0,dtype=np.float64),floor))

    def encode(self, value):
        return ((np.asarray(value)-self.mean)/self.scale).astype('float32')

    def decode(self, value):
        return (np.asarray(value)*self.scale+self.mean).astype('float32')

    def as_dict(self):
        return dict(mean=self.mean.tolist(),scale=self.scale.tolist())


class HandTrajectoryRetargeter(nn.Module):
    def __init__(self,width=128):
        super().__init__()
        self.width = width
        self.hand = nn.Sequential(nn.Linear(33,width),nn.SiLU(),nn.Linear(width,width))
        self.state = nn.Sequential(nn.Linear(36,width),nn.SiLU(),nn.Linear(width,width))
        self.time = nn.Parameter(torch.randn(1,24,width)*.02)
        self.query = nn.Parameter(torch.randn(1,24,width)*.02)
        layer = nn.TransformerDecoderLayer(width,4,width*4,dropout=0.,batch_first=True,norm_first=True)
        self.decoder = nn.TransformerDecoder(layer,2)
        self.output = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,12))

    def forward(self, hand, state):
        if hand.shape[1:] != (24,11,3) or state.shape != (len(hand),36):
            raise ValueError('retargeter requires [N,24,11,3] displacements and [N,36] current q/dq')
        context = torch.cat((self.hand(hand.flatten(2))+self.time,self.state(state)[:,None]),1)
        return self.output(self.decoder((self.query+self.time).expand(len(hand),-1,-1),context))
