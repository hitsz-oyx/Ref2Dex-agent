"""Versioned action labels and query-frozen targets for Inspire's controller.

The wrist joint chain is translation XYZ then intrinsic rotation XYZ (URDF
joint1..6, zero origins). Geometry uses link6 as the wrist frame, before the
fixed palm transform. SO(3) labels discard Euler winding: their physical pose
roundtrip is exact, but their generalized PD coordinates need not be unique.
"""
import warnings
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .paths import add_dependencies
add_dependencies()
from trajectory_policy.retargeter_data import ACTIVE
from trajectory_policy.pd_retargeter import command, couple

NATIVE = 'native-command.v1'
PD = 'pd-relative-current.v1'
GEOMETRY = 'wrist-geometry-relative-current.v1'
GEOMETRY_LIFTED = 'wrist-geometry-continuous-lift.v1'
REPRESENTATIONS = (NATIVE, PD, GEOMETRY, GEOMETRY_LIFTED)
R_ONLY_SCHEMA = 'ref2dex.rl660-physical-R.v1'


def validate_representation(value):
    if value not in REPRESENTATIONS:
        raise ValueError('unknown action representation: '+str(value))
    return value


def action_dimension(representation):
    validate_representation(representation)
    return 15 if representation==GEOMETRY_LIFTED else 12


def _rotation(angles):
    return Rotation.from_euler('XYZ', np.asarray(angles).reshape(-1,3))


def _nearest_euler(rotation, live):
    """Choose a nearby equivalent Euler branch; never silently wrap a servo."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)  # exact gimbal lock uses SciPy's canonical branch
        first=rotation.as_euler('XYZ').reshape(live.shape)
    second=np.stack((first[...,0]+np.pi,np.pi-first[...,1],first[...,2]+np.pi),-1)
    choices=np.stack((first,second),-2)
    choices+=2*np.pi*np.round((live[...,None,:]-choices)/(2*np.pi))
    pick=np.argmin(np.square(choices-live[...,None,:]).sum(-1),axis=-1)
    result=np.take_along_axis(choices,pick[...,None,None],axis=-2)[...,0,:]
    # At gimbal lock the x/z pair is a continuous family, not just two Euler
    # branches. Project the reference onto x + sign(y)*z = canonical x.
    singular=np.abs(np.cos(first[...,1]))<1e-7
    sign=np.where(np.sin(first[...,1])>=0,1.,-1.)
    phase=(first[...,0]-live[...,0]-sign*live[...,2]+np.pi)%(2*np.pi)-np.pi
    result[...,0]=np.where(singular,live[...,0]+phase/2,result[...,0])
    result[...,2]=np.where(singular,live[...,2]+sign*phase/2,result[...,2])
    return result


def lifted_euler(rotation, reference):
    """Restore the supervised winding; preserve exact labels at singularities."""
    result=_nearest_euler(rotation,reference)
    mismatch=(_rotation(reference).inv()*rotation).magnitude().reshape(reference.shape[:-1])
    return np.where((mismatch<1e-6)[...,None],reference,result)


def encode_targets(pd_targets, query_q, representation, native=None):
    """All eight targets use ONE query state; no future q enters this map."""
    validate_representation(representation)
    target=np.asarray(pd_targets);base=np.asarray(query_q)
    if target.shape != (*base.shape[:-1],8,18) or base.shape[-1]!=18:
        raise ValueError('eight actual PD targets and one query q18 required')
    if representation==NATIVE:
        if native is None or np.shape(native)!=(*base.shape[:-1],8,12):
            raise ValueError('recorded native labels required')
        return np.asarray(native,dtype=np.float32).copy()
    value=target[...,list(ACTIVE)]-base[...,None,list(ACTIVE)]
    if representation in (GEOMETRY,GEOMETRY_LIFTED):
        lift=value[...,3:6].copy()
        current=np.broadcast_to(base[...,None,3:6],target[...,3:6].shape)
        relative=_rotation(current).inv()*_rotation(target[...,3:6])
        value[...,3:6]=relative.as_rotvec().reshape(value[...,3:6].shape)
        if representation==GEOMETRY_LIFTED:value=np.concatenate((value,lift),axis=-1)
    return value.astype(np.float32)


def freeze_chunk(encoded, query_q, representation):
    """Convert once at query time; a buffer stores absolute targets, not deltas.

    Geometry buffers store an absolute wrist rotation vector. Euler inversion
    is delayed until consumption so it can choose a branch near live joints.
    """
    validate_representation(representation)
    dimension=action_dimension(representation)
    if encoded.shape != (*query_q.shape[:-1],8,dimension):
        raise ValueError('query q and representation-specific action chunk required')
    if representation==NATIVE:return encoded
    result=encoded[...,:12]+query_q[...,None,list(ACTIVE)]
    if representation in (GEOMETRY,GEOMETRY_LIFTED):
        base=np.broadcast_to(query_q[...,None,3:6].detach().cpu().numpy(),encoded[...,3:6].shape)
        relative=Rotation.from_rotvec(encoded[...,3:6].detach().cpu().numpy().reshape(-1,3))
        world=_rotation(base)*relative
        result=result.clone()
        result[...,3:6]=torch.as_tensor(world.as_rotvec().reshape(result[...,3:6].shape),device=encoded.device,dtype=encoded.dtype)
        if representation==GEOMETRY_LIFTED:
            result=torch.cat((result,encoded[...,12:15]+query_q[...,None,3:6]),dim=-1)
    return result


def consume_target(target, live_q, representation, controller):
    """Invert the controller using LIVE q; never re-anchor a frozen target."""
    validate_representation(representation)
    if target.shape!=(*live_q.shape[:-1],action_dimension(representation)):raise ValueError('target dimensions and live q18 differ')
    if representation==NATIVE:
        intended=live_q.new_zeros(live_q.shape);intended[...,list(ACTIVE)]=target
        applied=intended.clamp(-1,1)
        return intended,applied,(intended-applied).abs().amax(-1)>1e-6
    desired=live_q.new_zeros(live_q.shape);desired[...,list(ACTIVE)]=target[...,:12]
    if representation in (GEOMETRY,GEOMETRY_LIFTED):
        rotation=Rotation.from_rotvec(target[...,3:6].detach().cpu().numpy().reshape(-1,3))
        if representation==GEOMETRY_LIFTED:
            angles=lifted_euler(rotation,target[...,12:15].detach().cpu().numpy())
        else:angles=_nearest_euler(rotation,live_q[...,3:6].detach().cpu().numpy())
        desired[...,3:6]=torch.as_tensor(angles,device=live_q.device,dtype=live_q.dtype)
    offset=live_q.new_tensor(controller['offset']);scale=live_q.new_tensor(controller['scale'])
    return command(couple(desired),live_q,offset,scale)


def target_tensor(batch, representation):
    validate_representation(representation)
    if representation==NATIVE:return batch['native_action']
    if 'action_target' not in batch:raise ValueError('physical action target labels required')
    return batch['action_target']
