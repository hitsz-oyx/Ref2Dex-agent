"""Embodiment-specific measured state; shared task fields live elsewhere."""
from dataclasses import dataclass
import numpy as np
from .inspire_contract import NATIVE_TO_WAM,expand_action
from .inspire_shadow import current_state


@dataclass(frozen=True)
class RobotState:
    values:np.ndarray
    fields:dict
    units:dict


class InspireAdapter:
    native_hz=30
    action_dim=12
    robot_id='dexplore.inspire.measured.v1'

    @staticmethod
    def legacy_tau_column_permutation():
        """New canonical tau columns select these old native-order columns.

        Apply to both the first-layer weights and input normalization when
        migrating a legacy Inspire scheduled actor; a reshape is insufficient.
        """
        return np.arange(264).reshape(8,11,3)[:,NATIVE_TO_WAM].reshape(-1)

    @staticmethod
    def canonical_hand(native_hand):return np.asarray(native_hand)[...,NATIVE_TO_WAM,:].copy()

    @staticmethod
    def native_action(action):return expand_action(action)

    @staticmethod
    def state(snapshot,endpoint,previous,*,shape_feature):
        base=current_state(snapshot['q'],snapshot['dq'],snapshot['physical'],snapshot['objects'],endpoint,previous)
        leading=base.shape[:-1]
        shape=np.asarray(shape_feature,np.float32)
        if shape.shape!=leading+(64,):raise ValueError('a bound 64D object shape feature is required')
        dof_valid=np.broadcast_to(np.asarray(snapshot.get('dof_force_valid',False),bool),leading)
        contact_valid=np.broadcast_to(np.asarray(snapshot.get('contact_force_valid',False),bool),leading)
        dof=np.asarray(snapshot.get('dof_force',np.zeros(leading+(18,))),np.float32)
        contact=np.asarray(snapshot.get('contact_forces',np.zeros(leading+(5,3))),np.float32)
        if dof.shape!=leading+(18,) or contact.shape!=leading+(5,3):raise ValueError('measured force dimensions mismatch')
        if (dof_valid.any() and 'dof_force' not in snapshot) or (contact_valid.any() and 'contact_forces' not in snapshot):
            raise ValueError('missing force cannot be marked measured')
        dof=np.where(dof_valid[...,None],dof,0);contact=np.where(contact_valid[...,None,None],contact,0)
        values=np.concatenate((base,dof,contact.reshape(leading+(15,)),shape,
            dof_valid[...,None].astype(np.float32),contact_valid[...,None].astype(np.float32)),-1)
        if not np.isfinite(values).all():raise ValueError('finite robot state required')
        return RobotState(values,dict(legacy_measured312=312,dof_force=18,contact_force=15,object_shape=64,force_valid=2),
            dict(dof_force='native generalized force: N for prismatic / Nm for rotary DOFs',contact_force='N in world frame',
                force_valid='measured in this episode; reset observations are invalid',object_shape='bound learned shape encoder'))


class ShadowAdapter:
    native_hz=60
    action_dim=24
    robot_id='unidex.shadow.legacy300.v1'

    @staticmethod
    def state(current300):
        value=np.asarray(current300,np.float32)
        if value.shape[-1:]!=(300,) or not np.isfinite(value).all():raise ValueError('native current300 required')
        return RobotState(np.clip(value,-5,5),dict(legacy_current300=300),dict(legacy_current300='original clipped UniDex state and goal fields'))

    @staticmethod
    def canonical_hand(points):
        points=np.asarray(points)
        if points.shape[-2:]!=(11,3):raise ValueError('canonical palm/bases/tips hand11 required')
        return points.copy()

    @staticmethod
    def native_action(action):
        action=np.asarray(action,np.float32)
        if action.shape[-1:]!=(24,) or not np.isfinite(action).all():raise ValueError('finite native Shadow action24 required')
        return np.clip(action,-1,1).copy()
