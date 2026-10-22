"""Canonical representatives for Inspire's three continuous wrist rotations.

Only reset positions change. Translation, finger limits, velocities, runtime
actions and original source-PD targets retain their separate contracts.
"""
import math

RESET_ANGLE_POLICY='inspire.continuous-wrist-reset-principal-radians.v1'


def normalize_wrist_positions(q):
    import torch
    if q.ndim!=2 or q.shape[1]!=18 or not q.is_floating_point() or not torch.isfinite(q).all():
        raise ValueError('finite floating Bx18 Inspire reset positions required')
    result=q.clone()
    result[:,3:6]=torch.remainder(result[:,3:6]+math.pi,2*math.pi)-math.pi
    return result


def require_continuous_wrist_dofs(names,has_limits):
    if len(names)!=18 or list(names[3:6])!=['joint4','joint5','joint6'] or len(has_limits)!=18:
        raise ValueError('Inspire wrist DOF layout differs from the verified URDF')
    if any(bool(value) for value in has_limits[3:6]):
        raise ValueError('wrist normalization requires continuous, unlimited rotation DOFs')


def install_continuous_wrist_reset(task_class):
    if task_class.__dict__.get('_inspire_continuous_wrist_reset',False):return
    original=task_class._set_env_state
    def set_state(task,env_ids,dof_pos,dof_vel):
        # Normalize arguments BEFORE the original state/FK/reset chain consumes
        # them. Only selected environments are written by that indexed chain.
        return original(task,env_ids,normalize_wrist_positions(dof_pos),dof_vel)
    task_class._set_env_state=set_state
    task_class._inspire_continuous_wrist_reset=True
