"""Exact XYZ projection in the fixed decoder's temporal basis, no policy model."""
import numpy as np

from .decoder import HORIZON, KNOTS


def temporal_maps(dt):
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError('finite positive control interval required')
    times = np.arange(1, HORIZON + 1)
    identity = np.eye(len(KNOTS))
    position = np.stack([np.interp(times, KNOTS, identity[:, col]) for col in range(len(KNOTS))], -1)
    delta = np.diff(position, axis=0)
    velocity = np.concatenate((delta[:1], (delta[:-1]+delta[1:])/2, delta[-1:]))/dt
    return position, velocity, position + .1*velocity


def future_xyz_velocity(xyz, dt):
    delta = np.diff(xyz, axis=-2)
    return np.concatenate((delta[..., :1, :], (delta[..., :-1, :]+delta[..., 1:, :])/2,
                           delta[..., -1:, :]), axis=-2)/dt


def project(matrix, targets):
    """Each sample has its own unconstrained knot optimum; return normal checks."""
    targets = np.asarray(targets, dtype=np.float64)
    if matrix.ndim != 2 or targets.ndim != 3 or targets.shape[1] != len(matrix):
        raise ValueError('matrix rows must match batch target frames')
    if not np.isfinite(matrix).all() or not np.isfinite(targets).all():
        raise ValueError('finite projection inputs required')
    flat = targets.transpose(1, 0, 2).reshape(len(matrix), -1)
    solution = np.linalg.lstsq(matrix, flat, rcond=None)[0]
    nodes = solution.reshape(matrix.shape[1], len(targets), targets.shape[-1]).transpose(1, 0, 2)
    predicted = np.einsum('tk,nkd->ntd', matrix, nodes)
    residual = predicted-targets
    normal_error = float(np.max(np.abs(matrix.T @ (matrix @ solution-flat))))
    return nodes, residual, normal_error


def fit_prefix(xyz, dt=1/30):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[1:] != (24, 3):
        raise ValueError('batch of24future XYZ targets required')
    position, velocity, feedforward = temporal_maps(dt)
    target_ff = xyz + .1*future_xyz_velocity(xyz, dt)
    pos_nodes, pos_error, pos_normal = project(position[:8], xyz[:, :8])
    ff_nodes, ff_error, ff_normal = project(feedforward[:8], target_ff[:, :8])
    joint = np.concatenate((position[:8], feedforward[:8], .25*position[8:]))
    joint_target = np.concatenate((xyz[:, :8], target_ff[:, :8], .25*xyz[:, 8:]), axis=1)
    nodes, _, joint_normal = project(joint, joint_target)
    fitted_xyz = np.einsum('tk,nkd->ntd', position, nodes)
    fitted_ff = np.einsum('tk,nkd->ntd', feedforward, nodes)
    return dict(nodes=nodes, fitted_xyz=fitted_xyz, fitted_ff=fitted_ff, target_ff=target_ff,
        position_bound_error=pos_error, ff_bound_error=ff_error,
        position_bound_nodes=pos_nodes, ff_bound_nodes=ff_nodes,
        normal_errors=dict(position=pos_normal, feedforward=ff_normal, joint=joint_normal),
        position=position, velocity=velocity, feedforward=feedforward)
