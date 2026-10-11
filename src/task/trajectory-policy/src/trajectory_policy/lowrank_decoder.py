"""Fixed learned temporal D48, with independent actor intent and live anchoring."""
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from scipy.spatial.transform import Rotation

from .decoder import TrajectoryDecoder, TRANSLATION_BOUND_M
from consequence_evaluator.retargeter import wrist_rotation, closest_euler
from consequence_evaluator.reference_tracking import future_reference_velocity, apply_coupling
from consequence_evaluator.tau_tracking import FINGERS, FINGER_LIMITS

SCHEMA = 'ref2dex.lowrank-trajectory-decoder.v1'
COORD_SCALE = np.asarray([.1]*3+[.5]*3+[4.]*6, dtype=np.float32)


def coordinates(future_q, current_q, object_pose):
    future, current, obj = map(lambda x: np.asarray(x, np.float32), (future_q, current_q, object_pose))
    if future.shape != (len(current), 24, 18) or obj.shape != (len(current), 4, 4):
        raise ValueError('24 hand-derived future frames and current calibration required')
    rotation = obj[:, :3, :3]
    xyz = np.einsum('nij,nkj->nki', rotation.transpose(0, 2, 1), future[:, :, :3]-current[:, None, :3])
    if abs(xyz).max() >= TRANSLATION_BOUND_M:
        raise ValueError('trajectory outside translation coordinate bound')
    relative = rotation[:, None].transpose(0, 1, 3, 2) @ wrist_rotation(future) @ wrist_rotation(current)[:, None].transpose(0, 1, 3, 2) @ rotation[:, None]
    vector = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(len(current), 24, 3)
    angle = np.linalg.norm(vector, axis=-1, keepdims=True)
    angular = vector * (np.arctanh(np.minimum(angle/np.pi, 1-1e-6))/np.maximum(angle, 1e-12))
    fraction = future[:, :, list(FINGERS)]/np.asarray(FINGER_LIMITS)
    if fraction.min() < -1e-5 or fraction.max() > 1+1e-5:
        raise ValueError('future geometric fingers outside native bounds')
    fraction = np.clip(fraction, 1e-6, 1-1e-6)
    value = np.concatenate((np.arctanh(xyz/TRANSLATION_BOUND_M), angular, np.log(fraction/(1-fraction))), -1)
    if not np.isfinite(value).all():
        raise ValueError('finite trajectory coordinates required')
    return (value/COORD_SCALE).reshape(len(current), 288).astype(np.float32)


class LowrankDecoder(TrajectoryDecoder):
    def __init__(self, urdf, device, basis):
        super().__init__(urdf, device)
        with np.load(Path(basis), allow_pickle=False) as stream:
            if str(stream['schema']) != SCHEMA:
                raise ValueError('lowrank basis schema mismatch')
            self.mean, self.components, self.scale = (stream[key].copy() for key in ('mean', 'components', 'latent_scale'))
        if self.mean.shape != (288,) or self.components.shape != (48, 288) or self.scale.shape != (48,) or (self.scale <= 0).any():
            raise ValueError('fixed48D basis contract mismatch')
        if not all(np.isfinite(value).all() for value in (self.mean, self.components, self.scale)):
            raise ValueError('finite decoder basis required')

    def encode(self, future_q, current_q, object_pose):
        return ((coordinates(future_q, current_q, object_pose)-self.mean) @ self.components.T/self.scale).astype(np.float32)

    def decode(self, c, current_q, object_pose, dt):
        c, current, obj = map(lambda x: np.asarray(x, np.float32), (c, current_q, object_pose))
        if c.shape != (len(current), 48) or current.shape != (len(current), 18) or obj.shape != (len(current), 4, 4):
            raise ValueError('latent/current shape mismatch')
        if not all(np.isfinite(value).all() for value in (c, current, obj)):
            raise ValueError('finite action/state required')
        value = (self.mean+(c*self.scale)@self.components).reshape(-1, 24, 12)*COORD_SCALE
        rotation = obj[:, :3, :3]
        xyz = current[:, None, :3]+np.einsum('nij,nkj->nki', rotation, np.tanh(value[:, :, :3])*TRANSLATION_BOUND_M)
        angular = value[:, :, 3:6]
        radius = np.linalg.norm(angular, axis=-1, keepdims=True)
        vector = angular*(np.pi*np.tanh(radius)/np.maximum(radius, 1e-12))
        relative = Rotation.from_rotvec(vector.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
        matrices = rotation[:, None] @ relative @ rotation[:, None].transpose(0, 1, 3, 2) @ wrist_rotation(current)[:, None]
        q = np.zeros((len(c), 25, 18), np.float32)
        q[:, 1:, :3] = xyz
        q[:, 1:, list(FINGERS)] = expit(value[:, :, 6:])*np.asarray(FINGER_LIMITS)
        for row in range(len(c)):
            prior = current[row, 3:6]
            for tick, matrix in enumerate(matrices[row]):
                q[row, tick+1, 3:6] = closest_euler(matrix, prior)
                prior = q[row, tick+1, 3:6]
        tensor = apply_coupling(torch.as_tensor(q, device=self.device))
        tensor[:, 0] = torch.as_tensor(current, device=self.device)
        root = torch.zeros(len(c)*24, 13, device=self.device)
        root[:, 6] = 1
        hand = self.fk.positions(tensor[:, 1:].reshape(-1, 18), root).reshape(-1, 24, 11, 3)
        velocity = torch.stack([future_reference_velocity(v, dt) for v in tensor])
        if not torch.isfinite(hand).all():
            raise FloatingPointError('nonfinite decoded trajectory')
        return dict(c=c, q=tensor, hand=hand, velocity=velocity)
