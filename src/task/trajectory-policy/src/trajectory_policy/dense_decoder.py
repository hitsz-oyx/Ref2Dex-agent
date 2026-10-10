"""Direct 24-frame physical trajectory action, without a learned projection.

The 288 coordinates are independent of a reference actor. Geometry-derived
future q is used by encode only for offline labels/interface audits; decode
accepts an actor action and the measured current state.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .decoder import TrajectoryDecoder
from .metric_decoder import coordinates, COORD_SCALE
from consequence_evaluator.retargeter import wrist_rotation, closest_euler
from consequence_evaluator.reference_tracking import apply_coupling, future_reference_velocity
from consequence_evaluator.tau_tracking import FINGERS, FINGER_LIMITS

SCHEMA = 'ref2dex.dense-trajectory-decoder.v1'
LATENT_DIM = 288


class DenseDecoder(TrajectoryDecoder):
    @staticmethod
    def encode(future_q, current_q, object_pose):
        return coordinates(future_q, current_q, object_pose).astype(np.float32)

    def decode(self, c, current_q, object_pose, dt):
        c, current, obj = map(lambda value: np.asarray(value, np.float32),
                              (c, current_q, object_pose))
        if c.shape != (len(current), LATENT_DIM) or current.shape != (len(current), 18) or obj.shape != (len(current), 4, 4):
            raise ValueError('dense action/current shapes mismatch')
        if not all(np.isfinite(value).all() for value in (c, current, obj)) or not np.isfinite(dt) or dt <= 0:
            raise ValueError('finite action/state and positive dt required')
        value = c.reshape(-1, 24, 12)*COORD_SCALE
        rotation = obj[:, :3, :3]
        xyz = current[:, None, :3]+np.einsum('nij,nkj->nki', rotation, value[:, :, :3].clip(-1, 1))
        relative = Rotation.from_rotvec(value[:, :, 3:6].reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
        matrices = rotation[:, None] @ relative @ rotation[:, None].transpose(0, 1, 3, 2) @ wrist_rotation(current)[:, None]
        q = np.zeros((len(c), 25, 18), np.float32)
        q[:, 1:, :3] = xyz
        q[:, 1:, list(FINGERS)] = np.maximum(np.minimum(value[:, :, 6:], np.asarray(FINGER_LIMITS)), 0)
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
        velocity = torch.stack([future_reference_velocity(value, dt) for value in tensor])
        if not all(torch.isfinite(value).all() for value in (tensor, hand, velocity)):
            raise FloatingPointError('nonfinite dense trajectory')
        return dict(c=c, q=tensor, hand=hand, velocity=velocity)
