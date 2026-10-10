"""Fixed48D time basis learned in pose/nominal-feedforward units."""
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .decoder import TrajectoryDecoder
from consequence_evaluator.retargeter import wrist_rotation, closest_euler
from consequence_evaluator.reference_tracking import apply_coupling, future_reference_velocity
from consequence_evaluator.tau_tracking import FINGERS, FINGER_LIMITS

SCHEMA = 'ref2dex.metric-trajectory-decoder.v1'
COORD_SCALE = np.asarray([.01]*3+[.1]*9, np.float32)


def coordinates(future_q, current_q, object_pose):
    future, current, obj = map(lambda x: np.asarray(x, np.float32), (future_q, current_q, object_pose))
    if future.shape != (len(current), 24, 18) or current.shape != (len(current), 18) or obj.shape != (len(current), 4, 4):
        raise ValueError('24 geometric future frames and current calibration required')
    if not all(np.isfinite(value).all() for value in (future, current, obj)):
        raise ValueError('finite geometric labels/current state required')
    rotation = obj[:, :3, :3]
    xyz = np.einsum('nij,nkj->nki', rotation.transpose(0, 2, 1), future[:, :, :3]-current[:, None, :3])
    relative = rotation[:, None].transpose(0, 1, 3, 2) @ wrist_rotation(future) @ wrist_rotation(current)[:, None].transpose(0, 1, 3, 2) @ rotation[:, None]
    angular = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 24, 3)
    fingers = future[:, :, list(FINGERS)]
    if abs(xyz).max() > 1 or (fingers < -1e-5).any() or (fingers > np.asarray(FINGER_LIMITS)+1e-5).any():
        raise ValueError('geometric labels outside declared position/finger bounds')
    return (np.concatenate((xyz, angular, fingers), -1)/COORD_SCALE).reshape(-1, 288)


def fit_basis(values, device):
    """Weighted PCA in a fixed SPD metric; inverse maps restore physical units."""
    values = np.asarray(values, np.float64).reshape(-1, 24, 12)
    identity = np.eye(24)
    delta = np.diff(identity, axis=0)
    derivative = np.concatenate((delta[:1], (delta[:-1]+delta[1:])/2, delta[-1:]))*30
    ff = identity + .1*derivative
    weights = np.asarray([1.]*8+[.25]*16)
    pose_metric = np.diag(weights**2)
    factors = [np.linalg.cholesky(pose_metric+ff[:8].T@ff[:8]), np.linalg.cholesky(pose_metric)]
    transformed = np.empty_like(values)
    for columns, factor in ((slice(0, 6), factors[0]), (slice(6, 12), factors[1])):
        transformed[:, :, columns] = np.einsum('ntc,ts->nsc', values[:, :, columns], factor)
    tensor = torch.as_tensor(transformed.reshape(-1, 288), device=device, dtype=torch.float64)
    mean_y = tensor.mean(0)
    _, singular, vh = torch.linalg.svd(tensor-mean_y, full_matrices=False)
    components_y = vh[:48].cpu().numpy().reshape(48, 24, 12)
    mean = values.mean(0).reshape(288)
    components, encoder = np.empty_like(components_y), np.empty_like(components_y)
    for columns, factor in ((slice(0, 6), factors[0]), (slice(6, 12), factors[1])):
        components[:, :, columns] = np.einsum('ntc,ts->nsc', components_y[:, :, columns], np.linalg.inv(factor))
        encoder[:, :, columns] = np.einsum('ts,nsc->ntc', factor, components_y[:, :, columns])
    components, encoder = components.reshape(48, 288), encoder.reshape(48, 288).T
    sign = np.sign(components[np.arange(48), abs(components).argmax(1)])
    components *= sign[:, None]
    encoder *= sign[None]
    dual_error = float(abs(components@encoder-np.eye(48)).max())
    if dual_error > 1e-10:
        raise ValueError('metric encoder/decoder dual basis mismatch')
    return dict(schema=np.asarray(SCHEMA), mean=mean.astype(np.float32),
        components=components.astype(np.float32), encoder_components=encoder.astype(np.float32),
        latent_scale=np.maximum(singular[:48].cpu().numpy()/np.sqrt(len(values)-1), 1e-4).astype(np.float32),
        dual_error=np.asarray(dual_error))


class MetricDecoder(TrajectoryDecoder):
    def __init__(self, urdf, device, basis):
        super().__init__(urdf, device)
        with np.load(Path(basis), allow_pickle=False) as stream:
            if str(stream['schema']) != SCHEMA:
                raise ValueError('physical metric basis schema mismatch')
            self.mean, self.components, self.encoder, self.scale = (stream[key].copy() for key in
                ('mean', 'components', 'encoder_components', 'latent_scale'))
        if self.mean.shape != (288,) or self.components.shape != (48, 288) or self.encoder.shape != (288, 48) or self.scale.shape != (48,) or (self.scale <= 0).any():
            raise ValueError('fixed48D physical basis shapes mismatch')
        if not all(np.isfinite(value).all() for value in (self.mean, self.components, self.encoder, self.scale)):
            raise ValueError('finite basis required')

    def encode(self, future_q, current_q, object_pose):
        return ((coordinates(future_q, current_q, object_pose)-self.mean)@self.encoder/self.scale).astype(np.float32)

    def decode(self, c, current_q, object_pose, dt):
        c, current, obj = map(lambda x: np.asarray(x, np.float32), (c, current_q, object_pose))
        if c.shape != (len(current), 48) or current.shape != (len(current), 18) or obj.shape != (len(current), 4, 4):
            raise ValueError('latent/current shapes mismatch')
        if not all(np.isfinite(value).all() for value in (c, current, obj)):
            raise ValueError('finite action/current state required')
        value = (self.mean+(c*self.scale)@self.components).reshape(-1, 24, 12)*COORD_SCALE
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
        velocity = torch.stack([future_reference_velocity(v, dt) for v in tensor])
        if not torch.isfinite(hand).all():
            raise FloatingPointError('nonfinite decoded trajectory')
        return dict(c=c, q=tensor, hand=hand, velocity=velocity)
