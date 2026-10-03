"""Matched input slots for a bounded query/neighbor granularity probe."""
import numpy as np
from torch import nn
from src.task.CmResidual.surface_motion_prior import SurfaceMotionPrior

VARIANTS = ((64, 'mean'), (256, 'mean'), (64, 'detail'), (256, 'detail'))


def encode_granularity(raw, aggregation):
    assert aggregation in ('mean', 'detail')
    r = raw['pose'][:, :3, :3]
    origin = raw['pose'][:, None, :3, 3]
    def vector(value):
        return np.einsum('n...i,nij->n...j', value, r)
    obj = vector(raw['obj']-origin)
    normal = vector(raw['normal'])
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
    relative = vector(raw['hand']-raw['obj'][:, :, None])
    normals = vector(raw['hand_normal'])
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
    flow = vector(raw['next_hand']-raw['hand'])
    distance = np.linalg.norm(relative, axis=-1).min(-1)[..., None]
    def neighborhood(value):
        if aggregation == 'mean':
            value = np.broadcast_to(value.mean(2, keepdims=True), value.shape)
        return value.reshape(value.shape[0], value.shape[1], 12)
    features = np.concatenate((obj/.05, normal, neighborhood(relative)/.05,
                               neighborhood(normals), neighborhood(flow)/.01,
                               distance/.05,
                               np.broadcast_to(vector(raw['global_hand_flow'])[:, None], obj.shape)/.01,
                               vector(raw['obj']-raw['previous_obj'])/.01), -1)
    target = vector(raw['next_obj']-raw['obj'])/.01
    assert features.shape[-1] == 49
    assert np.isfinite(features).all() and np.isfinite(target).all()
    return features.astype(np.float32), target.astype(np.float32)


class GranularityPrior(SurfaceMotionPrior):
    def __init__(self):
        super().__init__()
        self.encoder[0] = nn.Linear(49, 128)


def granularity_gates(reports):
    gains = {}
    for hand in ('mano', 'inspire'):
        base = reports[hand+'_64_mean'][hand+'_eval']['parent_epe_mm']
        for variant in ('256_mean', '64_detail', '256_detail'):
            gains[hand+'_'+variant] = 1-reports[hand+'_'+variant][hand+'_eval']['parent_epe_mm']/base
    gates = {name: value >= .10 for name, value in gains.items()}
    label = 'PROMISING' if any(gates.values()) else ('UNCLEAR' if max(gains.values()) >= .05 else 'UNPROMISING')
    return gains, gates, label
