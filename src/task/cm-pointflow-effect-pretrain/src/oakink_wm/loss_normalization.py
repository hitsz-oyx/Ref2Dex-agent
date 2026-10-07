"""Train-only physical loss scales, independent of warm-start forward buffers."""
import numpy as np
import torch
import torch.nn.functional as F

from .model import geodesic, rigid_points


def validate_loss_statistics(stats, dataset_hash, source_names):
    if (stats.get('schema') != 'pointworld-wm24.loss-normalization.v1'
            or stats.get('split') != 'train'
            or stats.get('input_manifest_sha256') != dataset_hash
            or stats.get('sources') != list(source_names)
            or tuple(source_names) != ('oakink2', 'grab', 'arctic')
            or stats.get('role') != 'loss_only'):
        raise ValueError('main-dynamics train-only loss statistics identity mismatch')
    scales = {}
    for name, shape in (('flow_std', (24, 3)), ('translation_std', (24, 3)),
                        ('rotation_scale', (24,))):
        value = np.asarray(stats.get(name), dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all() or (value <= 0).any():
            raise ValueError('invalid physical loss scale: '+name)
        scales[name] = torch.from_numpy(value)
    return scales


def physical_loss(model, pred, batch, scales):
    """Use physical residuals with corpus scales; leave model buffers untouched.

    Both sides of the original Huber objective subtract the same mean, so that
    mean cancels. Changing only the residual denominator changes the objective
    without changing the interpretation of pretrained inputs or predictions.
    """
    gt = batch['effect'].float()
    actual = rigid_points(gt[..., :3, :3], gt[..., :3, 3], batch['points'])
    predicted = rigid_points(pred['rotation'], pred['translation'], batch['points'])
    weights = model.motion_weights(batch, actual)
    floor = 1 if model.motion_weighting == 'released_incremental' else 1e-6
    residual = (predicted-actual)/scales['flow_std'][None, None, :, None]
    point = (F.huber_loss(residual, torch.zeros_like(residual), reduction='none').mean(-1)*weights).sum()/weights.sum().clamp_min(floor)
    object_weights = weights.mean(-1)
    norm = object_weights.sum().clamp_min(1e-6)
    residual = (pred['translation']-gt[..., :3, 3])/scales['translation_std']
    translation = F.huber_loss(residual, torch.zeros_like(residual), reduction='none').mean(-1)
    angle = geodesic(pred['rotation'], gt[..., :3, :3])/scales['rotation_scale']
    rotation = F.huber_loss(angle, torch.zeros_like(angle), reduction='none')
    terms = dict(point=point, translation=(translation*object_weights).sum()/norm,
                 rotation=(rotation*object_weights).sum()/norm)
    return sum(terms.values()), terms
