"""Actor-only window weighting; source labels and world supervision are untouched."""
import torch
from torch.nn import functional as F


def weighted_trajectory_loss(prediction, target, valid, window_weight, scale=.05):
    if window_weight.shape != (len(prediction),):
        raise ValueError('one actor weight per window is required')
    if not torch.isfinite(window_weight).all() or (window_weight < 0).any():
        raise ValueError('actor weights must be finite and nonnegative')
    weight = valid.to(torch.float32) * window_weight[:, None, None, None]
    denominator = weight.sum()
    if denominator <= 0:
        raise ValueError('empty actor supervision')
    error = F.huber_loss(prediction.float() / scale, target.float() / scale,
                         reduction='none').mean(-1)
    return (error * weight).sum() / denominator
