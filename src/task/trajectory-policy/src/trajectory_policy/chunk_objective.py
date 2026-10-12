"""First-control objective for a chunk replanned at every native tick."""
import numpy as np
import torch

PD_UNITS = (.01, .1, .1)  # metres, wrist radians, independent finger radians


def independent_PD_scale(native_scale):
    """Native12 error -> independent PD error; finger conversion halves scale."""
    scale = native_scale.clone()
    if scale.shape != (12,) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('positive finite independent native12 scale required')
    scale[6:] /= 2
    return scale


def control_aware_loss(prediction, target, action_scale, pd_scale):
    """4*first + mean(tail7) + group-balanced physical first-target error.

    With the same measured current q and controller offset, both cancel from
    the independent PD-target difference. Use raw predictions during training
    so out-of-bounds outputs retain a corrective gradient; deployment clamps.
    """
    standardized = ((prediction-target)/action_scale).abs()
    first = standardized[:, 0].mean()
    tail = standardized[:, 1:].mean()
    delta = (prediction[:, 0]-target[:, 0])*pd_scale
    pd0 = sum(delta[:, start:end].abs().mean()/unit
              for (start, end), unit in zip(((0,3),(3,6),(6,12)), PD_UNITS))/3
    total = 4*first + tail + pd0
    return total, dict(first_action=first, tail_chunk=tail, first_PD=pd0)


def development_key(prediction, target, action_scale, pd_scale):
    """Lexicographic first applied wrist p95, then all8 standardized L1."""
    delta = (prediction[:, 0].clip(-1,1)-target[:, 0])*np.asarray(pd_scale,dtype=np.float64)
    p95 = float(np.percentile(np.linalg.norm(delta[:, :3], axis=-1), 95)*1000)
    chunk = float((np.abs(prediction-target)/action_scale).mean())
    return p95, chunk
