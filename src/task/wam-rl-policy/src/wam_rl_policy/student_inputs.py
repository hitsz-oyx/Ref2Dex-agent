"""Partial-observation student inputs shared by BC fitting and deployment."""
import numpy as np

P2_DIM = 139
FULL_DIM = 300
TAU_SHAPE = (8, 11, 3)
TAU_DIM = int(np.prod(TAU_SHAPE))


def student_features(observation, variant, *, tau=None, pad_to=0):
    values = np.asarray(observation, dtype=np.float32)
    expected = FULL_DIM if variant in ('full', 'full_tau') else P2_DIM
    if values.ndim < 1 or values.shape[-1] != expected:
        raise ValueError('observation must end in %d channels for %s' % (expected, variant))
    if variant in ('p2_tau', 'full_tau'):
        if tau is None:
            raise ValueError('p2_tau requires the current-state oracle trajectory')
        trajectory = np.asarray(tau, dtype=np.float32)
        leading = values.shape[:-1]
        if trajectory.shape not in (leading + TAU_SHAPE, leading + (TAU_DIM,)):
            raise ValueError('tau must match the observation batch and 8x11x3 contract')
        values = np.concatenate((values, trajectory.reshape(leading + (TAU_DIM,))), axis=-1)
    elif variant not in ('p2', 'full'):
        raise ValueError('unknown student input variant')
    if pad_to:
        if pad_to < values.shape[-1]:
            raise ValueError('pad-to is smaller than the input dimension')
        padding = np.zeros(values.shape[:-1] + (pad_to - values.shape[-1],), np.float32)
        values = np.concatenate((values, padding), axis=-1)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite student observation or trajectory')
    return values
