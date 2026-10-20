"""Shadow BC sampling, with the startup window expressed in physical time."""
import numpy as np

STARTUP_SECONDS = 8 / 60
SAMPLERS = ('startup_half', 'uniform')


def startup_rows(ticks, native_hz):
    ticks = np.asarray(ticks)
    if (ticks.ndim != 1 or not np.isfinite(ticks).all() or (ticks < 0).any()
            or not np.isfinite(native_hz) or native_hz <= 0):
        raise ValueError('finite nonnegative original ticks and positive clock required')
    return np.flatnonzero(ticks / native_hz < STARTUP_SECONDS)


def sample_rows(rng, count, early, batch, mode='startup_half'):
    if mode not in SAMPLERS or count < 1 or batch < 1:
        raise ValueError('known BC sampler and positive row/batch counts required')
    if mode == 'uniform':
        return rng.integers(0, count, size=batch)
    if not len(early):
        raise ValueError('startup-half BC requires real startup observations')
    half = batch // 2
    return np.concatenate((rng.choice(early, half), rng.integers(0, count, size=batch-half)))
