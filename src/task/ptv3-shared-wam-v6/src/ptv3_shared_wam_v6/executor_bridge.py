"""Explicit v6 joint and clock adapters; these do not wire the official PPO."""
import numpy as np
from ptv3_shared_wam_v5.contracts import SPARSE11_IN_21


def full21_to_sparse11(trajectory):
    """[...,24,2,21,3] -> [...,24,2,11,3], preserving both hand slots/masks.

    Works for NumPy arrays and Torch tensors. Input values are displacements;
    this index selection never adds the current hand or object translation.
    """
    if tuple(trajectory.shape[-4:]) != (24, 2, 21, 3):
        raise ValueError('v6 trajectory must be [...,24,2,21,3]')
    return trajectory[..., list(SPARSE11_IN_21), :]


def world_tau_to_executor(trajectory, *, native_hz=30, native_order=False):
    """Runtime world displacement -> eight future native ticks, optionally Inspire order.

    OnlineV6Runtime has already selected right and rotated to world. Prepend
    tau(0)=0 for interpolation; never rotate again or rescale displacement.
    Row zero is t+1/30, not the current instant. Native order is optional since
    ExecutorLayout uses canonical sparse11 while the saved Inspire686 uses native11.
    """
    values = np.asarray(trajectory, dtype=np.float32)
    if values.shape[-3:] != (24, 11, 3) or not np.isfinite(values).all():
        raise ValueError('finite world tau [...,24,11,3] required')
    if not np.isscalar(native_hz) or not np.isfinite(native_hz) or native_hz <= 0 or 8/native_hz > 24/30:
        raise ValueError('positive native clock within K24 coverage required')
    target = np.arange(1, 9)*30/float(native_hz)
    lo = np.floor(target).astype(int); hi = np.minimum(lo+1, 24)
    alpha = (target-lo)[:, None, None]
    source = np.concatenate((np.zeros_like(values[..., :1, :, :]), values), axis=-3)
    result = source[..., lo, :, :]*(1-alpha) + source[..., hi, :, :]*alpha
    if native_order:
        # Fixed Inspire mapping, checked against its production contract in tests.
        result = result[..., [0, 1, 6, 2, 7, 3, 8, 4, 9, 5, 10], :]
    return result.astype(np.float32)
