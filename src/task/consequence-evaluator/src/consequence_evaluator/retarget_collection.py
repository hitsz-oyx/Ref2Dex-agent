"""Structured bounded residuals used to collect full-action retarget data.

The collector keeps the frozen actor as the ordinary controller and adds a
small, deterministic residual family.  The saved target is the resulting
full native action, so the retargeter never needs the actor action at train or
execution time.
"""

import numpy as np


PHASE_NAMES = ("approach", "contact", "hold")
MODE_NAMES = (
    "wrist_translation",
    "wrist_rotation",
    "finger_open_close",
    "finger_preload",
    "wrist_finger",
)
ACTIVE_FINGERS = np.asarray([6, 8, 10, 12, 14, 15], dtype=np.int64)
STRUCTURED_PROFILE_NAMES = ("random", "zero", "finger-pulse", "paired-triplet")


def finger_pulse_residual(count, finger_index, value):
    """Build a one-channel diagnostic residual without sampling other joints.

    This is intentionally separate from :func:`sample_structured_residual`:
    the pulse profile is used only by the serial paired-capture diagnostic,
    while the ordinary structured rollout keeps its registered random family.
    ``finger_index`` names one of the six independently commanded native
    finger coordinates; the environment applies its coupled coordinates.
    """
    count = int(count)
    finger_index = int(finger_index)
    value = float(value)
    if count < 1:
        raise ValueError("positive residual count required")
    if finger_index not in set(int(index) for index in ACTIVE_FINGERS):
        raise ValueError("finger pulse must use an independently commanded finger")
    if not np.isfinite(value) or abs(value) > .120001:
        raise ValueError("finger pulse exceeds the registered residual bound")
    residual = np.zeros((count, 18), dtype=np.float32)
    residual[:, finger_index] = np.float32(value)
    return residual


def paired_triplet_pulse(count, finger_index, value):
    """Return ``control/+d/-d`` residuals in consecutive triplet order."""
    count = int(count)
    finger_index = int(finger_index)
    value = float(value)
    if count < 1 or count % 3:
        raise ValueError("paired triplet count must be a positive multiple of three")
    if finger_index not in set(int(index) for index in ACTIVE_FINGERS):
        raise ValueError("finger pulse must use an independently commanded finger")
    if not np.isfinite(value) or abs(value) > .120001:
        raise ValueError("finger pulse exceeds the registered residual bound")
    residual = np.zeros((count, 18), dtype=np.float32)
    residual[1::3, finger_index] = np.float32(value)
    residual[2::3, finger_index] = np.float32(-value)
    return residual


def paired_pulse_ticks(groups, phase):
    """Choose one deterministic pulse tick per triplet for a phase pilot."""
    groups = int(groups)
    if groups < 1:
        raise ValueError("positive paired group count required")
    if phase == "contact":
        start = 120
    elif phase == "hold":
        start = 240
    else:
        raise ValueError("paired phase must be contact or hold")
    ticks = np.arange(start, start + groups, dtype=np.int32)
    if int(ticks[-1]) >= 542:
        raise ValueError("paired pulse schedule exceeds the command sequence")
    return ticks


def phase_code(tick, contact_tick=120, hold_tick=240):
    """Return the fixed phase bucket used by the collection manifest."""
    tick = int(tick)
    if tick < contact_tick:
        return 0
    if tick < hold_tick:
        return 1
    return 2


def sample_structured_residual(rng, count, phase):
    """Sample one bounded residual and its mode for every environment.

    The values are action-space residuals.  Wrist translation uses metres per
    native tick, wrist rotation uses the native ``[-1, 1]`` rotation scale,
    and finger entries use the native target scale.  Dependent finger joints
    remain zero because the environment computes their coupling.
    """
    if not hasattr(rng, "uniform") or int(count) < 1:
        raise ValueError("a NumPy generator and positive count are required")
    phase = int(phase)
    if phase not in range(len(PHASE_NAMES)):
        raise ValueError("unknown collection phase")
    count = int(count)
    modes = rng.integers(0, len(MODE_NAMES), size=count)
    residual = np.zeros((count, 18), dtype=np.float32)

    # Keep approach motion small enough to retain ordinary contact examples;
    # contact/hold samples spend more of their dose on finger preload.
    wrist_scale = (1.0, 0.75, 0.55)[phase]
    finger_scale = (0.55, 0.85, 1.0)[phase]
    for row, mode in enumerate(modes):
        mode = int(mode)
        if mode in (0, 4):
            residual[row, :3] = rng.uniform(-1., 1., size=3).astype(np.float32)
            residual[row, :3] *= np.asarray([.010, .010, .015], dtype=np.float32) * wrist_scale
        if mode == 1:
            residual[row, 3:6] = rng.uniform(-1., 1., size=3).astype(np.float32)
            residual[row, 3:6] *= np.asarray([.035, .035, .050], dtype=np.float32) * wrist_scale
        if mode in (2, 3, 4):
            values = rng.uniform(-1., 1., size=len(ACTIVE_FINGERS)).astype(np.float32)
            if mode == 2:
                # A shared sign gives open/close coverage while the small
                # independent term prevents a single-finger coordinate from
                # being perfectly collinear with the others.
                sign = -1. if rng.integers(0, 2) == 0 else 1.
                values = sign * (.70 + .30 * values)
            elif mode == 3:
                # Preload is intentionally asymmetric across fingers.
                values *= .65
                values[rng.integers(len(values))] += rng.choice((-1., 1.))
            residual[row, ACTIVE_FINGERS] = np.clip(
                values * (.12 * finger_scale), -.12, .12).astype(np.float32)
    return residual, modes.astype(np.int8)


def validate_residual_family(residual):
    """Validate a saved residual tensor before it enters a manifest."""
    value = np.asarray(residual)
    if value.ndim != 3 or value.shape[-1] != 18 or not np.isfinite(value).all():
        raise ValueError("structured residual tensor must be finite [T,N,18]")
    if np.any(np.abs(value[:, :, :3]) > np.asarray([.010001, .010001, .015001])) or np.any(
            np.abs(value[:, :, 3:6]) > np.asarray([.035001, .035001, .050001])):
        raise ValueError("wrist residual exceeds the registered bound")
    if np.max(np.abs(value[:, :, ACTIVE_FINGERS])) > .120001:
        raise ValueError("finger residual exceeds the registered bound")
    return True
