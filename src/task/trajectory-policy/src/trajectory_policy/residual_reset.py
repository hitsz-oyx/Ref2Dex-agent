"""Measured teacher-prefix context for training-time phase resets.

Past measurements and actually applied commands are permitted reset context;
no teacher command or measured future is queried during the residual rollout.
"""
import numpy as np


def teacher_prefix_action(player, obs, envs):
    """Initialize the legacy player's batch seam before its first query."""
    raw = obs['obs'] if isinstance(obs, dict) else obs
    player.get_batch_size(raw, 1)
    padded = raw.repeat((64 + envs - 1) // envs, 1)[:64]
    return player.get_action(dict(obs=padded), True)[:envs].clamp(-1, 1)


def reset_context(prefix, starts, current):
    """Return true past H4 plus the reset measurement and previous applied action.

    A phase-zero reset repeats its own current measurement because no past
    exists. Later phases use three measured frames, never future frames.
    Lift always uses the teacher's task-start height, including after noise.
    """
    starts = np.asarray(starts, dtype=np.int64)
    n = len(starts)
    if starts.shape != (n,) or np.any(starts < 0) or np.any(starts >= len(prefix['obj'])):
        raise ValueError('phase outside measured prefix')
    history = {}
    for key in ('obj', 'hand'):
        live = np.asarray(current[key])
        if live.shape != prefix[key].shape[1:]:
            raise ValueError('reset measurement and teacher prefix shape mismatch')
        rows = []
        for offset in (-3, -2, -1):
            row = np.stack([prefix[key][max(0, int(phase) + offset), env]
                            if phase else live[env] for env, phase in enumerate(starts)])
            rows.append(row.copy())
        history[key] = rows + [live.copy()]
    previous = np.stack([prefix['applied'][int(phase) - 1, env] if phase else np.zeros(18, np.float32)
                         for env, phase in enumerate(starts)])
    return history, previous, prefix['obj'][0, :, 2, 3].copy()
