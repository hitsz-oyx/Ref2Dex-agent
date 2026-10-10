"""Actual future-hand / applied native-action supervision, without reference inputs."""
import numpy as np

ACTIVE = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15)
STATE_DIM = 87


def valid_windows(data, horizon=24, prefix=8):
    length = len(data['q'])
    indices = np.arange(max(length-max(horizon, prefix-1), 0))
    offsets = np.arange(max(horizon, prefix-1)+1)
    future = indices[:, None]+offsets
    continuous = (data['episode_start'][future] == data['episode_start'][indices, None]).all(1)
    continuous &= (data['episode_tick'][future] == data['episode_tick'][indices, None]+offsets).all(1)
    return indices[continuous]


def features(q, dq, hand, obj, velocity, future_hand):
    """One current state; displacement in current object's frame. No future obj/q."""
    rotation = obj[..., :3, :3]
    center = obj[..., :3, 3]
    local_hand = np.einsum('...pj,...jk->...pk', hand-center[..., None, :], rotation)
    state = np.concatenate((q, dq, local_hand.reshape(*q.shape[:-1], 33),
        obj[..., :3, :].reshape(*q.shape[:-1], 12), velocity), -1).astype(np.float32)
    tau = np.einsum('...tpj,...jk->...tpk', future_hand-hand[..., None, :, :], rotation)
    if state.shape[-1] != STATE_DIM or tau.shape[-2:] != (11, 3):
        raise ValueError('retargeter measured-state/hand shape mismatch')
    return state, tau.reshape(*tau.shape[:-2], 33).astype(np.float32)


def build_windows(data, indices, horizon=24, prefix=8):
    if not np.array_equal(data['action'], data['applied']):
        raise ValueError('labels must equal actually applied commands')
    if not np.isin(indices, valid_windows(data, horizon, prefix)).all():
        raise ValueError('cross-reset or incomplete window')
    future_hand = data['hand'][indices[:, None]+np.arange(1, horizon+1)]
    future_hand = np.moveaxis(future_hand, 2, 1)  # query, env, future, point, xyz
    state, tau = features(*(data[key][indices] for key in ('q', 'dq', 'hand', 'obj', 'velocity')), future_hand)
    action = data['applied'][indices[:, None]+np.arange(prefix)]
    action = np.moveaxis(action, 2, 1)[..., list(ACTIVE)]
    return state, tau, action.astype(np.float32)
