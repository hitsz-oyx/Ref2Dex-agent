"""Strict local-window supervision; metadata never enters the model."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

K = 24
K_EXEC = 8
SCHEMA = 'ref2dex.consequence-evaluator.windows.v1'
SPLITS = ('train', 'val', 'test')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def object_effect(poses):
    """T_t^-1 T_(t+1:t+24), matching PointWorld's rigid-anchor output.

    The input is 25 measured object-to-world transforms. Action[t] produces
    pose[t+1]. Coordinates are metres in the stationary simulator frame.
    """
    poses = np.asarray(poses)
    if poses.shape != (K + 1, 4, 4):
        raise ValueError('expected current plus 24 post-action poses')
    validate_rigid(poses)
    return (np.linalg.inv(poses[0]) @ poses[1:]).astype('float32')


def validate_rigid(poses):
    rotation = poses[..., :3, :3]
    if (not np.isfinite(poses).all()
            or not np.allclose(poses[..., 3, :], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1e-3)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-3)):
        raise ValueError('nonrigid or nonfinite object effects')


class Windows:
    """Read explicitly annotated local preferences, with episode-separated splits.

    No automatic episode-outcome-to-window label conversion is permitted.
    The producer must supply annotation provenance for each preference pair.
    """
    def __init__(self, root):
        root = Path(root)
        self.manifest = json.loads((root / 'manifest.json').read_text())
        m = self.manifest
        required = dict(schema=SCHEMA, horizon=K, execution_horizon=K_EXEC,
                        fps=30, units='m', rollout_kind='continuous',
                        preference_scope='local_window', progress_scope='episode_absolute',
                        action_semantics='executed_native_control',
                        future_semantics='current_anchor_frame_rigid_effect',
                        training_allowed=True)
        if any(m.get(key) != value for key, value in required.items()):
            raise ValueError('data contract mismatch; forked or unlabeled data cannot be fitted')
        if not m.get('history_contract') or not m.get('label_provenance'):
            raise ValueError('policy H contract and label provenance are required')
        path = root / 'windows.npz'
        if sha(path) != m.get('windows_sha256'):
            raise ValueError('window input identity changed')
        with np.load(path, allow_pickle=False) as source:
            required_arrays = {'history', 'action', 'effect', 'progress', 'progress_mask',
                               'episode', 'split', 'task', 'phase', 'quality', 'pairs',
                               'pair_annotation'}
            if set(source.files) != required_arrays:
                raise ValueError('array whitelist mismatch')
            self.arrays = {name: source[name].copy() for name in source.files}
        a = self.arrays
        n = len(a['history'])
        if not n or a['history'].ndim < 2 or a['action'].shape != (n, K, 18):
            raise ValueError('H or executed action shape mismatch')
        if a['effect'].shape != (n, K, 4, 4):
            raise ValueError('expected 24 rigid-anchor effects')
        validate_rigid(a['effect'])
        for name in ('history', 'action'):
            if not np.isfinite(a[name]).all():
                raise ValueError('nonfinite model input')
        if np.abs(a['action']).max() > 1 + 1e-6:
            raise ValueError('executed native controls outside [-1,1]')
        if a['progress'].shape != (n, K) or a['progress_mask'].shape != (n, K):
            raise ValueError('progress shape mismatch')
        if a['progress_mask'].dtype != np.bool_:
            raise ValueError('progress validity must be explicit boolean')
        valid = a['progress_mask']
        if (not np.isfinite(a['progress'][valid]).all()
                or ((a['progress'][valid] < 0) | (a['progress'][valid] > 1)).any()):
            raise ValueError('invalid absolute episode progress')
        for name in ('episode', 'split', 'task', 'phase', 'quality'):
            if a[name].shape != (n,) or a[name].dtype.kind not in 'US':
                raise ValueError('per-window string metadata required: ' + name)
        if not set(a['split']).issubset(SPLITS):
            raise ValueError('unknown split')
        for episode in np.unique(a['episode']):
            if len(np.unique(a['split'][a['episode'] == episode])) != 1:
                raise ValueError('episode leakage across splits')
        if valid[a['quality'] != 'expert_success'].any():
            raise ValueError('failed/suboptimal dense progress must be masked')
        pairs = a['pairs']
        if (pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs)
                or pairs.dtype.kind not in 'iu' or pairs.min() < 0 or pairs.max() >= n):
            raise ValueError('expected chosen/rejected local-window indices')
        if a['pair_annotation'].shape != (len(pairs),) or a['pair_annotation'].dtype.kind not in 'US':
            raise ValueError('explicit pair annotations required')
        if any(not str(value).strip() for value in a['pair_annotation']):
            raise ValueError('unattributed local preference')
        chosen, rejected = pairs.T
        for name in ('split', 'task', 'phase'):
            if (a[name][chosen] != a[name][rejected]).any():
                raise ValueError('preference must match split, task and current phase')
        if (a['episode'][chosen] == a['episode'][rejected]).any():
            raise ValueError('cross-episode phase-matched preferences required')
        if len(set(map(tuple, pairs.tolist()))) != len(pairs):
            raise ValueError('duplicate preference pairs')
        self.pair_ids = {split: np.flatnonzero(a['split'][chosen] == split) for split in SPLITS}

    def batch(self, ids, device):
        """Return physical inputs and supervision separately."""
        a = self.arrays
        inputs = dict(history=torch.as_tensor(a['history'][ids], device=device).float().flatten(1),
                      action=torch.as_tensor(a['action'][ids], device=device).float(),
                      future=torch.as_tensor(a['effect'][ids, :, :3, :], device=device).float().flatten(2))
        mask = a['progress_mask'][ids]
        progress = np.where(mask, a['progress'][ids], 0)
        labels = dict(progress=torch.as_tensor(progress, device=device).float(),
                      progress_mask=torch.as_tensor(mask, device=device))
        return inputs, labels

    def normalization(self):
        """Only train windows fit statistics; no task, phase or labels are read."""
        ids = np.flatnonzero(self.arrays['split'] == 'train')
        if not len(ids):
            raise ValueError('empty train split')
        inputs, _ = self.batch(ids, 'cpu')
        return {key: (value.mean(tuple(range(value.ndim - 1))),
                      value.std(tuple(range(value.ndim - 1)), unbiased=False).clamp_min(1e-4))
                for key, value in inputs.items()}
