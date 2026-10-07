"""Strict local-window supervision; metadata never enters the model."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .contracts import (K, K_EXEC, SCHEMA, ACTION_SEMANTICS, HAND_LINKS,
                        MIN_PREFERENCE_PAIRS, HISTORY_MATCH_MAX_RELATIVE_RMS)
from .supervision import CONTACT_SEMANTICS, RULE
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


def interaction_future(poses,points):
    """Measured future hand keypoints expressed in each future object frame."""
    if np.shape(points)!=(K+1,len(HAND_LINKS),3) or not np.isfinite(points).all():
        raise ValueError('current plus24 measured hand keypoint frames required')
    validate_rigid(poses)
    inverse=np.linalg.inv(poses[1:])
    return (np.einsum('tij,tkj->tki',inverse[:,:3,:3],points[1:])+inverse[:,:3,3][:,None]).astype('float32')


class Windows:
    """Read explicitly annotated local preferences, with episode-separated splits.

    No automatic episode-outcome-to-window label conversion is permitted.
    The producer must supply annotation provenance for each preference pair.
    """
    def __init__(self, root, require_complete=True):
        root = Path(root)
        self.manifest = json.loads((root / 'manifest.json').read_text())
        m = self.manifest
        if require_complete and m.get('status') != 'CONTRACT_PASS':
            raise ValueError('prepared data have not completed contract validation')
        required = dict(schema=SCHEMA, horizon=K, execution_horizon=K_EXEC,
                        fps=30, units='m', rollout_kind='continuous',
                        preference_scope='local_window', progress_scope='episode_absolute',
                        action_semantics=ACTION_SEMANTICS,
                        future_semantics='object_effect_and_measured_hand_object_relative_keypoints',
                        training_allowed=True)
        if any(m.get(key) != value for key, value in required.items()):
            raise ValueError('data contract mismatch; forked or unlabeled data cannot be fitted')
        if not m.get('history_contract') or not m.get('label_provenance'):
            raise ValueError('policy H contract and label provenance are required')
        provenance = m['label_provenance']
        if (provenance.get('rule') != RULE or provenance.get('contact_semantics') != CONTACT_SEMANTICS
                or provenance.get('rule_sha256') != sha(Path(__file__).with_name('supervision.py'))
                or provenance.get('contracts_sha256') != sha(Path(__file__).with_name('contracts.py'))
                or provenance.get('minimum_pair_coverage') != MIN_PREFERENCE_PAIRS
                or provenance.get('history_match_relative_rms') != HISTORY_MATCH_MAX_RELATIVE_RMS):
            raise ValueError('label provenance is stale for the current H-matched rule')
        if m.get('pair_coverage_required', True):
            if (m.get('all_experts_operationally_qualified') is not True
                    or provenance.get('all_experts_operationally_qualified') is not True):
                raise ValueError('prepared six-expert qualification provenance is missing')
            if (not m.get('route_sha256')
                    or provenance.get('route_sha256') != m['route_sha256']):
                raise ValueError('prepared route provenance is inconsistent')
            if (not m.get('label_report_sha256')
                    or provenance.get('report_sha256') != m['label_report_sha256']):
                raise ValueError('prepared label report provenance is inconsistent')
        if m.get('pair_coverage_required', True) and m.get('minimum_pair_coverage') != MIN_PREFERENCE_PAIRS:
            raise ValueError('minimum local preference coverage contract is missing or changed')
        path = root / 'windows.npz'
        if sha(path) != m.get('windows_sha256'):
            raise ValueError('window input identity changed')
        with np.load(path, allow_pickle=False) as source:
            required_arrays = {'history', 'action', 'effect','interaction','current_object','current_hand', 'progress', 'progress_mask',
                               'episode', 'split', 'task', 'phase', 'quality', 'pairs',
                               'pair_annotation','expert','motion'}
            if set(source.files) != required_arrays:
                raise ValueError('array whitelist mismatch')
            self.arrays = {name: source[name].copy() for name in source.files}
        a = self.arrays
        n = len(a['history'])
        if not n or a['history'].ndim < 2 or a['action'].shape != (n, K, 18):
            raise ValueError('H or requested residual plan shape mismatch')
        if a['effect'].shape != (n, K, 4, 4):
            raise ValueError('expected 24 rigid-anchor effects')
        validate_rigid(a['effect'])
        if a['interaction'].shape!=(n,K,len(HAND_LINKS),3) or not np.isfinite(a['interaction']).all():
            raise ValueError('invalid measured interaction future')
        if a['current_object'].shape!=(n,4,4) or a['current_hand'].shape!=(n,len(HAND_LINKS),3):
            raise ValueError('current physical-state matching metadata required')
        validate_rigid(a['current_object'])
        if not np.isfinite(a['current_hand']).all():raise ValueError('invalid current hand geometry')
        for name in ('history', 'action'):
            if not np.isfinite(a[name]).all():
                raise ValueError('nonfinite model input')
        if np.abs(a['action']).max() > .2 + 1e-6:
            raise ValueError('requested residual plans outside allowed amplitude')
        if a['progress'].shape != (n, K) or a['progress_mask'].shape != (n, K):
            raise ValueError('progress shape mismatch')
        if a['progress_mask'].dtype != np.bool_:
            raise ValueError('progress validity must be explicit boolean')
        valid = a['progress_mask']
        if (not np.isfinite(a['progress'][valid]).all()
                or ((a['progress'][valid] < 0) | (a['progress'][valid] > 1)).any()):
            raise ValueError('invalid absolute episode progress')
        for name in ('episode', 'split', 'task', 'phase', 'quality','expert','motion'):
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
        pair_directions = {}
        for left, right in pairs:
            key = tuple(sorted((int(left), int(right))))
            direction = (int(left), int(right))
            previous = pair_directions.get(key)
            if previous is not None and previous != direction:
                raise ValueError('contradictory local preference directions')
            pair_directions[key] = direction
        for name in ('split', 'task', 'phase','expert','motion'):
            if (a[name][chosen] != a[name][rejected]).any():
                raise ValueError('preference must match split, task, expert, motion and current phase')
        if (a['episode'][chosen] == a['episode'][rejected]).any():
            raise ValueError('cross-episode phase-matched preferences required')
        from .supervision import states_match
        if any(not states_match(dict(history=a['history'][i], object_pose=a['current_object'][i],
                                    hand_keypoints=a['current_hand'][i]),
                                dict(history=a['history'][j], object_pose=a['current_object'][j],
                                    hand_keypoints=a['current_hand'][j]), require_history=True)
               for i,j in pairs):
            raise ValueError('preference current H/object/hand states do not match')
        if len(set(map(tuple, pairs.tolist()))) != len(pairs):
            raise ValueError('duplicate preference pairs')
        groups = {split: set() for split in SPLITS}
        for i, j in pairs:
            split = str(a['split'][i])
            groups[split].add(tuple(sorted((str(a['episode'][i]), str(a['episode'][j])))))
        observed_groups = {split: len(values) for split, values in groups.items()}
        if (m.get('pair_coverage_required', True)
                and any(observed_groups[split] < minimum
                        for split, minimum in MIN_PREFERENCE_PAIRS.items())):
            raise ValueError('minimum unique episode-pair coverage is not met')
        self.pair_ids = {split: np.flatnonzero(a['split'][chosen] == split) for split in SPLITS}

    def batch(self, ids, device):
        """Return physical inputs and supervision separately."""
        a = self.arrays
        inputs = dict(history=torch.as_tensor(a['history'][ids], device=device).float().flatten(1),
                      action=torch.as_tensor(a['action'][ids], device=device).float(),
                      future=torch.cat((torch.as_tensor(a['effect'][ids, :, :3, :], device=device).float().flatten(2),
                                        torch.as_tensor(a['interaction'][ids],device=device).float().flatten(2)),dim=-1))
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
