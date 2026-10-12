"""Explicit point-track admission; observations and offline labels are separate."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class VideoPointWindows(Dataset):
    def __init__(self, root, split, engineering=False):
        self.root = Path(root)
        self.meta = json.loads((self.root/'manifest.json').read_text())
        if self.meta.get('schema') != 'goal-point-wam.video-points.v1' or self.meta.get('status') != 'COMPLETED':
            raise ValueError('unqualified video point manifest')
        if (self.meta.get('fps'), self.meta.get('history'), self.meta.get('horizon')) != (30, 4, 24):
            raise ValueError('H4/K24 time contract mismatch')
        if not self.meta.get('causal_observation'):
            if not (self.meta.get('observation_policy') == 'offline_clip_reconstruction'
                    and self.meta.get('offline_pretraining_only') is True
                    and self.meta.get('label_depth_backend') == 'offline-teacher'):
                raise ValueError('current/past observations or explicitly scoped offline pretraining required')
        if not self.meta.get('training_allowed') and not engineering:
            raise ValueError('engineering corpus cannot enter scientific training')
        records = self.meta['records']
        sessions = {s: {r['session'] for r in records if r['split'] == s} for s in ('train', 'val')}
        if sessions['train'] & sessions['val']:
            raise ValueError('video recording session leakage')
        if any(r['source_split'] != 'official_train' for r in records):
            raise ValueError('only verified official TRAIN videos admitted')
        self.records = [r for r in records if r['split'] == split]
        if not self.records:
            raise ValueError('empty video split')

    def __len__(self):
        return len(self.records)

    @lru_cache(maxsize=16)
    def __getitem__(self, i):
        record = self.records[i]
        path = self.root/record['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('video point arrays changed')
        with np.load(path, allow_pickle=False) as f:
            d = {k: f[k].copy() for k in f.files}
        xyz, valid = d['query_xyz'], d['query_valid'].astype(bool)
        n = len(xyz)
        if (xyz.shape != (n, 3) or d['point_target'].shape != (24, n, 3)
                or d['point_target_valid'].shape != (24, n)
                or len(np.unique(d['track_ids'])) != n):
            raise ValueError('unstable point correspondence')
        if d['hand'].shape != (28, 2, 11, 3) or d['hand_valid'].shape != (28, 2, 11):
            raise ValueError('native semantic hands required; do not pad10-point palm labels into11')
        hand, hv = d.pop('hand'), d.pop('hand_valid').astype(bool)
        known = hv[:4].all(0)
        start = np.where(known[..., None], hand[3], 0.).astype('float32')
        history = np.concatenate(((hand[3]-hand[2])*30, (hand[3]-2*hand[2]+hand[1])*900, hand[3]-hand[0]), -1)
        history = np.where(known[..., None], history, 0.).astype('float32')
        # Match the retained native action/statistics contract exactly:
        # absolute XYZ, displacement from current XYZ, one-tick displacement.
        # These are meter displacements, NOT velocity/acceleration features.
        action_valid = hv[4:] & hv[3][None] & hv[3:-1]
        action = np.concatenate((hand[4:], hand[4:]-hand[3], hand[4:]-hand[3:-1]), -1)
        action = np.where(action_valid[..., None], action, 0.).astype('float32')
        tvalid = hv[4:] & known[None]
        features = np.zeros((n, 18), 'float32')
        # Unknown target/static/local-instance metadata stays zero; field17 is
        # a native instance id, not a generic point-valid flag.
        hand_features = np.zeros((22, 18), 'float32')
        hand_features[:, 3:12] = history.reshape(22, 9)
        hand_features[:, 14] = 1
        hand_features[:, 15] = np.repeat([0, 1], 11)
        hand_features[:, 16] = np.tile(np.arange(11)/10, 2)
        for arr, mask in ((xyz, valid), (d['point_target'], d['point_target_valid']), (hand, hv)):
            if not np.isfinite(arr[mask]).all():
                raise ValueError('nonfinite valid point label')
        d['query_xyz'] = np.where(valid[..., None], xyz, 0.).astype('float32')
        if (np.abs(xyz[valid]) > 1.99).any() or (np.abs(start[known]) > 1.99).any():
            raise ValueError('observations outside sparse workspace')
        if not np.any(d['point_target_valid'] & valid[None]):
            raise ValueError('no video world supervision')
        d['point_target_valid'] &= valid[None]
        d.update(features=features, hand_features=hand_features, start_hand=start, start_hand_valid=known,
                 start_hand_history=history, trajectory_target=np.where(tvalid[..., None], hand[4:]-start[None], 0.).astype('float32'),
                 trajectory_valid=tvalid, action=action.reshape(24, 22, 9), action_valid=action_valid.reshape(24, 22))
        return d


def video_collate(samples):
    b, n = len(samples), max(len(s['query_xyz']) for s in samples)
    result = {}
    for key in ('start_hand', 'start_hand_valid', 'start_hand_history', 'trajectory_target', 'trajectory_valid',
                'action', 'action_valid', 'start_object_points', 'goal_object_points', 'object_goal_valid'):
        result[key] = torch.from_numpy(np.stack([s[key] for s in samples]))
    result.update(xyz=torch.zeros(b, n+22, 3), features=torch.zeros(b, n+22, 18),
                  point_valid=torch.zeros(b, n+22, dtype=torch.bool),
                  query_xyz=torch.zeros(b, n, 3), query_valid=torch.zeros(b, n, dtype=torch.bool),
                  point_target=torch.zeros(b, 24, n, 3), point_target_valid=torch.zeros(b, 24, n, dtype=torch.bool),
                  track_ids=torch.full((b, n), -1, dtype=torch.long),
                  scene_object=torch.full((b, n+22), -1, dtype=torch.long),
                  object_features=torch.zeros(b, 1, 15), object_valid=torch.zeros(b, 1, dtype=torch.bool),
                  video_points=torch.tensor(True))
    for row, s in enumerate(samples):
        size = len(s['query_xyz'])
        for key in ('query_xyz', 'query_valid', 'track_ids'):
            result[key][row, :size] = torch.from_numpy(s[key])
        for key in ('point_target', 'point_target_valid'):
            result[key][row, :, :size] = torch.from_numpy(s[key])
        result['xyz'][row, :size] = torch.from_numpy(s['query_xyz'])
        result['features'][row, :size] = torch.from_numpy(s['features'])
        result['point_valid'][row, :size] = torch.from_numpy(s['query_valid'])
        result['xyz'][row, n:] = result['start_hand'][row].reshape(22, 3)
        result['features'][row, n:] = torch.from_numpy(s['hand_features'])
        result['point_valid'][row, n:] = result['start_hand_valid'][row].flatten()
    return result
