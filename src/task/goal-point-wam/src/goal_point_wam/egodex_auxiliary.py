"""Native hand-only auxiliary batches for the unchanged object-goal WAM."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EgoDexActorWindows(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.meta = json.loads((self.root/'manifest.json').read_text())
        if (self.meta.get('schema') != 'goal-point-wam.egodex-native-actor.v1'
                or self.meta.get('status') != 'COMPLETED' or not self.meta.get('training_allowed')
                or self.meta.get('supervision') != 'native_hand_trajectory_only'
                or self.meta.get('world_supervision_available') or self.meta.get('scene_teacher_admitted')):
            raise ValueError('unqualified EgoDex hand auxiliary corpus')
        if (self.meta['fps'], self.meta['history'], self.meta['horizon']) != (30, 4, 24):
            raise ValueError('native EgoDex time contract mismatch')
        self.records = self.meta['records']
        index = self.root/f'index_{split}.npy'
        if hashlib.sha256(index.read_bytes()).hexdigest() != self.meta['index_hashes'][split]:
            raise ValueError('EgoDex index identity changed')
        self.rows = np.load(index)
        if self.rows.ndim != 2 or self.rows.shape[1] != 2 or not len(self.rows):
            raise ValueError('invalid auxiliary index')
        for record_id in np.unique(self.rows[:, 0]):
            if self.records[int(record_id)]['split'] != split:
                raise ValueError('train/dev record leakage')
        train_sessions = {r['session'] for r in self.records if r['split'] == 'train'}
        val_sessions = {r['session'] for r in self.records if r['split'] == 'val'}
        if train_sessions & val_sessions:
            raise ValueError('train/dev session leakage')

    def __len__(self):
        return len(self.rows)

    @lru_cache(maxsize=32)
    def record(self, i):
        r = self.records[i]
        path = self.root/r['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != r['sha256']:
            raise ValueError('EgoDex source arrays changed')
        with np.load(path) as f:
            return {k:f[k].copy() for k in f.files}

    def __getitem__(self, i):
        record_id, tick = map(int, self.rows[i])
        d = self.record(record_id)
        hand, valid = d['hand'], d['valid']
        if not 3 <= tick < len(hand)-24:
            raise ValueError('noncontiguous H4/K24 window')
        # Calibration at tick is observed. No future camera transforms enter.
        C = np.linalg.inv(d['camera'][tick])
        h = hand[tick-3:tick+25] @ C[:3, :3].T + C[:3, 3]
        history_valid = valid[tick-3:tick+1].all(0)
        # Suppress dynamics whenever any of the four history measurements is missing.
        start = np.where(history_valid[..., None], h[3], 0).astype('float32')
        dynamics = np.concatenate(((h[3]-h[2])*30, (h[3]-2*h[2]+h[1])*900, h[3]-h[0]), -1)
        dynamics = np.where(history_valid[..., None], dynamics, 0).astype('float32')
        trajectory_valid = valid[tick+1:tick+25] & history_valid[None]
        target = np.where(trajectory_valid[..., None], h[4:]-start[None], 0).astype('float32')
        flat_valid = history_valid.reshape(22)
        metadata = np.stack((np.zeros(22), np.zeros(22), np.ones(22), np.repeat([0, 1], 11),
                             np.tile(np.arange(11)/10, 2), np.zeros(22)), -1)
        features = np.concatenate((np.zeros((22, 3)), dynamics.reshape(22, 9), metadata), -1).astype('float32')
        if flat_valid.sum() < 6 or not trajectory_valid.any():
            raise ValueError('auxiliary window has no observed hand supervision')
        xyz = start.reshape(22, 3)[flat_valid]
        if not np.isfinite(xyz).all() or (np.abs(xyz) > 1.99).any():
            raise ValueError('native hands outside fixed sparse workspace')
        return dict(xyz=xyz, features=features[flat_valid], start_hand=start,
                    start_hand_history=dynamics, start_hand_valid=history_valid,
                    trajectory_target=target, trajectory_valid=trajectory_valid,
                    start_object_points=np.zeros((32, 3), 'float32'),
                    goal_object_points=np.zeros((32, 3), 'float32'),
                    object_goal_valid=np.zeros(32, bool), sample_id=np.asarray([record_id, tick], 'int64'))


def ego_collate(samples):
    n, size = len(samples), max(len(s['xyz']) for s in samples)
    out = {k:torch.as_tensor(np.stack([s[k] for s in samples])) for k in samples[0]
           if k not in ('xyz', 'features')}
    for key, dims in [('xyz', 3), ('features', 18)]:
        out[key] = torch.zeros((n, size, dims))
        for row, s in enumerate(samples):
            out[key][row, :len(s[key])] = torch.from_numpy(s[key])
    out['point_valid'] = torch.arange(size)[None] < torch.tensor([len(s['xyz']) for s in samples])[:, None]
    out['scene_object'] = torch.full((n, size), -1, dtype=torch.long)
    # This slot is padding for the scene encoder tensor contract, never a target.
    out['object_features'] = torch.zeros((n, 1, 15))
    out['object_valid'] = torch.zeros((n, 1), dtype=torch.bool)
    out['ego_actor_only'] = torch.tensor(True)
    return out


def auxiliary_model_class(base_class):
    class NativeHandAuxiliaryWAM(base_class):
        def forward(self, batch):
            if not batch.get('ego_actor_only', False):
                return super().forward(batch)
            if batch['object_goal_valid'].any() or batch['object_valid'].any():
                raise ValueError('EgoDex must not invent object goals or world labels')
            scene = self.scene(batch)
            tokens, mask = self.goal(batch['start_object_points'], batch['goal_object_points'], batch['object_goal_valid'])
            prediction = self.trajectory(scene['scene_tokens'], scene['scene_mask'], tokens, mask,
                                         self.observed_hand_condition(batch))
            # Avoid the all-masked world decoder, including its NaN*0 failure mode.
            return dict(trajectory=prediction)

        def loss(self, prediction, batch):
            if not batch.get('ego_actor_only', False):
                return super().loss(prediction, batch)
            trajectory = self.trajectory_loss(prediction['trajectory'], batch)
            zero = trajectory.detach().new_zeros(())
            return trajectory, dict(trajectory=trajectory.detach(), world=zero,
                                    world_point=zero, world_translation=zero, world_rotation=zero)
    return NativeHandAuxiliaryWAM
