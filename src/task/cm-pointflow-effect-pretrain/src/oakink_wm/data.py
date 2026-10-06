"""Fixed-current-frame 30Hz windows with masked hands and current-only local selection."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

HISTORY = 4
HORIZON = 24


def transform_points(points, pose):
    return np.einsum('...ij,...nj->...ni', pose[..., :3, :3], points) + pose[..., None, :3, 3]


def relative_effect(anchor_now, object_now, object_future):
    """F=(C*T_future)*inverse(C*T_now), with ONE current C for the window."""
    C = np.linalg.inv(anchor_now)
    now = C @ object_now
    future = C @ object_future
    return future @ np.linalg.inv(now)


def local_objects(poses, valid, anchor, canonical_centers=None):
    centers = poses[:, :3, 3]
    if canonical_centers is not None:
        centers = np.einsum('mij,mj->mi', poses[:, :3, :3], canonical_centers) + centers
    return np.flatnonzero(valid & (np.linalg.norm(centers - centers[anchor], axis=-1) <= .5))


class Windows(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.meta = json.loads((self.root / 'processed/manifest.json').read_text())
        assert self.meta['status'] == 'COMPLETED' and self.meta['fps'] == 30
        self.rows = np.load(self.root / ('processed/index_' + split + '.npy'), mmap_mode='r')
        self.sequences = self.meta['sequences']
        self.groups = [np.flatnonzero(self.rows[:, 3] == k) for k in range(3)]
        if not len(self.rows): raise ValueError('empty split ' + split)

    @lru_cache(maxsize=32)
    def sequence(self, seq):
        path = self.root / 'processed/sequences' / seq
        meta = json.loads((path / 'meta.json').read_text())
        arrays = {key: np.load(path / (key + '.npy'), mmap_mode='r') for key in
                  ('hand', 'hand_valid', 'poses', 'pose_valid', 'program', 'near', 'frame_ids', 'centers')}
        arrays['objects'] = meta['objects']
        return arrays

    @lru_cache(maxsize=256)
    def canonical(self, obj):
        with np.load(self.root / 'processed/canonical' / (obj + '.npz')) as f:
            return f['points'].copy(), f['normals'].copy(), float(f['radius']), f['center'].copy()

    def __len__(self): return len(self.rows)

    def __getitem__(self, item):
        s, anchor, tick, category = map(int, self.rows[item])
        seq = self.sequences[s]
        d = self.sequence(seq)
        hist = np.arange(tick - 3, tick + 1)
        future = np.arange(tick + 1, tick + 25)
        poses = d['poses']
        selected = local_objects(poses[tick], d['pose_valid'][tick], anchor, d['centers'])
        assert anchor in selected
        assert np.array_equal(d['frame_ids'][np.arange(tick-3, tick+25)], d['frame_ids'][tick] + 4*np.arange(-3, 25))
        assert d['pose_valid'][tick-3:tick+25, selected].all()
        C = np.linalg.inv(poses[tick, anchor])
        # This matrix alone is used for history, observed action and GT effect.
        current = C @ poses[tick, selected]
        history = C @ poses[hist][:, selected]
        targets = C @ poses[future][:, selected]
        effect = targets @ np.linalg.inv(current)[None]
        handhist = transform_points(d['hand'][hist].reshape(4, 22, 3), C)
        handfuture = transform_points(d['hand'][future].reshape(24, 22, 3), C)
        hvalid = np.repeat(d['hand_valid'][tick], 11)
        av = np.repeat(d['hand_valid'][future], 11, axis=-1)
        # Donor chunks carry all their features; current-relative and step deltas
        # must not be recomputed against the recipient H after shuffling.
        act = np.concatenate((handfuture, handfuture - handhist[-1],
                              np.diff(np.concatenate((handhist[-1:], handfuture)), axis=0)), axis=-1)
        act[~av] = 0
        scene_xyz, scene_feat, scene_obj = [], [], []
        pointsets, radius, objfeat = [], [], []
        for slot, objidx in enumerate(selected):
            p, normal, r, center = self.canonical(d['objects'][objidx])
            ph = transform_points(p[None], history[:, slot])
            now = ph[-1]
            n = normal @ current[slot, :3, :3].T
            velocity = (ph[-1] - ph[-2]) * 30
            acceleration = (ph[-1] - 2 * ph[-2] + ph[-3]) * 900
            displacement = ph[-1] - ph[0]
            static = float(np.max(np.linalg.norm(ph[-1] - ph[0], axis=-1)) < .002)
            target = float(objidx == anchor)
            scene_xyz.append(now)
            # normals3 + velocity3 + acceleration3 + history displacement3,
            # target/static/type/handID/keypointID/local-instanceID =6.
            extra = np.tile([target, static, 0, 0, 0, slot / 32], (len(p), 1))
            scene_feat.append(np.concatenate((n, velocity, acceleration, displacement, extra), axis=-1))
            scene_obj.append(np.full(len(p), slot, np.int64))
            pointsets.append(now)
            radius.append(r)
            objfeat.append(np.concatenate((current[slot, :3, :3] @ center + current[slot, :3, 3], current[slot, :3, :3].reshape(-1),
                                           [r, target, slot / 32])))
        hv = (handhist[-1] - handhist[-2]) * 30
        ha = (handhist[-1] - 2 * handhist[-2] + handhist[-3]) * 900
        hd = handhist[-1] - handhist[0]
        extra = np.stack((np.zeros(22), np.zeros(22), np.ones(22), np.repeat([0, 1], 11),
                          np.tile(np.arange(11) / 10, 2), np.zeros(22)), axis=-1)
        hf = np.concatenate((np.zeros((22, 3)), hv, ha, hd, extra), axis=-1)
        scene_xyz.append(handhist[-1][hvalid])
        scene_feat.append(hf[hvalid])
        scene_obj.append(np.full(int(hvalid.sum()), -1, np.int64))
        return dict(xyz=np.concatenate(scene_xyz).astype('float32'), features=np.concatenate(scene_feat).astype('float32'),
                    scene_object=np.concatenate(scene_obj), action=act.astype('float32'), action_valid=av,
                    points=np.stack(pointsets).astype('float32'), radius=np.array(radius, 'float32'),
                    object_features=np.stack(objfeat).astype('float32'),
                    effect=effect.transpose(1, 0, 2, 3).astype('float32'), category=category,
                    sample_id=np.array([s, anchor, tick], dtype=np.int64), hand_presence=d['hand_valid'][tick].copy())


def collate(samples):
    max_points = max(len(s['xyz']) for s in samples)
    max_objects = max(len(s['points']) for s in samples)
    batch = len(samples)
    out = {key: torch.from_numpy(np.stack([s[key] for s in samples])) for key in
           ('action', 'action_valid', 'sample_id', 'hand_presence')}
    for key, shape, dtype in (
        ('xyz', (batch, max_points, 3), torch.float32),
        ('features', (batch, max_points, 18), torch.float32),
        ('points', (batch, max_objects, 512, 3), torch.float32),
        ('radius', (batch, max_objects), torch.float32),
        ('object_features', (batch, max_objects, 15), torch.float32),
        ('effect', (batch, max_objects, 24, 4, 4), torch.float32)):
        out[key] = torch.zeros(shape, dtype=dtype)
        for i, sample in enumerate(samples): out[key][i, :len(sample[key])] = torch.from_numpy(sample[key])
    out['point_valid'] = torch.arange(max_points)[None] < torch.tensor([len(s['xyz']) for s in samples])[:, None]
    out['object_valid'] = torch.arange(max_objects)[None] < torch.tensor([len(s['points']) for s in samples])[:, None]
    out['category'] = torch.tensor([s['category'] for s in samples])
    return out


def balanced_indices(dataset, count, seed):
    rng = np.random.default_rng(seed)
    if any(not len(g) for g in dataset.groups): raise ValueError('empty sampling stratum')
    categories = rng.choice(3, count, p=[.6, .2, .2])
    out = np.empty(count, np.int64)
    for k, pool in enumerate(dataset.groups):
        mask = categories == k
        out[mask] = rng.choice(pool, int(mask.sum()))
    return out


def shuffle_action(batch, seed):
    """Derangement within hand-presence groups; singleton donors are explicitly unavailable."""
    rng = np.random.default_rng(seed)
    presence = batch['hand_presence'].cpu().numpy()
    mapping = np.arange(len(presence))
    available = np.zeros(len(presence), dtype=bool)
    for code in (1, 2, 3):
        ids = np.flatnonzero(presence[:, 0] + 2 * presence[:, 1] == code)
        if len(ids) < 2: continue
        ids = rng.permutation(ids)
        mapping[ids] = np.roll(ids, 1)
        available[ids] = True
    index = torch.as_tensor(mapping, device=batch['action'].device)
    shuffled = dict(batch)
    shuffled['action'] = batch['action'][index]
    shuffled['action_valid'] = batch['action_valid'][index]
    return shuffled, available
