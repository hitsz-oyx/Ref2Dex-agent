"""One current anchor, native masks, shared CAD; never store overlapping labels."""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA = 'ref2dex.pose-mesh.v1'


def xyzquat_to_matrix(values):
    values = np.asarray(values)
    if values.shape[-1] != 7 or not np.isfinite(values).all():
        raise ValueError('finite xyz/xyzw pose required')
    if not np.allclose(np.linalg.norm(values[..., 3:], axis=-1), 1, atol=1e-4):
        raise ValueError('unit pose quaternion required')
    out = np.broadcast_to(np.eye(4), (*values.shape[:-1], 4, 4)).copy()
    out[..., :3, :3] = Rotation.from_quat(values[..., 3:].reshape(-1, 4)).as_matrix().reshape(*values.shape[:-1], 3, 3)
    out[..., :3, 3] = values[..., :3]
    return out


def validate_sequence(sequence):
    t = np.asarray(sequence['time'])
    hand, valid = sequence['hand'], sequence['hand_valid']
    poses, known = sequence['object_poses'], sequence['object_pose_valid']
    if t.ndim != 1 or len(t) < 28 or not np.isfinite(t).all() or not np.allclose(np.diff(t), 1/30, atol=1e-6, rtol=0):
        raise ValueError('explicit contiguous30Hz sequence required; resample in the native reader')
    if hand.shape != (len(t), 2, 11, 3) or valid.shape != hand.shape[:-1] or valid.dtype.kind != 'b':
        raise ValueError('right/left native11-point hand and masks required')
    if poses.ndim != 4 or poses.shape[0] != len(t) or poses.shape[-2:] != (4, 4) or not 1 <= poses.shape[1] <= 6:
        raise ValueError('one to six rigid object trajectories required')
    if known.shape != poses.shape[:2] or known.dtype.kind != 'b' or not np.isfinite(hand[valid]).all():
        raise ValueError('finite known hand/object data required')
    p = poses[known]
    if not np.isfinite(p).all() or not np.allclose(p[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError('known homogeneous object poses required')
    r = p[:, :3, :3]
    if not np.allclose(r.swapaxes(-1, -2) @ r, np.eye(3), atol=2e-4) or not np.allclose(np.linalg.det(r), 1, atol=2e-4):
        raise ValueError('proper world-from-object rotations required')
    if np.shape(sequence['segment']) != (len(t),):
        raise ValueError('explicit episode/reset segments required')


def pose_mesh_sample(sequence, geometries, tick, target_slot, source_id, sample_id, profile):
    """All current/H4/future geometry uses T_world_target(t)^-1 in metres."""
    if not 3 <= tick < len(sequence['time'])-24:
        raise ValueError('H4/K24 outside sequence')
    window = slice(tick-3, tick+25)
    segments = sequence['segment'][window]
    if (segments != segments[0]).any():
        raise ValueError('window crosses reset/gap segment')
    poses, known = sequence['object_poses'], sequence['object_pose_valid']
    active = np.flatnonzero(known[tick-3:tick+1].all(0))
    if target_slot not in active or not known[tick+24, target_slot]:
        raise ValueError('observed target and endpoint pose must be known')
    anchor = np.linalg.inv(poses[tick, target_slot])
    rotation, translation = anchor[:3, :3], anchor[:3, 3]
    valid = sequence['hand_valid'][window]
    raw_hand = np.where(valid[..., None], sequence['hand'][window], 0.)
    hand = raw_hand @ rotation.T + translation
    present = valid[:4].all(0)
    if not present.any():
        raise ValueError('no observed native hand point')
    history = np.concatenate(((hand[3]-hand[2])*30, (hand[3]-2*hand[2]+hand[1])*900, hand[3]-hand[0]), -1)
    future_hand_valid = valid[4:] & present[None]
    xyz, features, scene_object, starts, targets, masks = [], [], [], [], [], []
    for slot, object_id in enumerate(active):
        geometry = geometries[object_id]
        surface, normals = geometry['surface'], geometry['normals']
        if (surface.ndim != 2 or surface.shape[1] != 3 or normals.shape != surface.shape
                or not 32 <= len(surface) <= 512 or not np.isfinite(surface).all() or not np.isfinite(normals).all()):
            raise ValueError('finite32..512 CAD points and same-order normals required')
        ids = np.linspace(0, len(surface)-1, 32, dtype=int)
        query = geometry.get('query', surface[ids])
        if query.shape != (32, 3) or len(np.unique(query, axis=0)) != 32 or not np.isfinite(query).all():
            raise ValueError('32 unique fixed CAD query identities required')
        observed = anchor @ poses[tick-3:tick+1, object_id]
        cloud = np.einsum('tij,pj->tpi', observed[:, :3, :3], surface) + observed[:, None, :3, 3]
        f = np.zeros((len(surface), 18), np.float32)
        f[:, :3] = normals @ observed[-1, :3, :3].T
        f[:, 3:6] = (cloud[3]-cloud[2])*30
        f[:, 6:9] = (cloud[3]-2*cloud[2]+cloud[1])*900
        f[:, 9:12] = cloud[3]-cloud[0]
        f[:, 12] = object_id == target_slot
        start = query @ observed[-1, :3, :3].T + observed[-1, :3, 3]
        future_known = known[tick+1:tick+25, object_id]
        safe = np.where(future_known[:, None, None], poses[tick+1:tick+25, object_id], np.eye(4))
        relative = anchor @ safe
        future = np.einsum('tij,pj->tpi', relative[:, :3, :3], query) + relative[:, None, :3, 3]
        xyz.append(cloud[-1]); features.append(f); scene_object.append(np.full(len(surface), slot, np.int64))
        starts.append(start); targets.append(np.where(future_known[:, None, None], future-start, 0.))
        masks.append(np.broadcast_to(future_known[:, None], (24, 32)).copy())
    target = int(np.flatnonzero(active == target_slot)[0])
    starts, targets, masks = np.asarray(starts, np.float32), np.asarray(targets, np.float32), np.asarray(masks)
    return dict(xyz=np.concatenate(xyz).astype(np.float32), features=np.concatenate(features),
        scene_object=np.concatenate(scene_object), object_valid=np.ones(len(active), bool),
        scene_points=starts, scene_target=targets, scene_valid=masks,
        start_hand=np.where(present[..., None], hand[3], 0.).astype(np.float32),
        start_hand_history=np.where(present[..., None], history, 0.).astype(np.float32), start_hand_valid=present,
        trajectory_target=np.where(future_hand_valid[..., None], hand[4:]-hand[3], 0.).astype(np.float32),
        trajectory_valid=future_hand_valid, start_object_points=starts[target].copy(),
        goal_object_points=starts[target]+targets[target, -1], object_goal_valid=masks[target, -1].copy(),
        goal_object_slot=np.int64(target), source_id=np.int64(source_id), sample_id=np.asarray(sample_id, np.int64),
        anchor_world_to_sample=anchor, source_profile=np.array(profile), training_allowed=np.array(False))


class PoseMeshWindows:
    """Explicit manifest, TRAIN/dev group isolation, bounded sequence cache."""
    def __init__(self, root, split, engineering=False, stride=8, cache_size=8):
        self.root = Path(root).resolve()
        self.meta = json.loads((self.root/'manifest.json').read_text())
        if self.meta.get('schema') != SCHEMA or self.meta.get('units') != 'm' or self.meta.get('fps') != 30:
            raise ValueError('native3D metre30Hz manifest required')
        if not engineering or self.meta.get('training_allowed') is not False:
            raise ValueError('provisional geometry corpus requires engineering=True; not training admission')
        if split not in ('train', 'dev') or not 1 <= stride <= 24 or not 1 <= cache_size <= 32:
            raise ValueError('bounded split/stride/cache required')
        rows = self.meta['sequences']
        groups = {s: {r['group_id'] for r in rows if r['split'] == s} for s in ('train', 'dev')}
        if not groups['train'] or not groups['dev'] or groups['train'] & groups['dev']:
            raise ValueError('nonempty TRAIN/dev with disjoint source groups required')
        self.geometry = {}
        for row in self.meta['objects']:
            if row['key'] in self.geometry:
                raise ValueError('duplicate mesh-scale geometry key')
            with np.load(self.root/row['path'], allow_pickle=False) as data:
                self.geometry[row['key']] = {k: data[k].copy() for k in data.files}
        self.cache_size, self.cache, self.index = cache_size, OrderedDict(), []
        for i, row in enumerate(rows):
            if row['split'] != split:
                continue
            sequence = self._sequence(i)
            target = row['target_slot']
            if len(row['objects']) != sequence['object_poses'].shape[1] or not 0 <= target < len(row['objects']):
                raise ValueError('sequence object identities/target mismatch')
            for tick in range(3, len(sequence['time'])-24, stride):
                if ((sequence['segment'][tick-3:tick+25] == sequence['segment'][tick]).all()
                    and sequence['object_pose_valid'][tick-3:tick+1, target].all()
                    and sequence['object_pose_valid'][tick+24, target]
                    and sequence['hand_valid'][tick-3:tick+1].all(0).any()):
                    self.index.append((i, tick))
        self.cache.clear()
        if not self.index:
            raise ValueError('no complete known-target forecasting window in split')

    def _sequence(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        row = self.meta['sequences'][index]
        with np.load(self.root/row['path'], allow_pickle=False) as data:
            sequence = {k: data[k].copy() for k in data.files}
        if row['format'] == 'unidexgrasp-compact':
            if (str(sequence['object_code']) != row['object_code'] or float(sequence['scale']) != row['scale']
                    or str(sequence['split']) != row['split'] or sequence['reset'].any()):
                raise ValueError('actual UniDex identity/reset mismatch')
            t = len(sequence['time'])
            hand = np.zeros((t, 2, 11, 3), np.float32); hand[:, 0] = sequence['hand']
            mask = np.zeros((t, 2, 11), bool); mask[:, 0] = True
            sequence = dict(time=sequence['time'], hand=hand, hand_valid=mask,
                object_poses=xyzquat_to_matrix(sequence['object'])[:, None],
                object_pose_valid=np.ones((t, 1), bool), segment=np.zeros(t, np.int64))
        elif row['format'] != 'canonical':
            raise ValueError('unknown native reader format')
        validate_sequence(sequence)
        self.cache[index] = sequence
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return sequence

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        sequence_id, tick = self.index[index]
        row = self.meta['sequences'][sequence_id]
        return pose_mesh_sample(self._sequence(sequence_id), [self.geometry[k] for k in row['objects']],
            tick, row['target_slot'], row['source_id'], [sequence_id, tick], row['profile'])
