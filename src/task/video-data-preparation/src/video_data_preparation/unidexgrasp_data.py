"""On-demand geometry forecasting from explicit ShadowHand/CAD trajectories.

Source 4 is provisional UniDexGrasp geometry, distinct from human sources 0..2
and the existing stage-specific source 3. No native robot/action batch is emitted.
"""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PROFILE = 'unidexgrasp-shadow-complete-cad-provisional'


def object_poses(values):
    values = np.asarray(values)
    if values.shape[-1] != 7 or not np.isfinite(values).all():
        raise ValueError('finite xyz/xyzw object poses required')
    if np.max(abs(np.linalg.norm(values[..., 3:7], axis=-1)-1)) > 1e-4:
        raise ValueError('recorded object quaternion is not unit length')
    out = np.tile(np.eye(4), (*values.shape[:-1], 1, 1))
    out[..., :3, :3] = Rotation.from_quat(values[..., 3:7].reshape(-1, 4)).as_matrix().reshape(*values.shape[:-1], 3, 3)
    out[..., :3, 3] = values[..., :3]
    return out


def core_window(hand, obj, surface, tick):
    if not 3 <= tick < len(hand)-24 or len(obj) != len(hand):
        raise ValueError('continuous H4/K24 window outside trajectory')
    anchor = np.linalg.inv(obj[tick])
    hh = (hand[tick-3:tick+25]-obj[tick, :3, 3]) @ obj[tick, :3, :3]
    start = np.zeros((2, 11, 3), np.float32); start[0] = hh[3]
    valid = np.zeros((2, 11), bool); valid[0] = True
    history = np.zeros((2, 11, 9), np.float32)
    history[0] = np.concatenate(((hh[3]-hh[2])*30, (hh[3]-2*hh[2]+hh[1])*900, hh[3]-hh[0]), -1)
    future = np.zeros((24, 2, 11, 3), np.float32); future[:, 0] = hh[4:]-hh[3]
    rel = anchor @ obj[tick+np.arange(1, 25)]
    query = surface[np.linspace(0, len(surface)-1, 32, dtype=int)].copy()
    moving = np.einsum('tij,pj->tpi', rel[:, :3, :3], query)+rel[:, None, :3, 3]
    target = (moving-query).astype(np.float32)
    return dict(start_hand=start, start_hand_history=history, start_hand_valid=valid,
                trajectory_target=future, trajectory_valid=np.broadcast_to(valid, (24, 2, 11)).copy(),
                scene_points=query[None].astype(np.float32), scene_target=target[None],
                scene_valid=np.ones((1, 24, 32), bool), start_object_points=query.astype(np.float32),
                goal_object_points=moving[-1].astype(np.float32), object_goal_valid=np.ones(32, bool),
                current_cad_points=surface.astype(np.float32), anchor_world_to_current_object=anchor,
                source_profile=np.array(PROFILE), training_allowed=np.array(False))


def forecasting_sample(trajectory, geometry, tick, trajectory_id):
    """Current scene normals and motion use only the four observed poses."""
    surface = geometry['surface']
    obj = object_poses(trajectory['object'])
    sample = core_window(trajectory['hand'], obj, surface, tick)
    history = obj[tick-3:tick+1]
    world = np.einsum('tij,pj->tpi', history[:, :3, :3], surface)+history[:, None, :3, 3]
    observed = (world-obj[tick, :3, 3]) @ obj[tick, :3, :3]
    features = np.zeros((len(surface), 18), np.float32)
    features[:, :3] = geometry['normals']
    features[:, 3:6] = (observed[3]-observed[2])*30
    features[:, 6:9] = (observed[3]-2*observed[2]+observed[1])*900
    features[:, 9:12] = observed[3]-observed[0]
    features[:, 12] = 1  # Entire current complete CAD is the target rigid object.
    sample.update(xyz=surface.copy(), features=features, scene_object=np.zeros(len(surface), np.int64),
                  object_valid=np.ones(1, bool), goal_object_slot=np.int64(0), source_id=np.int64(4),
                  sample_id=np.array([trajectory_id, tick], np.int64))
    return sample


class UniDexGraspWindows:
    """Bounded LRU trajectory loading; duplicated temporal windows are never saved.

    Engineering access must be explicit until visual/source admission is complete.
    Success and failure trajectories remain available; filtering is not implicit.
    """
    def __init__(self, root, split, engineering=False, stride=8, cache_size=8):
        self.root = Path(root).resolve()
        self.meta = json.loads((self.root/'trajectory-manifest.json').read_text())
        if self.meta['profile'] != PROFILE:
            raise ValueError('explicit ShadowHand/CAD source profile required')
        if not engineering or self.meta.get('training_allowed', False):
            raise ValueError('this provisional provider requires engineering=True and is not a training admission')
        if split not in ('train', 'dev') or not 1 <= stride <= 24 or not 1 <= cache_size <= 32:
            raise ValueError('bounded split/stride/cache required')
        objects = self.meta['objects']
        train = {o['object_code'] for o in objects if o['split'] == 'train'}
        dev = {o['object_code'] for o in objects if o['split'] == 'dev'}
        if not train or not dev or train & dev:
            raise ValueError('nonempty object-disjoint train/dev required')
        self.geometry = {}
        for i, row in enumerate(objects):
            with np.load(self.root/'geometry'/('object-%03d.npz' % i), allow_pickle=False) as data:
                values = {key: data[key].copy() for key in data.files}
            if str(values['object_code']) != row['object_code'] or float(values['scale']) != row['scale']:
                raise ValueError('shared CAD identity mismatch')
            self.geometry[row['object_code']] = values
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.index = []
        identities = {o['object_code']: o for o in objects}
        for i, row in enumerate(self.meta['trajectories']):
            spec = identities[row['object_code']]
            if row['split'] != spec['split'] or row['scale'] != spec['scale']:
                raise ValueError('trajectory source identity/split mismatch')
            if row['split'] != split:
                continue
            trajectory = self._trajectory(i)
            self.index.extend((i, tick) for tick in range(3, len(trajectory['time'])-24, stride))
        self.cache.clear()

    def _trajectory(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        row = self.meta['trajectories'][index]
        path = self.root/row['path']
        if path.resolve().parent != self.root/'trajectories':
            raise ValueError('trajectory escaped immutable corpus')
        with np.load(path, allow_pickle=False) as data:
            values = {key: data[key].copy() for key in data.files}
        if str(values['object_code']) != row['object_code'] or str(values['split']) != row['split'] or float(values['scale']) != row['scale']:
            raise ValueError('recorded trajectory identity mismatch')
        if values['hand'].shape != (len(values['time']), 11, 3) or not np.allclose(np.diff(values['time']), 1/30, atol=1e-7):
            raise ValueError('actual contiguous 30Hz hand trajectory required')
        if values['reset'].any() or not np.isfinite(values['hand']).all():
            raise ValueError('internal reset/nonfinite hand in trajectory')
        self.cache[index] = values
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return values

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        trajectory_id, tick = self.index[index]
        trajectory = self._trajectory(trajectory_id)
        return forecasting_sample(trajectory, self.geometry[str(trajectory['object_code'])], tick, trajectory_id)
