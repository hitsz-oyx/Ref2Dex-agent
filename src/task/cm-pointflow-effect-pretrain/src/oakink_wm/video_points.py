"""History-only weak video points; isolated from the rigid native data contract."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

HISTORY, HORIZON = 4, 24
MOTION_INTERVALS = HISTORY - 1
MOTION_FEATURES = 15


def interval_motion_stats(history_points, kinds, timestamps):
    """Compute causal metric-3D interval tokens for object/background groups.

    The 15 features are mean displacement xyz, mean/p99 displacement magnitude,
    elapsed time, mean velocity xyz, mean/p99 speed, and mean acceleration xyz
    plus mean acceleration magnitude.  All values use only the four observed
    history frames; the first acceleration uses the next observed interval.
    """
    points = np.asarray(history_points, dtype=np.float32)
    kinds = np.asarray(kinds)
    stamps = np.asarray(timestamps, dtype=np.float32)
    if points.ndim != 3 or points.shape[0] != HISTORY or points.shape[-1] != 3:
        raise ValueError('history_points must have shape (HISTORY, points, 3)')
    if kinds.shape != (points.shape[1],):
        raise ValueError('kinds must match history point count')
    dt = np.diff(stamps)
    if dt.shape != (MOTION_INTERVALS,) or not np.isfinite(dt).all() or (dt <= 0).any():
        raise ValueError('history timestamps must have positive finite intervals')
    displacement = points[1:] - points[:-1]
    velocity = displacement / dt[:, None, None]
    acceleration = np.empty_like(velocity)
    acceleration[0] = (velocity[1] - velocity[0]) / ((dt[1] + dt[0]) / 2)
    for interval in range(1, MOTION_INTERVALS):
        acceleration[interval] = (velocity[interval] - velocity[interval - 1]) / (
            (dt[interval] + dt[interval - 1]) / 2
        )
    result = np.zeros((MOTION_INTERVALS, 2, MOTION_FEATURES), dtype=np.float32)
    valid = np.zeros((MOTION_INTERVALS, 2), dtype=bool)
    for interval in range(MOTION_INTERVALS):
        displacement_mag = np.linalg.norm(displacement[interval], axis=-1)
        speed = np.linalg.norm(velocity[interval], axis=-1)
        acceleration_mag = np.linalg.norm(acceleration[interval], axis=-1)
        for group in (0, 1):
            selected = kinds == group
            if not selected.any():
                continue
            valid[interval, group] = True
            result[interval, group] = np.concatenate((
                displacement[interval, selected].mean(0),
                np.asarray((
                    float(displacement_mag[selected].mean()),
                    float(np.percentile(displacement_mag[selected], 99)),
                    float(dt[interval]),
                ), dtype=np.float32),
                velocity[interval, selected].mean(0),
                np.asarray((
                    float(speed[selected].mean()),
                    float(np.percentile(speed[selected], 99)),
                ), dtype=np.float32),
                acceleration[interval, selected].mean(0),
                np.asarray((float(acceleration_mag[selected].mean()),), dtype=np.float32),
            ))
    if not np.isfinite(result).all():
        raise ValueError('history motion statistics are nonfinite')
    return result, valid


def sample_window(points, valid, kind, timestamps, start, quota=128):
    """Point identity/selection and features depend on HISTORY exclusively."""
    history = slice(start, start + HISTORY)
    future = slice(start + HISTORY, start + HISTORY + HORIZON)
    if start < 0 or start + HISTORY + HORIZON > len(points):
        raise ValueError('window outside source support')
    if not (np.diff(timestamps) > 0).all():
        raise ValueError('timestamps must strictly increase')
    eligible = valid[history].all(0) & np.isfinite(points[history]).all((0, 2))
    selected = []
    for group in (1, 0):
        pool = np.flatnonzero(eligible & (kind == group))
        if group == 1 and len(pool) < 16:
            raise ValueError('fewer than16 history-valid object tracks')
        if len(pool) > quota:
            pool = pool[np.linspace(0, len(pool) - 1, quota, dtype=int)]
        selected.extend(pool.tolist())
    selected = np.asarray(selected, dtype=np.int64)
    ph = points[history][:, selected].astype('float32')
    kinds = kind[selected]
    motion_stats, motion_valid = interval_motion_stats(ph, kinds, timestamps[history])
    center = ph[-1, kinds == 1].mean(0)
    ph -= center
    dt = np.diff(timestamps[history]).astype('float32')
    v = (ph[-1] - ph[-2]) / dt[-1]
    before = (ph[-2] - ph[-3]) / dt[-2]
    acceleration = (v - before) / ((dt[-1] + dt[-2]) / 2)
    displacement = ph[-1] - ph[0]
    static = (np.linalg.norm(displacement, axis=-1) < .002).astype('float32')
    extra = np.stack((kinds, static, np.zeros(len(kinds)), np.zeros(len(kinds)),
                      np.zeros(len(kinds)), np.zeros(len(kinds))), axis=-1)
    features = np.concatenate((np.zeros_like(v), v, acceleration, displacement, extra), axis=-1)
    target_valid = valid[future][:, selected].T.copy()
    pf = points[future][:, selected].transpose(1, 0, 2)
    target_valid &= np.isfinite(pf).all(-1)
    # Exclude invalid labels BEFORE arithmetic, never multiply NaN by zero.
    flow = np.zeros_like(pf, dtype='float32')
    now_world = points[start + HISTORY - 1, selected]
    rows, steps = np.nonzero(target_valid)
    flow[rows, steps] = pf[rows, steps] - now_world[rows]
    elapsed = (timestamps[future] - timestamps[start + HISTORY - 1]).astype('float32')
    return dict(xyz=ph[-1], features=features.astype('float32'), velocity=v,
                target_flow=flow, target_valid=target_valid, point_kind=kinds.astype('int64'),
                elapsed=elapsed, selected_track_ids=selected,
                motion_stats=motion_stats, motion_valid=motion_valid)


class VideoWindows(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest['status'] != 'QUALIFIED_VIDEO_PROBE_ONLY':
            raise ValueError('video corpus is not qualified for the declared Probe')
        self.sequences, self.index = {}, []
        for entry in self.manifest['sequences']:
            if entry['split'] != split:
                continue
            path = self.root / entry['file']
            with np.load(path, allow_pickle=False) as z:
                arrays = {key: z[key].copy() for key in ('points', 'valid', 'kind', 'timestamps')}
            self.sequences[entry['scene']] = arrays
            for start in entry['window_starts']:
                self.index.append((entry['scene'], start))
        if not self.index:
            raise ValueError('empty video split ' + split)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        scene, start = self.index[index]
        a = self.sequences[scene]
        if a['points'].ndim == 4:
            # Each block births identities at its own HISTORY start. No identity
            # continuity is asserted between overlapping blocks.
            sample = sample_window(a['points'][start], a['valid'][start],
                                   a['kind'][start], a['timestamps'][start], 0,
                                   self.manifest['protocol']['points_per_kind'])
        else:
            sample = sample_window(a['points'], a['valid'], a['kind'], a['timestamps'], start,
                                   self.manifest['protocol']['points_per_kind'])
        horizon = self.manifest['protocol'].get('supervised_horizon', HORIZON)
        if not 1 <= horizon <= HORIZON:
            raise ValueError('unsupported video supervision horizon')
        sample['target_valid'][:, horizon:] = False
        return sample


def collate_video(samples):
    b, n = len(samples), max(len(s['xyz']) for s in samples)
    point_shapes = dict(xyz=(b, n, 3), features=(b, n, 18), velocity=(b, n, 3),
                        target_flow=(b, n, HORIZON, 3), target_valid=(b, n, HORIZON),
                        point_kind=(b, n), point_valid=(b, n))
    out = {k: torch.zeros(shape, dtype=torch.bool if k.endswith('valid') else
                           torch.long if k == 'point_kind' else torch.float32)
           for k, shape in point_shapes.items()}
    out['motion_stats'] = torch.zeros((b, MOTION_INTERVALS, 2, MOTION_FEATURES),
                                      dtype=torch.float32)
    out['motion_valid'] = torch.zeros((b, MOTION_INTERVALS, 2), dtype=torch.bool)
    for i, sample in enumerate(samples):
        count = len(sample['xyz'])
        for key in point_shapes:
            if key != 'point_valid':
                out[key][i, :count] = torch.from_numpy(sample[key])
        out['point_valid'][i, :count] = True
        out['motion_stats'][i] = torch.from_numpy(sample['motion_stats'])
        out['motion_valid'][i] = torch.from_numpy(sample['motion_valid'])
    out['elapsed'] = torch.from_numpy(np.stack([s['elapsed'] for s in samples]))
    return out


def balanced_flow_loss(prediction, batch, scale=.01):
    """Average supported groups/windows; padding and invalid targets are absent."""
    terms = []
    for b in range(len(prediction)):
        groups = []
        for group in (0, 1):
            mask = (batch['target_valid'][b] & batch['point_valid'][b, :, None]
                    & (batch['point_kind'][b, :, None] == group))
            if mask.any():
                groups.append(torch.nn.functional.huber_loss(prediction[b][mask] / scale,
                              batch['target_flow'][b][mask] / scale, reduction='mean'))
        if groups:
            terms.append(torch.stack(groups).mean())
    if not terms:
        raise ValueError('batch has no supported video targets')
    return torch.stack(terms).mean()


def flow_metrics(prediction, batch, horizon=HORIZON):
    error = torch.linalg.vector_norm(prediction - batch['target_flow'], dim=-1)
    result = {}
    for group, name in ((0, 'background'), (1, 'object')):
        mask = batch['target_valid'] & batch['point_valid'][..., None] & (batch['point_kind'][..., None] == group)
        for step, suffix in ((None, 'all'), (horizon - 1, 'h%d' % horizon)):
            selected = mask[..., :horizon] if step is None else mask[..., step]
            values = error[..., :horizon] if step is None else error[..., step]
            result[name + '/' + suffix] = (float(values[selected].sum()), int(selected.sum()))
    return result
