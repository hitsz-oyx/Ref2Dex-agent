"""History-only weak video points; isolated from the rigid native data contract."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

HISTORY, HORIZON = 4, 24


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
                elapsed=elapsed, selected_track_ids=selected)


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
        return sample_window(a['points'], a['valid'], a['kind'], a['timestamps'], start,
                             self.manifest['protocol']['points_per_kind'])


def collate_video(samples):
    b, n = len(samples), max(len(s['xyz']) for s in samples)
    shapes = dict(xyz=(b, n, 3), features=(b, n, 18), velocity=(b, n, 3),
                  target_flow=(b, n, HORIZON, 3), target_valid=(b, n, HORIZON),
                  point_kind=(b, n), point_valid=(b, n))
    out = {k: torch.zeros(shape, dtype=torch.bool if k.endswith('valid') else
                           torch.long if k == 'point_kind' else torch.float32)
           for k, shape in shapes.items()}
    for i, sample in enumerate(samples):
        count = len(sample['xyz'])
        for key in shapes:
            if key != 'point_valid':
                out[key][i, :count] = torch.from_numpy(sample[key])
        out['point_valid'][i, :count] = True
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


def flow_metrics(prediction, batch):
    error = torch.linalg.vector_norm(prediction - batch['target_flow'], dim=-1)
    result = {}
    for group, name in ((0, 'background'), (1, 'object')):
        mask = batch['target_valid'] & batch['point_valid'][..., None] & (batch['point_kind'][..., None] == group)
        for horizon, suffix in ((None, 'all'), (23, 'h24')):
            selected = mask if horizon is None else mask[..., horizon]
            values = error if horizon is None else error[..., horizon]
            result[name + '/' + suffix] = (float(values[selected].sum()), int(selected.sum()))
    return result
