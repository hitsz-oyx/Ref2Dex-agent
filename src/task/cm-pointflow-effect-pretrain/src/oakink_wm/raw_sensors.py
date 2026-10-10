"""Original byte sensor targets and relative hand shape, without world claims."""
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

HISTORY, HORIZON = 4, 24


def sensor_window(sensors, joints, timestamps, start):
    end = start + HISTORY + HORIZON
    if start < 0 or end > len(sensors):
        raise ValueError('window outside original source')
    raw = sensors[start:end]
    hand = joints[start:end]
    ts = timestamps[start:end]
    if raw.shape != (28, 512) or hand.shape != (28, 2, 21, 3):
        raise ValueError('original raw/hand shape differs')
    if not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 255:
        raise ValueError('raw sensor byte range differs')
    if not np.isfinite(hand).all() or not np.all(np.diff(ts) > 0):
        raise ValueError('invalid hand/clock')
    history_span = np.linalg.norm(hand[:4, :, 9] - hand[:4, :, 0], axis=-1)
    if (history_span <= 1e-6).any():
        raise ValueError('degenerate observed hand shape')
    scale = np.median(history_span, axis=0)
    # Each frame's root-relative shape; scale is exclusively observed HISTORY.
    relative = (hand - hand[:, :, :1]) / scale[None, :, None, None]
    normalized = raw.astype('float32') / 255.
    return dict(sensor_history=normalized[:4].copy(), hand_history=relative[:4].astype('float32'),
                future_hand_shape=relative[4:].astype('float32'),
                sensor_current=normalized[3].copy(), sensor_target=normalized[4:].copy(),
                target_delta=(normalized[4:] - normalized[3]).copy(),
                elapsed=(ts[4:] - ts[3]).astype('float32'))


class SensorWindows(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest['status'] != 'QUALIFIED_RAW_SENSOR_PROBE_ONLY':
            raise ValueError('raw sensor corpus not qualified')
        self.records, self.index = {}, []
        for entry in self.manifest['records']:
            if entry['split'] != split:
                continue
            with np.load(self.root / entry['file'], allow_pickle=False) as z:
                self.records[entry['record']] = {k: z[k].copy() for k in ('sensors', 'joints', 'timestamps')}
            self.index.extend((entry['record'], start) for start in entry['window_starts'])
        if not self.index:
            raise ValueError('empty raw sensor split')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        name, start = self.index[idx]
        a = self.records[name]
        return sensor_window(a['sensors'], a['joints'], a['timestamps'], start)
