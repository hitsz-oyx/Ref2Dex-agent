"""Frozen controls for a fixed-compute small/full native-data comparison."""
import json
from pathlib import Path

import numpy as np

from .data import ShadowWindows

SCHEMA = 'ref2dex.shadow-data-scale.v1'
STARTUP_SCHEMA = 'ref2dex.shadow-startup-sampling.v1'
SHAPES = dict(state_mean=(300,), state_scale=(300,), action_mean=(8, 24), action_scale=(8, 24))


def stamp(path):
    s = Path(path).stat()
    return [s.st_size, s.st_mtime_ns]


def trajectory_key(row):
    return str(Path(row['raw_episode']).resolve()), int(row['env'])


def window_key(dataset, index):
    record, tick = dataset.keys[int(index)]
    return [*trajectory_key(dataset.records[record]), int(tick)]


def verify_nested(small, full):
    """Match physical identities and saved tensors, independently of row ordering."""
    mapping = {}
    for row in full['records']:
        key = trajectory_key(row)
        if key in mapping:
            raise ValueError('duplicate full trajectory identity')
        mapping[key] = row
    for row in small['records']:
        other = mapping.get(trajectory_key(row))
        if other is None:
            raise ValueError('small trajectory absent from full data')
        for field in ('object_code', 'scale', 'split'):
            if row[field] != other[field]:
                raise ValueError('small/full identity mismatch: '+field)
        if Path(row['geometry']).resolve() != Path(other['geometry']).resolve():
            raise ValueError('small/full CAD mismatch')
        if not set(row['ticks']).issubset(other['ticks']):
            raise ValueError('small windows absent from full data')
        with np.load(Path(small['_root'])/row['path'], allow_pickle=False) as a:
            with np.load(Path(full['_root'])/other['path'], allow_pickle=False) as b:
                if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                    raise ValueError('small/full cached state or label differs')


def normalizer_arrays(values):
    if set(values) != set(SHAPES):
        raise ValueError('complete state/action normalizer required')
    result = {key: np.asarray(values[key], dtype=np.float32) for key in SHAPES}
    for key, array in result.items():
        if array.shape != SHAPES[key] or not np.isfinite(array).all():
            raise ValueError('invalid normalizer: '+key)
        floor = .05 if key == 'state_scale' else .01
        if key.endswith('_scale') and np.any(array < floor):
            raise ValueError('normalizer below declared floor: '+key)
    return result


def load_controls(path, train_cache, parent, seed, updates, batch_size, startup_fraction=0.):
    c = json.loads(Path(path).read_text())
    if c.get('schema') not in (SCHEMA,STARTUP_SCHEMA) or c.get('status') != 'PREPARED':
        raise ValueError('prepared scale controls required')
    expected_fraction=c.get('startup_fraction',0.) if c['schema']==STARTUP_SCHEMA else 0.
    if startup_fraction!=expected_fraction:
        raise ValueError('comparison startup sampling changed')
    if str(Path(train_cache).resolve()) not in c['training_caches'].values():
        raise ValueError('unregistered training cache')
    if c['settings'] != dict(parent=str(Path(parent).resolve()), seed=seed,
                             updates=updates, batch_size=batch_size):
        raise ValueError('scale comparison settings changed')
    for record in c['input_stamps']:
        if stamp(record['path']) != record['stamp']:
            raise ValueError('comparison input changed: '+record['path'])
    dev = ShadowWindows(c['validation_cache'], 'dev')
    validation = np.asarray(c['validation_ids'], dtype=np.int64)
    if len(validation) != len(set(validation)) or not len(validation):
        raise ValueError('unique nonempty shared DEV panel required')
    if (validation < 0).any() or (validation >= len(dev)).any():
        raise ValueError('shared DEV indices out of range')
    if [window_key(dev, i) for i in validation] != c['validation_keys']:
        raise ValueError('shared DEV physical identities changed')
    return c, dev, validation, normalizer_arrays(c['normalization'])
