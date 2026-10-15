"""Read the frozen cleaning index, keeping provenance outside model tensors."""
from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .cleaning import CLEANING_SCHEMA
from .windows import PoseMeshWindows, SCHEMA, pose_mesh_sample


@dataclass(frozen=True)
class CleanIdentity:
    manifest: str
    source_manifest: str
    source_id: int
    source_sequence: str
    row: int
    tick: int
    profile: str
    clock_verified: bool
    split: str


@dataclass(frozen=True)
class CleanWindow:
    model_inputs: dict
    provenance: CleanIdentity


class CleanPoseMeshWindows(PoseMeshWindows):
    """Engineering reader: only frozen, filtered windows; no admission override."""
    def __init__(self, root, split, *, engineering=False, verified_clock_only=True, cache_size=8):
        self.root = Path(root).resolve()
        self.meta = json.loads((self.root/'manifest.json').read_text())
        progress = json.loads((self.root/'progress.json').read_text())
        if (self.meta.get('schema') != SCHEMA or self.meta.get('cleaning_schema') != CLEANING_SCHEMA
                or self.meta.get('status') != 'COMPLETED' or self.meta.get('training_allowed') is not False
                or not engineering or self.meta.get('fps') != 30 or self.meta.get('units') != 'm'
                or progress.get('status') != 'COMPLETED'):
            raise ValueError('completed cleaned engineering snapshot required; no training admission')
        if split not in ('train', 'dev') or not 1 <= cache_size <= 32:
            raise ValueError('known split and bounded cache required')
        rows = self.meta['sequences']
        groups = {s: {r['group_id'] for r in rows if r['split'] == s} for s in ('train', 'dev')}
        if groups['train'] & groups['dev']:
            raise ValueError('source group split overlap')
        self.geometry = {}
        for row in self.meta['objects']:
            with np.load(self.root/row['path'], allow_pickle=False) as values:
                geometry = {k: values[k].copy() for k in values.files}
            if row['key'] in self.geometry or not np.allclose(np.linalg.norm(geometry['normals'], axis=1), 1, atol=1e-5):
                raise ValueError('unique cleaned unit-normal geometry required')
            self.geometry[row['key']] = geometry
        key = ('verified_' if verified_clock_only else 'all_') + split
        self._index_stamp = (self.meta['window_index_size_bytes'], self.meta['window_index_mtime_ns'])
        self._check_index()
        with np.load(self.root/self.meta['window_index'], allow_pickle=False) as indices:
            index = indices[key].copy()
        if index.ndim != 2 or index.shape[1] != 2 or index.dtype.kind not in 'iu':
            raise ValueError('invalid frozen window index')
        self.index = [tuple(map(int, r)) for r in index]
        self.cache_size, self.cache = cache_size, OrderedDict()
        self.verified_clock_only = verified_clock_only
        for row_id, tick in self.index:
            if not 0 <= row_id < len(rows) or rows[row_id]['split'] != split or tick < 3:
                raise ValueError('frozen window identity/split mismatch')
            if rows[row_id].get('training_allowed') is not False:
                raise ValueError('cleaned engineering row cannot grant training')
            if verified_clock_only and rows[row_id].get('clock_verified') is not True:
                raise ValueError('unverified source clock in verified index')
        if not self.index:
            raise ValueError('no clean window in requested split/clock subset')

    def _sequence(self, index):
        row = self.meta['sequences'][index]
        path = self.root/row['path']
        stamp = path.stat()
        if (stamp.st_size != row['payload_size_bytes'] or stamp.st_mtime_ns != row['payload_mtime_ns']):
            raise ValueError('source payload changed after cleaning snapshot')
        return super()._sequence(index)

    def _check_index(self):
        stamp = (self.root/self.meta['window_index']).stat()
        if (stamp.st_size, stamp.st_mtime_ns) != self._index_stamp:
            raise ValueError('frozen window index changed after cleaning')

    def __getitem__(self, index):
        self._check_index()
        sequence_id, tick = self.index[index]
        row = self.meta['sequences'][sequence_id]
        sequence = self._sequence(sequence_id)
        if not sequence['object_pose_valid'][tick+1:tick+25, row['target_slot']].all():
            raise ValueError('filtered target horizon no longer valid')
        sample = pose_mesh_sample(sequence, [self.geometry[k] for k in row['objects']], tick,
            row['target_slot'], row['source_id'], [sequence_id, tick], row['profile'])
        record = CleanIdentity(str(self.root/'manifest.json'), row['source_manifest'], row['source_id'],
            row['source_sequence'], sequence_id, tick, row['profile'], row['clock_verified'], row['split'])
        return CleanWindow(sample, record)


def clean_collate(windows, scene_cap=4096):
    """Explicit model inputs + provenance; reject truncation and identity mismatch."""
    if not windows:
        raise ValueError('nonempty clean batch required')
    for window in windows:
        if not isinstance(window, CleanWindow):
            raise ValueError('bound clean windows required')
        sample, record = window.model_inputs, window.provenance
        if (int(sample['source_id']) != record.source_id
                or not np.array_equal(sample['sample_id'], [record.row, record.tick])
                or str(sample['source_profile']) != record.profile
                or bool(sample['training_allowed']) or len(sample['xyz']) > scene_cap):
            raise ValueError('clean batch identity/admission/truncation mismatch')
    from ptv3_shared_wam.egodex_data import forecasting_collate
    tensors = forecasting_collate([w.model_inputs for w in windows], scene_cap=scene_cap)
    return dict(model_inputs=tensors, provenance=tuple(w.provenance for w in windows), training_allowed=False)
