"""Read the complete preparation census without granting training admission.

Sequence IDs in this census are plan IDs, not list offsets. The existing v6
training recipe validator intentionally continues to reject this data-only
manifest. This reader shares the real sample_window and collate contracts and
always constructs the owner-admitted unknown goal/event inputs.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from . import paths
from .events import UNKNOWN_EVENT_POLICY, apply_event_input_policy
from .goal_policy import UNKNOWN_GOAL_POLICY, apply_goal_input_policy
from .geometry_gt_object import dense_object_future_gt
from ptv3_shared_wam_v5.cache_contract import metadata_identity
from ptv3_shared_wam_v5.data import sample_window, collate


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


class PreparedWindows(Dataset):
    """Complete mmap index, strict binding, bounded caches, no optimizer sampler."""

    def __init__(self, manifest, split, geometry_manifest, expected_sha256,
                 geometry_gt_object=True, cache_size=2):
        if split not in ('train', 'dev') or not 1 <= cache_size <= 8:
            raise ValueError('known split and bounded cache required')
        self.path = Path(manifest).resolve()
        self.geometry_path = Path(geometry_manifest).resolve()
        self.expected_sha256 = expected_sha256
        if sha256(self.path) != expected_sha256:
            raise ValueError('preparation manifest changed')
        self.manifest = read(self.path)
        if self.manifest.get('schema') != 'ref2dex.nine-source-readiness.v1':
            raise ValueError('complete preparation census required')
        self.split, self.cache_size = split, cache_size
        self.geometry_gt_object = geometry_gt_object
        for name in ('sequences.jsonl', split + '-all.npy', split + '-clock-valid.npy'):
            expected = self.manifest['files'][name]
            file = self.path.parent / name
            if file.stat().st_size != expected['bytes'] or sha256(file) != expected['sha256']:
                raise ValueError('bound preparation file changed: ' + name)
        with (self.path.parent / 'sequences.jsonl').open() as stream:
            rows = [json.loads(line) for line in stream]
        self.rows = {r['sequence_id']: r for r in rows}
        if len(self.rows) != len(rows) or len(rows) != self.manifest['expected_mothers']:
            raise ValueError('incomplete or duplicate mother census')
        self.index = np.load(self.path.parent / (split + '-all.npy'), mmap_mode='r', allow_pickle=False)
        self.clock = np.load(self.path.parent / (split + '-clock-valid.npy'), mmap_mode='r', allow_pickle=False)
        expected = sum(s[split]['admitted_windows'] for s in self.manifest['sources'])
        if (self.index.shape != (expected, 3) or self.index.dtype != np.int32
                or self.clock.shape != (expected,) or self.clock.dtype != bool or not self.clock.all()):
            raise ValueError('complete ordered int32 tuples and valid clocks required')
        self.geometry = {r['key']: r for r in read(self.geometry_path)['objects']}
        self.cache, self.geometry_cache = OrderedDict(), OrderedDict()

    def __len__(self):
        return len(self.index)

    def __getstate__(self):
        return dict(manifest=str(self.path), split=self.split,
                    geometry_manifest=str(self.geometry_path), expected_sha256=self.expected_sha256,
                    geometry_gt_object=self.geometry_gt_object, cache_size=self.cache_size)

    def __setstate__(self, state):
        self.__init__(**state)

    def sequence(self, sequence_id):
        if sequence_id not in self.cache:
            row = self.rows[sequence_id]
            if row['split'] != self.split or row['status'] != 'ADMITTED':
                raise ValueError('reader split/admission mismatch')
            groups = []
            for field, identity in [('path', 'hand_meta_identity'),
                                    ('contact_path', 'contact_meta_identity'),
                                    ('event_path', 'event_meta_identity')]:
                folder = Path(row[field])
                if metadata_identity(folder / 'meta.json') != row[identity]:
                    raise ValueError('sequence metadata changed: ' + row['key'])
                groups.append({f.stem: np.load(f, mmap_mode='r', allow_pickle=False)
                               for f in sorted(folder.glob('*.npy'))})
            self.cache[sequence_id] = tuple(groups)
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(sequence_id)
        return self.cache[sequence_id]

    def geometry_for(self, sequence_id, slot):
        row = self.rows[sequence_id]
        name = row['objects'][slot]
        key = 's%d:%s' % (row['source_id'], name) if row['source_id'] < 3 else name
        if key not in self.geometry_cache:
            with np.load(self.geometry_path.parent / self.geometry[key]['path'], allow_pickle=False) as z:
                self.geometry_cache[key] = {k: z[k] for k in z.files}
            while len(self.geometry_cache) > 16:
                self.geometry_cache.popitem(last=False)
        self.geometry_cache.move_to_end(key)
        return self.geometry_cache[key]

    def __getitem__(self, index):
        sid, slot, tick = map(int, self.index[int(index)])
        sequence, contact, _ = self.sequence(sid)
        geometry = self.geometry_for(sid, slot)
        sample = sample_window(sequence, geometry, contact, tick, slot,
                               self.rows[sid]['source_id'], [sid, slot, tick])
        if self.geometry_gt_object:
            sample['geometry_object_future_gt'] = dense_object_future_gt(sample, sequence, tick, slot)
        sample = apply_goal_input_policy(sample, UNKNOWN_GOAL_POLICY)
        sample = apply_event_input_policy(sample, UNKNOWN_EVENT_POLICY)
        sample.update(event_raw_invalid=np.bool_(True), event_condition_dropout=np.bool_(False))
        return sample
