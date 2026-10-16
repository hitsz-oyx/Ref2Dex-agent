"""Provenance-preserving native -> Shared-WAM-v2 engineering boundary.

This wrapper does not grant training admission or change the v2 training loader.
The reader's in-memory manifest snapshot is the authority, rather than mutable
sample metadata or the periodically refreshed live index. Python callers remain
responsible for using this boundary; it is not a security sandbox.
"""
from dataclasses import dataclass, field
import json

import numpy as np

from .windows import PoseMeshWindows


@dataclass(frozen=True)
class NativeProvenance:
    manifest_path: str
    sequence_path: str
    source_id: int
    profile: str
    group_id: str
    split: str
    geometry_keys: tuple
    target_slot: int
    active_object_slots: tuple
    sample_id: tuple
    anchor_world_to_sample: tuple
    clock_status: str
    training_allowed: bool = field(default=False, init=False)

    @property
    def identity(self):
        # Local sequence/tick pairs collide across source manifests.
        return (self.manifest_path, self.source_id, *self.sample_id)


@dataclass(frozen=True)
class NativeV2Example:
    tensors: dict
    provenance: NativeProvenance


@dataclass(frozen=True)
class NativeV2Batch:
    model_inputs: dict
    provenance: tuple
    training_allowed: bool = field(default=False, init=False)


def _scalar(sample, key, kind):
    if key not in sample:
        raise ValueError('missing native metadata: ' + key)
    value = np.asarray(sample[key])
    if value.shape != () or value.dtype.kind not in kind:
        raise ValueError('strict scalar dtype required for ' + key)
    return value.item()


def _validate(example):
    if not isinstance(example, NativeV2Example):
        raise ValueError('bound NativeV2Example required, not an unbound sample')
    sample, record = example.tensors, example.provenance
    if not isinstance(record, NativeProvenance) or record.training_allowed is not False:
        raise ValueError('provisional native provenance required')
    if _scalar(sample, 'source_id', 'iu') != record.source_id:
        raise ValueError('source_id differs from manifest snapshot')
    if _scalar(sample, 'source_profile', 'U') != record.profile:
        raise ValueError('source_profile differs from manifest snapshot')
    if _scalar(sample, 'training_allowed', 'b') is not False:
        raise ValueError('engineering adapter requires training_allowed=False')
    sample_id = np.asarray(sample.get('sample_id'))
    if (sample_id.shape != (2,) or sample_id.dtype.kind not in 'iu'
            or not np.array_equal(sample_id, record.sample_id)):
        raise ValueError('sample_id differs from bound sequence/tick')
    anchor = np.asarray(sample.get('anchor_world_to_sample'))
    if (anchor.shape != (4, 4) or anchor.dtype.kind not in 'fi'
            or not np.isfinite(anchor).all()
            or not np.allclose(anchor, record.anchor_world_to_sample, atol=1e-6, rtol=0)):
        raise ValueError('anchor differs from source target pose')
    expected_slot = record.active_object_slots.index(record.target_slot)
    if _scalar(sample, 'goal_object_slot', 'iu') != expected_slot:
        raise ValueError('goal_object_slot differs from source target slot')
    if len(sample['scene_points']) != len(record.active_object_slots):
        raise ValueError('scene object slots differ from source observations')


class NativeV2EngineeringWindows:
    """Bind native samples to an immutable, source-local reader snapshot.

    A frozen combined engineering manifest can contain all six sources, including
    a dev-only or train-only source. The underlying reader's disjoint, nonempty
    overall TRAIN/dev policy is retained; no source split is silently changed.
    """
    def __init__(self, root, split, *, engineering=False, stride=8, cache_size=8):
        self._base = PoseMeshWindows(root, split, engineering=engineering,
                                     stride=stride, cache_size=cache_size)
        self.manifest_path = str(self._base.root / 'manifest.json')
        # Serialized private snapshot avoids aliasing the reader's public dict.
        self._manifest_snapshot = json.dumps(self._base.meta)
        self._rows = tuple(json.dumps(row) for row in self._base.meta['sequences'])
        self._index = tuple(self._base.index)
        for encoded in self._rows:
            row = json.loads(encoded)
            source = row.get('source_id')
            if type(source) is not int or source not in range(4, 10):
                raise ValueError('explicit integer native source_id 4..9 required')
            for key in ('profile', 'path', 'group_id', 'split'):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError('explicit nonempty manifest field required: ' + key)
            if 'training_allowed' in row and row['training_allowed'] is not False:
                raise ValueError('provisional row training_allowed must be False')

    def __len__(self):
        return len(self._index)

    def __getitem__(self, index):
        sequence_id, tick = self._index[index]
        row = json.loads(self._rows[sequence_id])
        # Reject in-memory reader mutations as well as forged sample metadata.
        if self._base.index[index] != (sequence_id, tick) or self._base.meta['sequences'][sequence_id] != row:
            raise ValueError('reader identity changed after manifest snapshot')
        if self._base.meta.get('training_allowed') is not False:
            raise ValueError('engineering manifest training_allowed must remain False')
        sequence = self._base._sequence(sequence_id)
        active = tuple(int(i) for i in np.flatnonzero(
            sequence['object_pose_valid'][tick-3:tick+1].all(0)))
        anchor = np.linalg.inv(sequence['object_poses'][tick, row['target_slot']])
        record = NativeProvenance(
            manifest_path=self.manifest_path,
            sequence_path=str((self._base.root / row['path']).resolve()),
            source_id=row['source_id'], profile=row['profile'], group_id=row['group_id'],
            split=row['split'], geometry_keys=tuple(row['objects']), target_slot=row['target_slot'],
            active_object_slots=active, sample_id=(sequence_id, tick),
            anchor_world_to_sample=tuple(tuple(float(v) for v in line) for line in anchor),
            clock_status=row.get('clock_status', 'unknown'))
        example = NativeV2Example(self._base[index], record)
        _validate(example)
        return example

    def manifest_snapshot(self):
        """Return an independent copy for audit output, never a live index read."""
        return json.loads(self._manifest_snapshot)


def native_v2_engineering_collate(examples, scene_cap=4096):
    """Keep provenance separate from model tensors; never truncate an object.

    This engineering batch is intentionally not the dict accepted by a training
    loop. Callers explicitly select ``model_inputs`` for a bounded wiring check.
    """
    if not examples or type(scene_cap) is not int or not 32 <= scene_cap <= 4096:
        raise ValueError('nonempty engineering batch and bounded scene_cap required')
    for example in examples:
        _validate(example)
        if len(example.tensors['xyz']) > scene_cap:
            raise ValueError('scene_cap would truncate native objects; increase cap')
    from ptv3_shared_wam.egodex_data import forecasting_collate
    tensors = forecasting_collate([example.tensors for example in examples], scene_cap=scene_cap)
    records = tuple(example.provenance for example in examples)
    if tensors['source_id'].tolist() != [record.source_id for record in records]:
        raise ValueError('collated sources differ from provenance order')
    return NativeV2Batch(tensors, records)
