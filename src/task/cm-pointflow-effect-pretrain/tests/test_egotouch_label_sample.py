"""Pinned-byte integrity and official TRAIN membership of schema acquisition."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/audit/sample_egotouch_labels.py'
spec = importlib.util.spec_from_file_location('egotouch_label_sample', SCRIPT)
sample = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sample)


def test_pinned_git_and_lfs_bytes_reject_corruption():
    data = b'valid-label'
    blob = hashlib.sha1(b'blob 11\0' + data).hexdigest()
    for entry in [dict(size=11, oid=blob),
                  dict(size=11, lfs=dict(oid=hashlib.sha256(data).hexdigest()))]:
        sample.verify_bytes(data, entry)
        with pytest.raises(ValueError, match='checksum'):
            sample.verify_bytes(b'bad-label!!', entry)
        with pytest.raises(ValueError, match='length'):
            sample.verify_bytes(b'short', entry)


def test_small_holdout_recording_is_never_selected():
    inventory = []
    for record, size in [('Home/task_a/train', 10), ('Home/task_b/train', 12),
                         ('Home/task_c/test', 1)]:
        for label in sample.LABELS:
            inventory.append(dict(type='file', path=record + '/' + label, size=size))
    entries = sample.select(inventory, dict(train=['/root/Home/task_a/train.hdf5',
                                                  '/root/Home/task_b/train.hdf5'],
                                            test=['/root/Home/task_c/test.hdf5']))
    assert len(entries) == 10
    assert all('/test/' not in e['path'] for e in entries)


def test_label_clock_preserves_gaps_instead_of_renumbering():
    spec = importlib.util.spec_from_file_location(
        'egotouch_label_audit', SCRIPT.with_name('audit_egotouch_label_schema.py'))
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    result = audit.clock([dict(frame_index=0, ts=0.), dict(frame_index=2, ts=2 / 30),
                          dict(frame_index=3, ts=3 / 30)])
    assert result['frame_ids'] == [0, 2, 3]
    assert not result['contiguous_from_zero']
    assert result['monotonic_frame_ids']
    duplicate = audit.clock([dict(frame_index=0, ts=0.), dict(frame_index=0, ts=0.)])
    assert not duplicate['unique_frame_ids']
    assert not duplicate['strictly_increasing_timestamps']
