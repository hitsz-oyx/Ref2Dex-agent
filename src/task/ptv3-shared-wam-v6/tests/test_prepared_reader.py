"""Reject split/identity drift at the new data-only reader boundary."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK/'src'))
from ptv3_shared_wam_v6.prepared_reader import PreparedWindows, sha256


@pytest.fixture
def census(tmp_path):
    rows = [dict(sequence_id=41, split='train', status='ADMITTED'),
            dict(sequence_id=9, split='dev', status='ADMITTED')]
    (tmp_path/'sequences.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for split, sid in [('train',41),('dev',9)]:
        np.save(tmp_path/(split+'-all.npy'), np.array([[sid,0,3]],np.int32))
        np.save(tmp_path/(split+'-clock-valid.npy'), np.array([True]))
    files = {p.name:dict(bytes=p.stat().st_size,sha256=sha256(p))
             for p in tmp_path.iterdir() if p.is_file()}
    manifest = dict(schema='ref2dex.nine-source-readiness.v1',expected_mothers=2,
        sources=[dict(train=dict(admitted_windows=1),dev=dict(admitted_windows=1))],files=files)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'geometry.json').write_text(json.dumps(dict(objects=[])))
    return tmp_path


def reader(root, split='train'):
    return PreparedWindows(root/'manifest.json',split,root/'geometry.json',sha256(root/'manifest.json'))


def test_plan_ids_are_not_list_offsets(census):
    ds = reader(census)
    assert len(ds) == 1 and ds.rows[41]['split'] == 'train'
    assert ds.index[0].tolist() == [41,0,3]
    # A dev mother must be rejected before opening arrays, even with a valid ID.
    with pytest.raises(ValueError,match='split/admission'):
        ds.sequence(9)


def test_bound_tuple_mutation_rejected(census):
    np.save(census/'train-all.npy',np.array([[9,0,3]],np.int32))
    with pytest.raises(ValueError,match='changed'):
        reader(census)


def test_manifest_pin_rejected(census):
    with pytest.raises(ValueError,match='manifest changed'):
        PreparedWindows(census/'manifest.json','train',census/'geometry.json','0'*64)


def test_false_clock_rejected_even_when_rebound(census):
    file = census/'train-clock-valid.npy'; np.save(file,np.array([False]))
    manifest = json.loads((census/'manifest.json').read_text())
    manifest['files'][file.name] = dict(bytes=file.stat().st_size,sha256=sha256(file))
    (census/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='valid clocks'):
        reader(census)
