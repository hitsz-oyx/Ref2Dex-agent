"""Recovery must reuse immutable receipts and preserve interrupted attempts."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('collector', Path(__file__).resolve().parents[1]/'tools/run/collect_rl660.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def receipt(root, name, batch, seed, status='COMPLETED'):
    folder = root/'batches'/name
    folder.mkdir(parents=True)
    records = []
    for sequence in batch:
        target = folder/'sequences'/sequence
        target.mkdir(parents=True)
        for artifact in ('trajectory.npz','canonical.npz'):
            (target/artifact).write_bytes(b'fixture')
        records.append(dict(sequence=sequence,path='sequences/'+sequence))
    value = dict(schema='ref2dex.rl660-batch.v1',status=status,sequences=batch,
        seed=seed,records=records,native_controller=dict(offset=[0],scale=[1]))
    path = folder/'manifest.json'
    path.write_text(json.dumps(value))
    return path


def test_resume_reuses_only_complete_owned_receipts(tmp_path):
    batches = [['a','b'],['c','d']]
    path = receipt(tmp_path,'batch000',batches[0],120)
    original = path.read_bytes()
    receipt(tmp_path,'batch001',batches[1],121,'RUNNING')
    indices,records,controller = collector.recover_batches(tmp_path,batches,120)
    assert indices == {0}
    assert [r['sequence'] for r in records] == ['a','b']
    assert all((tmp_path/r['path']/'trajectory.npz').is_file() for r in records)
    assert controller['scale'] == [1]
    assert path.read_bytes() == original


def test_resume_rejects_changed_seed_or_duplicate_receipt(tmp_path):
    batches = [['a']]
    receipt(tmp_path,'batch000',batches[0],120)
    with pytest.raises(ValueError,match='seed'):
        collector.recover_batches(tmp_path,batches,121)
    receipt(tmp_path,'batch000-resume01',batches[0],120)
    with pytest.raises(ValueError,match='duplicate'):
        collector.recover_batches(tmp_path,batches,120)


def test_resume_rejects_missing_artifacts(tmp_path):
    path = receipt(tmp_path,'batch000',['a'],120)
    (path.parent/'sequences/a/trajectory.npz').unlink()
    with pytest.raises(ValueError,match='artifacts'):
        collector.recover_batches(tmp_path,[['a']],120)
