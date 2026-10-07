"""Official held-out isolation and immutable native reuse; synthetic CPU fixtures."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / 'src'))
from oakink_wm.native_splits import arctic_protocol, grab_split, official_split

spec = importlib.util.spec_from_file_location('native_full_entry', TASK / 'tools/run/prepare_native_full.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def make_pack(path):
    """Truthfully synthetic motion; its audit marker tests prior-pack wiring only."""
    root = path / 'processed'
    (root / 'canonical').mkdir(parents=True)
    cloud = np.zeros((512, 3), np.float32)
    cloud[:, 0] = np.linspace(-.01, .01, 512)
    np.savez(root / 'canonical/cloud.npz', points=cloud, normals=np.zeros_like(cloud),
             radius=.01, center=np.zeros(3))
    records, rows = [], []
    for sid, (source, name) in enumerate([('grab', 's1/elephant_lift.npz'),
                                       ('arctic', 's05/box_grab_01.mano.npy'),
                                       ('arctic', 's01/box_grab_01.mano.npy')]):
        seq = source + '_' + name.replace('/', '_').split('.')[0]
        dest = root / 'sequences' / seq
        dest.mkdir(parents=True)
        poses = np.broadcast_to(np.eye(4, dtype=np.float32), (40, 1, 4, 4)).copy()
        # Moves then returns; endpoint-only classification would be wrong.
        poses[10:15, 0, 0, 3] = .01
        arrays = dict(hand=np.zeros((40, 2, 11, 3), np.float32),
                      hand_valid=np.ones((40, 2), bool), poses=poses,
                      pose_valid=np.ones((40, 1), bool), program=np.zeros((40, 1), bool),
                      near=np.ones((40, 1), bool), frame_ids=np.arange(40),
                      source_frame_ids=np.arange(40), timestamps=np.arange(40) / 30,
                      centers=np.zeros((1, 3), np.float32))
        for key, value in arrays.items():
            np.save(dest / (key + '.npy'), value)
        raw = path / ('fixture_source_' + str(sid))
        raw.write_bytes(b'synthetic engineering fixture')
        record = dict(source=source, sequence=seq, source_sequence=name, source_path=str(raw),
                      source_sha256=entry.sha(raw), dependencies={}, frames=40,
                      eligible_windows=1, objects=['cloud'], split='train',
                      program_available=False, motion_windows=1)
        entry.write_json(dest / 'meta.json', record)
        records.append(record)
        rows.append([sid, 0, 3, 1])
    np.save(root / 'index_train.npy', np.asarray(rows, np.int64))
    for split in ('val', 'test'):
        np.save(root / ('index_' + split + '.npy'), np.empty((0, 4), np.int64))
    meta = dict(schema='ref2dex.native-wm30.v1', status='COMPLETED', training_allowed=True,
                fps=30, records=records, sequences=[r['sequence'] for r in records])
    entry.write_json(root / 'manifest.json', meta)
    entry.write_json(path / 'audit.json', dict(verdict='ENGINEERING_PASS', fixture=True))
    return records


def args_for(tmp_path, pack):
    protocol = tmp_path / 'protocol.json'
    entry.write_json(protocol, dict(train=['s01/box_grab_01'], val=['s05/box_grab_01'], test=['s03/box_grab_01']))
    output = tmp_path / 'new'
    output.mkdir()
    return SimpleNamespace(reindex=pack, output=output, protocol=protocol,
                           max_sequences=2000, seconds=60, max_output_gib=5, gpu=0)


def test_author_splits_are_object_based_for_grab_and_sequence_based_for_arctic(tmp_path):
    assert grab_split('s1/camera_lift.npz') == 'test'
    assert grab_split('s10/camera_lift.npz') == 'test'
    assert grab_split('s1/elephant_inspect.npz') == 'val'
    assert grab_split('s9/airplane_lift.npz') == 'train'
    args = args_for(tmp_path, tmp_path)
    protocol = arctic_protocol(args.protocol)
    assert official_split('arctic', 's05/box_grab_01.mano.npy', protocol) == 'val'
    with pytest.raises(ValueError, match='missing'):
        official_split('arctic', 's05/unknown.mano.npy', protocol)
    entry.write_json(args.protocol, dict(train=['s01/box'], val=['s01/box'], test=[]))
    with pytest.raises(ValueError, match='duplicate'):
        arctic_protocol(args.protocol)


def test_reindex_keeps_inputs_and_real_clock_but_isolates_official_validation(tmp_path):
    pack = tmp_path / 'old'
    records = make_pack(pack)
    before = {str(p): entry.sha(p) for p in pack.rglob('*') if p.is_file()}
    args = args_for(tmp_path, pack)
    entry.worker(args)
    meta = json.loads((args.output / 'processed/manifest.json').read_text())
    assert meta['status'] == 'COMPLETED' and meta['training_allowed']
    assert meta['windows'] == {'train': 1, 'val': 2, 'test': 0}
    assert [r['split'] for r in meta['records']] == ['val', 'val', 'train']
    assert json.loads((args.output / 'audit.json').read_text())['checked_motion_categories'] == 3
    for record in records:
        array = args.output / 'processed/sequences' / record['sequence'] / 'poses.npy'
        assert array.is_symlink()
        assert entry.sha(array) == before[str(pack / 'processed/sequences' / record['sequence'] / 'poses.npy')]
    assert all(entry.sha(p) == digest for p, digest in before.items())


def test_returning_motion_misclassification_is_rejected(tmp_path):
    pack = tmp_path / 'old'
    make_pack(pack)
    args = args_for(tmp_path, pack)
    entry.worker(args)
    path = args.output / 'processed/index_train.npy'
    rows = np.load(path)
    rows[:, 3] = 2
    np.save(path, rows)
    with pytest.raises(ValueError, match='full-future'):
        entry.audit_pack(args.output, arctic_protocol(args.protocol))


def test_source_change_fails_without_training_permission(tmp_path):
    pack = tmp_path / 'old'
    make_pack(pack)
    (pack / 'fixture_source_0').write_bytes(b'changed')
    args = args_for(tmp_path, pack)
    with pytest.raises(ValueError, match='source changed'):
        entry.worker(args)
    meta = json.loads((args.output / 'processed/manifest.json').read_text())
    assert meta['status'] == 'FAILED' and not meta['training_allowed']


def test_bounds_reject_input_output_overlap_and_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, 'ROOT', tmp_path)
    source = tmp_path / 'outputs/immutable'
    with pytest.raises(ValueError, match='overlap'):
        entry.guard_output(source / 'new', [source])
    with pytest.raises(ValueError, match='repository outputs'):
        entry.guard_output(tmp_path / 'outside')
    with pytest.raises(TimeoutError):
        entry.check_budget(tmp_path, 0, 1, 5)
