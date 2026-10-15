"""Merging snapshots preserves window rows and refuses duplicated lineage/leakage."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/run'))
import merge_clean_corpora
from pose_mesh_data.cleaning import CLEANING_SCHEMA


def snapshot(root, identity, split='train', group=None):
    root.mkdir()
    arrays = {k:np.empty((0,2),np.int64) for k in ('all_train','all_dev','verified_train','verified_dev')}
    arrays['all_'+split] = arrays['verified_'+split] = np.array([[0,3]],np.int64)
    np.savez(root/'windows.npz', **arrays)
    stamp = (root/'windows.npz').stat()
    row = dict(source_manifest=str(root/'original.json'), source_row=0,
        source_manifest_snapshot='original.json', path='sequence.npz', source_id=identity,
        split=split, group_id=group or str(identity)+'/'+split)
    meta = dict(cleaning_schema=CLEANING_SCHEMA,status='COMPLETED',training_allowed=False,
        units='m',fps=30,window_policy={'stride':8},source_profiles={str(identity):{'source_fps':30}},
        objects=[],sequences=[row],window_index='windows.npz',
        window_index_size_bytes=stamp.st_size,window_index_mtime_ns=stamp.st_mtime_ns)
    (root/'manifest.json').write_text(json.dumps(meta))
    (root/'progress.json').write_text(json.dumps({'status':'COMPLETED'}))
    return root


def test_merge_offsets_and_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(merge_clean_corpora,'REPO',tmp_path)
    monkeypatch.setattr(merge_clean_corpora.subprocess,'check_output',lambda *a,**k:'test-commit\n')
    parent = tmp_path/'outputs/3d-data-preparation'
    parent.mkdir(parents=True)
    first,second = snapshot(tmp_path/'a',8),snapshot(tmp_path/'b',10,'dev')
    result = merge_clean_corpora.merge([first,second],parent/'merged')
    assert result['windows']['verified_train']==result['windows']['verified_dev']==1
    with np.load(parent/'merged/windows.npz') as z:
        assert z['verified_train'].tolist()==[[0,3]]
        assert z['verified_dev'].tolist()==[[1,3]]
    meta=json.loads((parent/'merged/manifest.json').read_text())
    assert meta['training_allowed'] is False
    assert meta['sequences'][1]['path']==str(second/'sequence.npz')
    with pytest.raises(ValueError,match='duplicate source lineage'):
        merge_clean_corpora.merge([first,first],parent/'duplicated')
    third=snapshot(tmp_path/'c',11,'dev',group='8/train')
    with pytest.raises(ValueError,match='group overlap'):
        merge_clean_corpora.merge([first,third],parent/'leakage')
