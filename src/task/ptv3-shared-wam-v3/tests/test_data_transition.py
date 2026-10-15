import copy,json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ptv3_shared_wam_v3.data_transition import validate_subset_resume


def setup(tmp_path):
    payload=tmp_path/'payload.npz';payload.write_bytes(b'fixed payload');stamp=payload.stat()
    rows=[dict(source_id=s,source_sequence=str(i),split='train',path=str(payload),objects=['cad'],
        target_slot=0,profile='profile',payload_size_bytes=stamp.st_size,payload_mtime_ns=stamp.st_mtime_ns)
        for i,s in enumerate((4,5))]
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    indices=[np.array([[0,3],[0,11],[1,3]]),np.array([[0,3],[1,3]])]
    recipes=[]
    for root,index in zip((old,new),indices):
        np.savez(root/'windows.npz',verified_train=index,all_train=index,
            verified_dev=np.empty((0,2),np.int64),all_dev=np.empty((0,2),np.int64))
        s=(root/'windows.npz').stat()
        meta=dict(window_index='windows.npz',window_index_size_bytes=s.st_size,
            window_index_mtime_ns=s.st_mtime_ns,sequences=rows,objects=[dict(key='cad',path=str(payload))])
        meta['motion_admission']=dict(manual_decisions=dict(schema='ref2dex.manual-motion-decisions.v1',
            decision_source='explicit_user_visual_review',rows=[dict(row=0,source_sequence='0',verdict='exclude_tail',first_suspect_frame=33)]))
        (root/'manifest.json').write_text(json.dumps(meta))
        recipes.append(dict(main3_root='same',main3_identity='same',source_probabilities={'4':.5,'5':.5},
            native_root=str(root),native_index_stamp=[s.st_size,s.st_mtime_ns],training_allowed=True,
            source_counts={'4':int((index[:,0]==0).sum()),'5':1}))
    stats=dict(recipe=recipes[0],hand=dict(std=[[1.,2.,3.]]),scene=dict(std=[[1.,2.,3.]]))
    parent=dict(config={'seed':170},provenance=dict(world_size=4),stats=stats)
    newstats={**copy.deepcopy(stats),'recipe':recipes[1]}
    return parent,newstats


def test_reviewed_subset_retains_exact_normalizer_optimizer_and_dev(tmp_path):
    parent,stats=setup(tmp_path);r=validate_subset_resume(parent,parent['config'],stats,4)
    assert r['removed_windows']['verified_train']==1 and r['removed_windows']['verified_dev']==0
    assert r['reviewed_rows']==[0] and r['dev_unchanged']


def test_changed_normalization_is_rejected(tmp_path):
    parent,stats=setup(tmp_path);stats['hand']['std'][0][0]=2.
    with pytest.raises(ValueError,match='normalization'):validate_subset_resume(parent,parent['config'],stats,4)


def test_unreviewed_window_change_is_rejected(tmp_path):
    parent,stats=setup(tmp_path);root=Path(stats['recipe']['native_root'])
    z=dict(np.load(root/'windows.npz'));z['verified_train']=z['verified_train'][:1];np.savez(root/'windows.npz',**z)
    s=(root/'windows.npz').stat();m=json.loads((root/'manifest.json').read_text());m.update(window_index_size_bytes=s.st_size,window_index_mtime_ns=s.st_mtime_ns)
    (root/'manifest.json').write_text(json.dumps(m));stats['recipe']['native_index_stamp']=[s.st_size,s.st_mtime_ns]
    with pytest.raises(ValueError,match='reviewed strict subset'):validate_subset_resume(parent,parent['config'],stats,4)


def test_changed_payload_is_rejected(tmp_path):
    parent,stats=setup(tmp_path);(tmp_path/'payload.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='payload mutated'):validate_subset_resume(parent,parent['config'],stats,4)
