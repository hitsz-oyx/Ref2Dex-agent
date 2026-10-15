import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ptv3_shared_wam_v3.review_data import Catalog,Record,Sequence,transform


def test_native_main3_uses_30hz_timestamps_not_camera_frame_ids(tmp_path):
    root=tmp_path/'corpus';path=root/'processed/sequences/s0';path.mkdir(parents=True)
    canonical=root/'processed/canonical';canonical.mkdir()
    (path/'meta.json').write_text(json.dumps({'objects':['object']}))
    t=28;poses=np.broadcast_to(np.eye(4),(t,1,4,4)).copy();poses[:,0,0,3]=np.arange(t)*.001
    arrays=dict(hand=np.zeros((t,2,11,3)),hand_valid=np.ones((t,2),bool),poses=poses,
        pose_valid=np.ones((t,1),bool),frame_ids=np.arange(t),timestamps=np.arange(t)/30)
    for key,value in arrays.items():np.save(path/(key+'.npy'),value)
    np.savez(canonical/'object.npz',points=np.random.default_rng(17).normal(0,.01,(512,3)))
    catalog=Catalog.__new__(Catalog)
    catalog.sources={2:({'kind':'native'}, {})}
    catalog.indices={(2,'train'):np.array([[0,0,3,0]])}
    record=Record(2,0,'s0','train',1,root,'main3')
    sequence=catalog.load(record)
    assert sequence.horizon_info(3,0)['seconds']==pytest.approx(.8)
    assert sequence.horizon_info(3,0)['max_query_displacement_m']==pytest.approx(.024)
    assert sequence.speed[3,0]==pytest.approx(.03)


def sequence():
    t=30;poses=np.broadcast_to(np.eye(4),(t,1,4,4)).copy();poses[:,0,0,3]=np.arange(t)*.01
    known=np.ones((t,1),bool);known[10:12]=False
    segment=np.zeros(t,int);segment[20:]=1
    data=dict(time=np.arange(t)/30,hand=np.zeros((t,2,11,3)),hand_valid=np.ones((t,2,11),bool),
        object_poses=poses,object_pose_valid=known,segment=segment)
    points=np.random.default_rng(17).normal(0,.01,(32,3))
    record=Record(4,1,'fixture','train',1,Path('.'),'native')
    return Sequence(record,data,[points],[points],['object'],0,np.array([[1,3]]),np.array([3]),'fixture')


def test_paths_do_not_bridge_missing_poses_or_reset_segments():
    data=sequence();segments=data.path_segments(0)
    # 29 links minus 3 links touching unknown poses, minus the reset link.
    assert len(segments)==25
    assert np.isnan(data.speed[10:13,0]).all()
    assert np.isnan(data.speed[20,0])
    assert data.horizon_info(3,0)['max_query_displacement_m']==pytest.approx(.16)
    assert data.horizon_info(3,0)['training_window']
    assert not data.horizon_info(4,0)['training_window']


def test_fixed_anchor_does_not_follow_moving_object():
    data=sequence();anchor=np.linalg.inv(data.poses[3,0])
    current=data.frame(3,anchor)[0][0];future=data.frame(27,anchor)[0][0]
    assert np.allclose(future-current,[.24,0,0])
    assert np.allclose(data.frame(10)[0][0],np.empty((0,3)))



def test_fully_quarantined_native_sequence_remains_in_catalog(tmp_path):
    main=tmp_path/'main';native=tmp_path/'native';recipe=tmp_path/'recipe'
    for root in (main/'processed',native,recipe):root.mkdir(parents=True)
    idx=main/'empty.npy';np.save(idx,np.empty((0,4),np.int64))
    sources=[]
    for sid in range(3):
        root=tmp_path/str(sid);(root/'processed').mkdir(parents=True)
        (root/'processed/manifest.json').write_text(json.dumps(dict(sequences=[])))
        sources.append(dict(root=str(root),indices={s:dict(path=str(idx)) for s in ('train','val')}))
    (main/'processed/manifest.json').write_text(json.dumps(dict(status='COMPLETED',sources=sources)))
    (recipe/'recipe.json').write_text(json.dumps(dict(main3_root=str(main),native_root=str(native))))
    row=dict(source_id=4,split='train',source_sequence='bad',source_manifest='batch001/manifest.json',
        motion_quality=dict(first_suspect_frame=1,reasons={'initial_fast_flight':[1]}))
    (native/'manifest.json').write_text(json.dumps(dict(status='COMPLETED',objects=[],sequences=[row],window_index='windows.npz')))
    np.savez(native/'windows.npz',verified_train=np.empty((0,2),np.int64),verified_dev=np.empty((0,2),np.int64))
    catalog=Catalog(recipe);records=catalog.select(4,'train')
    assert len(records)==1 and records[0].windows==0
    assert '整段隔离' in records[0].hint
    assert catalog.rows(records[0]).shape==(0,2)


def test_empty_training_ticks_still_allow_raw_sequence_review():
    data=sequence();data.rows=np.empty((0,2),np.int64);data.ticks=np.empty(0,np.int64)
    assert not data.horizon_info(3,0)['training_window']
    assert len(data.frame(3)[0][0])==32
