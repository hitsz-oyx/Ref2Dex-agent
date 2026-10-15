"""Encoder admission uses current inputs and keeps normalization on the same subset."""
import json
from pathlib import Path
import sys

import numpy as np

from pose_mesh_data.statistics import Moments,observed_workspace,sequence_moments,subtract_moments
from pose_mesh_data.cleaning import CLEANING_SCHEMA
from pose_mesh_data.clean_data import CleanPoseMeshWindows

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/run'))
import prepare_encoder_view

WORKSPACE=dict(origin_m=[-2,-2,-2],voxel_m=.01,max_grid=65534)


def fixture():
    n=60
    hand=np.zeros((n,2,11,3),np.float32)
    valid=np.zeros((n,2,11),bool);valid[:,0]=True
    poses=np.broadcast_to(np.eye(4),(n,1,4,4)).copy()
    poses[11,0,0,3]=3
    seq=dict(time=np.arange(n)/30,hand=hand,hand_valid=valid,object_poses=poses,
        object_pose_valid=np.ones((n,1),bool),segment=np.zeros(n,np.int64))
    surface=np.column_stack((np.linspace(0,.1,32),np.zeros((32,2))))
    geometry=dict(surface=surface,query=surface,normals=np.tile([1.,0,0],(32,1)))
    return seq,geometry


def test_admission_does_not_consult_future_and_ignores_absent_hand():
    s,g=fixture()
    s['hand'][:,1]=np.nan
    assert observed_workspace(s,[3,11],[g],0,WORKSPACE).tolist()==[True,False]
    before=observed_workspace(s,[3],[g],0,WORKSPACE)
    s['object_poses'][4:]=np.nan;s['hand'][4:]=np.nan
    np.testing.assert_array_equal(observed_workspace(s,[3],[g],0,WORKSPACE),before)


def test_subtracted_moments_match_direct_remaining_population():
    rng=np.random.RandomState(2)
    a,b=rng.normal(size=(24,13,3)),rng.normal(size=(24,7,3))
    total,removed,reference=Moments(),Moments(),Moments()
    total.update(np.concatenate((a,b),1),np.ones((24,20),bool))
    removed.update(b,np.ones((24,7),bool));reference.update(a,np.ones((24,13),bool))
    result=subtract_moments(total.result(),removed.result())
    np.testing.assert_array_equal(result['count'],reference.count)
    np.testing.assert_allclose(result['mean'],reference.mean,atol=1e-14)
    np.testing.assert_allclose(result['m2'],reference.m2,atol=1e-13)


def test_view_updates_both_splits_and_subtracts_only_train_moments(tmp_path,monkeypatch):
    monkeypatch.setattr(prepare_encoder_view,'REPO',tmp_path)
    monkeypatch.setattr(prepare_encoder_view.subprocess,'check_output',lambda *a,**k:'test-commit\n')
    root=tmp_path/'parent';root.mkdir()
    s,g=fixture()
    np.savez(root/'geometry.npz',**g)
    rows=[]
    for split in ('train','dev'):
        p=root/(split+'.npz');np.savez(p,**s);stamp=p.stat()
        rows.append(dict(path=p.name,format='canonical',source_id=8,profile='test',split=split,
            group_id=split,objects=['test'],target_slot=0,clock_verified=True,training_allowed=False,
            source_manifest=str(root/'manifest.json'),source_manifest_snapshot='manifest.json',source_sequence=split,
            payload_size_bytes=stamp.st_size,payload_mtime_ns=stamp.st_mtime_ns,clean_windows=2))
    arrays={k:np.array([[int(k.endswith('dev')),3],[int(k.endswith('dev')),11]],np.int64)
            for k in ('all_train','verified_train','all_dev','verified_dev')}
    np.savez(root/'windows.npz',**arrays);stamp=(root/'windows.npz').stat()
    meta=dict(schema='ref2dex.pose-mesh.v1',cleaning_schema=CLEANING_SCHEMA,status='COMPLETED',
        training_allowed=False,units='m',fps=30,hand_order=['right','left'],objects=[dict(key='test',path='geometry.npz')],
        sequences=rows,source_profiles={'8':{'source_fps':30}},window_index='windows.npz',
        window_index_size_bytes=stamp.st_size,window_index_mtime_ns=stamp.st_mtime_ns,limitations=[])
    (root/'manifest.json').write_text(json.dumps(meta));(root/'progress.json').write_text('{"status":"COMPLETED"}')
    m=sequence_moments(s,[3,11],[g],0,WORKSPACE)
    m['workspace_incompatible_windows']=len(m.pop('workspace_incompatible_ticks'))
    stats=dict(status='COMPLETED',training_allowed=False,split='train',input_manifest=str(root/'manifest.json'),
        windows=2,geometry_files_read=1,workspace=WORKSPACE,workspace_incompatible_windows=1,source_moments={'8':m})
    moments=tmp_path/'moments';moments.mkdir()
    (moments/'source-moments.json').write_text(json.dumps(stats));(moments/'progress.json').write_text('{"status":"COMPLETED"}')
    np.savez(moments/'workspace-incompatible-windows.npz',row_tick=np.array([[0,11]],np.int64))
    output=tmp_path/'outputs/3d-data-preparation/view';output.parent.mkdir(parents=True)
    result=prepare_encoder_view.prepare(root,moments,output,workers=1)
    assert result['excluded_train']==result['excluded_all_dev']==1
    assert result['windows']['verified_train']==result['windows']['verified_dev']==1
    for split in ('train','dev'):
        data=CleanPoseMeshWindows(output,split,engineering=True)
        assert len(data)==1 and data[0].provenance.tick==3
    actual=json.loads((output/'source-moments.json').read_text())['source_moments']['8']
    expected=sequence_moments(s,[3],[g],0)
    for name in ('hand','scene'):
        np.testing.assert_array_equal(actual[name]['count'],expected[name]['count'])
        np.testing.assert_allclose(actual[name]['mean'],expected[name]['mean'],atol=1e-12)
        np.testing.assert_allclose(actual[name]['m2'],expected[name]['m2'],atol=1e-10)
    assert json.loads((root/'manifest.json').read_text())==meta
