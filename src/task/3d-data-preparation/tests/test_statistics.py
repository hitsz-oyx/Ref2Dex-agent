"""Vectorized moments agree with actual reader labels and source sampling math."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_mesh_data.statistics import Moments,mix_moments,sequence_moments
from pose_mesh_data.windows import pose_mesh_sample


def test_vector_moments_match_reader_with_rotating_anchor_missing_hand_and_object():
    n=80
    poses=np.broadcast_to(np.eye(4),(n,2,4,4)).copy()
    poses[:,0,:3,:3]=Rotation.from_euler('z',np.arange(n)*.03).as_matrix()
    poses[:,0,0,3]=np.arange(n)*.01
    poses[:,1,1,3]=np.arange(n)*.02
    hand=np.arange(n*2*11*3,dtype=np.float32).reshape(n,2,11,3)/1000
    hand_valid=np.ones((n,2,11),bool)
    hand_valid[8:12,1,3]=False
    hand[~hand_valid]=np.nan
    known=np.ones((n,2),bool);known[17:22,1]=False
    poses[~known]=np.nan
    seq=dict(time=np.arange(n)/30,hand=hand,hand_valid=hand_valid,
        object_poses=poses,object_pose_valid=known,segment=np.zeros(n,np.int64))
    surface=np.column_stack((np.linspace(0,.1,32),np.zeros((32,2))))
    geometry=dict(surface=surface,query=surface,normals=np.tile([1.,0,0],(32,1)))
    ticks=[3,11,35]
    result=sequence_moments(seq,ticks,[geometry,geometry],0,
        workspace=dict(origin_m=[-2,-2,-2],voxel_m=.01,max_grid=65534),chunk_size=2)
    expected={'hand':Moments(),'scene':Moments()}
    for tick in ticks:
        sample=pose_mesh_sample(seq,[geometry,geometry],tick,0,8,[0,tick],'test')
        expected['hand'].update(sample['trajectory_target'],sample['trajectory_valid'])
        expected['scene'].update(np.moveaxis(sample['scene_target'],1,0),np.moveaxis(sample['scene_valid'],1,0))
    for name,reference in expected.items():
        np.testing.assert_array_equal(result[name]['count'],reference.count)
        np.testing.assert_allclose(result[name]['mean'],reference.mean,atol=1e-7)
        np.testing.assert_allclose(result[name]['m2'],reference.m2,atol=1e-6)
    assert result['windows']==3 and not result['workspace_incompatible_ticks']
    incompatible=sequence_moments(seq,ticks,[geometry,geometry],0,
        workspace=dict(origin_m=[10,10,10],voxel_m=.01,max_grid=65534))
    assert incompatible['workspace_incompatible_ticks']==ticks


def test_source_composition_accounts_for_windows_and_valid_points():
    one,ten=Moments(),Moments()
    one.update(np.ones((24,1,3)),np.ones((24,1),bool))
    ten.update(np.zeros((24,10,3)),np.ones((24,10),bool))
    sources={'a':dict(windows=1,hand=one.result(),scene=one.result()),
             'b':dict(windows=1,hand=ten.result(),scene=ten.result())}
    mixed=mix_moments(sources,{'a':1,'b':1},.001)
    np.testing.assert_allclose(mixed['hand']['mean'],1/11)
    np.testing.assert_allclose(mixed['hand']['raw_std'],np.sqrt(10/121))
    # Duplicating a source's windows does not change its chosen source probability.
    many=Moments()
    for _ in range(10):many.combine(one.result())
    sources['a']=dict(windows=10,hand=many.result(),scene=many.result())
    np.testing.assert_allclose(mix_moments(sources,{'a':1,'b':1},.001)['hand']['mean'],1/11)
    with pytest.raises(ValueError,match='known sources'):
        mix_moments(sources,{'unknown':1},.001)


def test_invalid_nan_never_enters_sufficient_statistics():
    m=Moments()
    values=np.ones((24,2,3));values[:,1]=np.nan
    valid=np.zeros((24,2),bool);valid[:,0]=True
    m.update(values,valid)
    assert np.array_equal(m.count,np.ones(24)) and (m.mean==1).all() and not m.m2.any()


def test_accumulator_never_reads_dev_and_rejects_a_dev_row_in_train_index(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/run'))
    import accumulate_clean_moments
    from pose_mesh_data.cleaning import CLEANING_SCHEMA
    monkeypatch.setattr(accumulate_clean_moments,'REPO',tmp_path)
    monkeypatch.setattr(accumulate_clean_moments.subprocess,'check_output',lambda *a,**k:'test-commit\n')
    root=tmp_path/'clean';root.mkdir()
    n=60
    hand=np.zeros((n,2,11,3),np.float32)
    poses=np.broadcast_to(np.eye(4),(n,1,4,4)).copy()
    np.savez(root/'sequence.npz',time=np.arange(n)/30,hand=hand,hand_valid=np.ones((n,2,11),bool),
        object_poses=poses,object_pose_valid=np.ones((n,1),bool),segment=np.zeros(n,np.int64))
    surface=np.column_stack((np.linspace(0,.1,32),np.zeros((32,2))))
    np.savez(root/'geometry.npz',surface=surface,query=surface,normals=np.tile([1.,0,0],(32,1)))
    s=(root/'sequence.npz').stat()
    row=dict(path='sequence.npz',format='canonical',split='train',clock_verified=True,
        source_id=8,target_slot=0,objects=['test'],payload_size_bytes=s.st_size,payload_mtime_ns=s.st_mtime_ns)
    # Deliberately missing DEV payload: success proves that only TRAIN was read.
    rows=[row,{**row,'split':'dev','path':'must-not-read-dev.npz'}]
    np.savez(root/'windows.npz',verified_train=np.array([[0,3],[0,11]],np.int64))
    stamp=(root/'windows.npz').stat()
    meta=dict(cleaning_schema=CLEANING_SCHEMA,status='COMPLETED',training_allowed=False,
        sequences=rows,objects=[dict(key='test',path='geometry.npz')],
        source_profiles={'8':{'source_fps':30}},window_index='windows.npz',
        window_index_size_bytes=stamp.st_size,window_index_mtime_ns=stamp.st_mtime_ns)
    (root/'manifest.json').write_text(json.dumps(meta))
    (root/'progress.json').write_text(json.dumps({'status':'COMPLETED'}))
    encoder=tmp_path/'encoder.json'
    encoder.write_text(json.dumps(dict(voxel_origin_m=[-2,-2,-2],voxel_m=.01)))
    parent=tmp_path/'outputs/3d-data-preparation';parent.mkdir(parents=True)
    output=parent/'moments'
    result=accumulate_clean_moments.accumulate(root,output,encoder,workers=1)
    assert result['windows']==2 and result['split']=='train' and result['training_allowed'] is False
    artifact=json.loads((output/'source-moments.json').read_text())
    assert artifact['source_moments']['8']['hand']['count']==[44]*24
    np.savez(root/'windows.npz',verified_train=np.array([[1,3]],np.int64))
    stamp=(root/'windows.npz').stat()
    meta.update(window_index_size_bytes=stamp.st_size,window_index_mtime_ns=stamp.st_mtime_ns)
    (root/'manifest.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='DEV/unverified'):
        accumulate_clean_moments.accumulate(root,parent/'bad-moments',encoder,workers=1)
    assert not (parent/'bad-moments/source-moments.json').exists()
