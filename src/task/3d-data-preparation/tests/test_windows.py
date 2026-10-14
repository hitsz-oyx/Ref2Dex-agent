"""Physical known-answer tests independent of the source implementation."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_mesh_data.windows import pose_mesh_sample, validate_sequence


def sequence():
    clock=np.arange(80)/30
    poses=np.broadcast_to(np.eye(4),(80,2,4,4)).copy()
    poses[:,0,:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix()
    poses[:,0,0,3]=clock*.3
    poses[:,1,1,3]=.2
    hand=np.zeros((80,2,11,3));hand[:,0,:,0]=.02+.6*clock[:,None]
    mask=np.zeros((80,2,11),bool);mask[:,0]=True
    return dict(time=clock,hand=hand,hand_valid=mask,object_poses=poses,
                object_pose_valid=np.ones((80,2),bool),segment=np.zeros(80,np.int64))


def geometry():
    surface=np.column_stack((np.linspace(0,.1,32),np.zeros((32,2)))).astype(np.float32)
    return dict(surface=surface,normals=np.tile([1.,0.,0.],(32,1)).astype(np.float32))


def sample(s=None):return pose_mesh_sample(s or sequence(),[geometry(),geometry()],27,0,4,[0,27],'test-full-cad')


def test_multi_object_history_normal_and_future_use_same_current_anchor():
    s=sample()
    np.testing.assert_allclose(s['start_hand'][0,:,1],-.29,atol=1e-6)
    np.testing.assert_allclose(s['start_hand_history'][0,:,1],-.6,atol=1e-6)
    np.testing.assert_allclose(s['features'][:32,4],-.3,atol=1e-6)
    np.testing.assert_allclose(s['features'][32:,:3],np.tile([0,-1,0],(32,1)),atol=1e-6)
    np.testing.assert_allclose(s['scene_target'][0,-1,:,1],-.24,atol=1e-6)
    np.testing.assert_allclose(s['scene_target'][1],0,atol=1e-6)
    np.testing.assert_allclose(s['trajectory_target'][-1,0,:,1],-.48,atol=1e-6)
    np.testing.assert_allclose(s['goal_object_points']-s['start_object_points'],s['scene_target'][0,-1],atol=1e-6)
    assert not s['trajectory_valid'][:,1].any()
    assert (s['features'][:32,12]==1).all() and (s['features'][32:,12]==0).all()


def test_missing_future_masks_never_create_stationary_object_labels():
    s=sequence();s['object_pose_valid'][32:35,1]=False;s['object_poses'][32:35,1]=np.nan
    result=sample(s)
    assert not result['scene_valid'][1,4:7].any()
    assert not result['scene_target'][1,4:7].any()
    assert np.isfinite(result['scene_target']).all()
    s['hand_valid'][32:35,0,0]=False;s['hand'][32:35,0,0]=np.nan
    assert not sample(s)['trajectory_valid'][4:7,0,0].any()


def test_future_changes_task_goal_and_labels_but_leave_all_observations_fixed():
    s=sequence();a=sample(s)
    s['hand'][28:]+=5;s['object_poses'][28:,:,:3,3]+=10
    b=sample(s)
    for k in ('xyz','features','scene_points','start_hand','start_hand_history','start_object_points'):
        np.testing.assert_array_equal(a[k],b[k])
    assert not np.array_equal(a['scene_target'],b['scene_target'])
    assert not np.array_equal(a['goal_object_points'],b['goal_object_points'])


def test_reset_clock_and_reflected_rotations_are_rejected():
    s=sequence();s['segment'][35:]=1
    with pytest.raises(ValueError,match='reset'):sample(s)
    s=sequence();s['time'][40:]+=.01
    with pytest.raises(ValueError,match='30Hz'):validate_sequence(s)
    s=sequence();s['object_poses'][40,1,0,0]=-1
    with pytest.raises(ValueError,match='proper'):validate_sequence(s)


def test_shared_collate_keeps_context_objects_and_target_pairing():
    from ptv3_shared_wam.egodex_data import forecasting_collate
    a=sample();b=pose_mesh_sample(sequence(),[geometry(),geometry()],27,1,5,[1,27],'test-full-cad')
    batch=forecasting_collate([a,b])
    assert batch['scene_target'].shape==(2,2,24,32,3)
    assert batch['goal_object_slot'].tolist()==[0,1]
    assert batch['source_id'].tolist()==[4,5]
    np.testing.assert_allclose(batch['goal_object_delta'][1],b['scene_target'][1,-1],atol=1e-6)
    assert 'native_action' not in batch
