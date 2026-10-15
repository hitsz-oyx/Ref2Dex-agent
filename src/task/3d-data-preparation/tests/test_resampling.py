"""Native15 interpolation must preserve time, observations, SO(3), and missingness."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_mesh_data.resampling import resample_native15, validate_resampled_clock
from pose_mesh_data.cleaning import CleaningPolicy, clean_sequence, window_ticks


def native():
    n=300
    pose=np.broadcast_to(np.eye(4),(n,1,4,4)).copy()
    pose[:,0,:3,:3]=Rotation.from_euler('z',np.arange(n)*.1).as_matrix()
    pose[:,0,0,3]=np.arange(n)*.2
    hand=np.zeros((n,2,11,3),np.float32)
    hand[:,0,:,0]=np.arange(n)[:,None]*.3
    mask=np.zeros((n,2,11),bool);mask[:,0]=True
    return dict(native_frame_id=np.arange(n),hand=hand,hand_valid=mask,
        object_poses=pose,object_pose_valid=np.ones((n,1),bool))


def row():
    return dict(native_frames=300,action_duration_s=20,clock_status='CONSISTENT_WITH_NOMINAL15')


def test_exact_observations_midpoint_translation_and_short_arc_rotation():
    original=native();result=resample_native15(original,row())
    assert len(result['time'])==599 and result['time'][-1]==pytest.approx(299/15)
    np.testing.assert_array_equal(result['hand'][::2],original['hand'])
    np.testing.assert_allclose(result['object_poses'][::2],original['object_poses'],atol=1e-7)
    np.testing.assert_allclose(result['hand'][1,0,:,0],.15,atol=1e-7)
    np.testing.assert_allclose(result['object_poses'][1,0,:3,3],[.1,0,0],atol=1e-7)
    np.testing.assert_allclose(result['object_poses'][1,0,:3,:3],Rotation.from_euler('z',.05).as_matrix(),atol=1e-7)
    r=result['object_poses'][:,:,:3,:3]
    np.testing.assert_allclose(np.linalg.det(r),1,atol=1e-6)
    assert result['observed_frame'].sum()==300 and result['interpolated_frame'].sum()==299
    validate_resampled_clock(result,15)
    clean,quality,audit=clean_sequence(result,{'format':'canonical'},
        {'clock_kind':'resampled_fixed_fps','source_fps':15},CleaningPolicy())
    assert audit['clock_verified'] and audit['observed_frames']==300
    assert audit['interpolated_frames']==299 and not audit['changed_arrays']
    ticks,_=window_ticks(clean,quality,0,CleaningPolicy())
    assert ticks.tolist()==list(range(3,575,8))


def test_no_gap_filling_across_missing_annotation_and_no_false_second_hand():
    original=native()
    original['hand_valid'][10,0]=False;original['hand'][10,0]=np.nan
    original['object_pose_valid'][10]=False;original['object_poses'][10]=np.nan
    result=resample_native15(original,row())
    assert not result['hand_valid'][19:22].any()
    assert not result['object_pose_valid'][19:22].any()
    assert not result['hand_valid'][:,1].any()


def test_duration_conflict_and_forged_observation_mapping_are_rejected():
    with pytest.raises(ValueError,match='clock subset'):
        resample_native15(native(),{**row(),'action_duration_s':10})
    result=resample_native15(native(),row())
    result['observed_frame'][1]=True
    with pytest.raises(ValueError,match='mapping'):
        validate_resampled_clock(result,15)
