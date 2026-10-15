import numpy as np
from pose_mesh_data.motion_diagnostics import transition_diagnostics,window_coverage


def fixture():
    n=30;poses=np.broadcast_to(np.eye(4),(n,1,4,4)).copy()
    data=dict(time=np.arange(n)/30,object_poses=poses,object_pose_valid=np.ones((n,1),bool),
        hand=np.zeros((n,2,11,3)),hand_valid=np.ones((n,2,11),bool),segment=np.zeros(n,int))
    query=np.zeros((1,32,3));query[0,:,0]=.1
    return data,query


def values(data,query):return {k:(v,m,a,b) for k,v,m,a,b in transition_diagnostics(data,query)}


def test_recorded_15hz_transitions_keep_actual_clock_and_frame_ids():
    data,q=fixture();data['observed_frame']=np.arange(30)%2==0
    data['object_poses'][:,0,0,3]=np.arange(30)/30*2
    out=values(data,q)
    v,m,a,b=out['object_speed_mps']
    assert np.allclose(v[m],2) and np.array_equal(b,np.arange(2,30,2))
    assert np.allclose(out['object_step_m'][0][m],2/15)


def test_masks_and_segments_prevent_fake_jumps_from_missing_poses():
    data,q=fixture();data['object_poses'][10,0,0,3]=100;data['object_pose_valid'][10]=False
    data['hand'][12]=100;data['hand_valid'][12]=False
    data['segment'][20:]=1;data['object_poses'][20:,0,0,3]=10
    out=values(data,q)
    for key in ('object_speed_mps','hand_speed_mps','object_acceleration_mps2','hand_acceleration_mps2'):
        v,m,a,b=out[key];assert np.all(v[m]==0)


def test_surface_motion_detects_rotation_at_a_stationary_object_origin():
    data,q=fixture();angle=np.arange(30)*.01
    data['object_poses'][:,0,0,0]=np.cos(angle);data['object_poses'][:,0,0,1]=-np.sin(angle)
    data['object_poses'][:,0,1,0]=np.sin(angle);data['object_poses'][:,0,1,1]=np.cos(angle)
    out=values(data,q)
    assert np.all(out['object_step_m'][0]==0)
    assert np.allclose(out['angular_speed_radps'][0],.3)
    assert np.allclose(out['surface_step_m'][0],2*.1*np.sin(.005))


def test_window_coverage_includes_all_history_and_future_boundaries():
    assert np.array_equal(np.flatnonzero(window_coverage([3],30)),np.arange(28))
    assert not window_coverage([],30).any()
