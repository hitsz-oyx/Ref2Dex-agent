import numpy as np
import pytest

from goal_point_wam.object_motion_goal import pack_object_motion_goal, rigid_consistency


def sample():
    xyz=np.array([[0,0,1],[.1,0,1],[0,.1,1],[.1,.1,1],[.2,0,1]],np.float32)
    future=xyz[None]+np.arange(1,25,dtype=np.float32)[:,None,None]*np.array([.001,0,0])
    return dict(query_xyz=xyz,query_valid=np.ones(5,bool),point_target=future.astype(np.float32),
                point_target_valid=np.ones((24,5),bool))


def test_current_selection_does_not_depend_on_future_visibility_or_motion():
    a=sample();original={k:v.copy() for k,v in a.items()}
    first=pack_object_motion_goal(a,np.arange(5),count=4)
    a['point_target'][:]+=100
    a['point_target_valid'][15:]=False
    a['point_target'][15:]=np.nan
    second=pack_object_motion_goal(a,np.arange(5),count=4)
    np.testing.assert_array_equal(first['selected_query_indices'],second['selected_query_indices'])
    np.testing.assert_array_equal(first['start_object_points'],second['start_object_points'])
    assert not second['desired_object_valid'][1:].any()
    assert np.isfinite(second['desired_object_points']).all()
    assert second['desired_object_points'][1:].sum()==0
    for k in ('query_xyz','query_valid'):np.testing.assert_array_equal(a[k],original[k])


def test_time_correspondence_padding_and_rigidity():
    a=sample();g=pack_object_motion_goal(a,np.arange(5),count=8)
    assert g['current_object_valid'].sum()==5
    assert (g['selected_query_indices'][5:]==-1).all()
    np.testing.assert_array_equal(g['goal_offsets'],[8,16,24])
    for h,t in enumerate(g['goal_offsets']):
        np.testing.assert_allclose(g['desired_object_points'][h,:5]-g['start_object_points'][:5],
                                   np.tile([t*.001,0,0],(5,1)),atol=1e-7)
    assert all(r['residual_p95_mm']<1e-4 and r['identifiable'] for r in rigid_consistency(g))
    with pytest.raises(ValueError):pack_object_motion_goal(a,np.arange(5),offsets=(16,8))


def test_static_is_specified_and_null_is_absent():
    a=sample();a['point_target'][:]=a['query_xyz']
    static=pack_object_motion_goal(a,np.arange(5))
    null=pack_object_motion_goal(a,np.array([],np.int64))
    assert static['desired_object_valid'].sum()==15
    assert not null['desired_object_valid'].any()


def test_goal_packet_has_no_hand_dependency_or_shared_world_storage():
    a=sample();a['hand']=np.full((28,2,11,3),np.nan)
    original={k:v.copy() for k,v in a.items()}
    first=pack_object_motion_goal(a,np.arange(5))
    for k in a:np.testing.assert_array_equal(a[k],original[k])
    a['hand'][:]=1000
    second=pack_object_motion_goal(a,np.arange(5))
    for k in first:np.testing.assert_array_equal(first[k],second[k])
    # Simulate arbitrary user goal editing; observed World geometry/labels stay intact.
    first['start_object_points'][:]=1000
    first['desired_object_points'][:]=1000
    first['desired_object_valid'][:]=False
    for k in ('query_xyz','point_target','point_target_valid'):
        np.testing.assert_array_equal(a[k],original[k])
