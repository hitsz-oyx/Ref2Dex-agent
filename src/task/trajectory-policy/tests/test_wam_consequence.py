import numpy as np
import pytest

from trajectory_policy.wam_consequence import actual_consequence_sample, future_control_metadata
from trajectory_policy.wam_shadow import NATIVE_TO_WAM


def fixture():
    objects = np.tile(np.eye(4), (50, 4, 1, 1))
    for tick in range(50):
        for role in range(4):
            a=.02*tick+.1*role
            objects[tick,role,:3,:3]=[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]]
            objects[tick,role,:3,3]=[.002*tick,.001*tick**2,.05*role]
    hand = np.arange(50*4*11*3).reshape(50,4,11,3)*.0001
    canonical=dict(points=np.linspace(-.03,.03,512*3).reshape(512,3).astype(np.float32),
                   normals=np.zeros((512,3),np.float32),center=np.zeros(3),radius=.05)
    goal=objects[10,2].copy(); goal[:3,3]+=[.04,.01,.02]
    return objects,hand,goal,canonical


def sample(objects,hand,goal,canonical,tick=10,role=2):
    return actual_consequence_sample(objects,hand,tick,role,goal,canonical)


def test_action_is_absolute_current_displacement_and_meter_increment():
    objects,hand,goal,canonical=fixture(); s=sample(objects,hand,goal,canonical)
    current=objects[10,2]; future=hand[11:35,2][:,NATIVE_TO_WAM]
    p=(future-current[:3,3])@current[:3,:3]
    p0=(hand[10,2][NATIVE_TO_WAM]-current[:3,3])@current[:3,:3]
    a=s['action'].reshape(24,2,11,9)[:,0]
    assert np.allclose(a[:,:,:3],p,atol=1e-7)
    assert np.allclose(a[:,:,3:6],p-p0,atol=1e-7)
    increment=np.diff(np.concatenate((p0[None],p)),axis=0)
    assert np.allclose(a[:,:,6:9],increment,atol=1e-7)
    assert not np.allclose(a[:,:,6:9],increment*30)
    assert s['action_valid'].reshape(24,2,11)[:,0].all()
    assert not s['action_valid'].reshape(24,2,11)[:,1].any()
    assert not s['action'].reshape(24,2,11,9)[:,1].any()
    assert not s['trajectory_valid'].any() and not s['trajectory_target'].any()


def test_effect_moves_current_canonical_points_to_actual_future_in_fixed_anchor():
    objects,hand,goal,canonical=fixture(); s=sample(objects,hand,goal,canonical)
    current=objects[10,2]; effect=s['effect'][0]
    assert np.allclose(effect,np.linalg.inv(current)@objects[11:35,2],atol=1e-7)
    actual=np.einsum('tij,pj->tpi',objects[11:35,2,:3,:3],canonical['points'])+objects[11:35,2,None,:3,3]
    actual=(actual-current[:3,3])@current[:3,:3]
    predicted=np.einsum('tij,pj->tpi',effect[:,:3,:3],s['points'][0])+effect[:,None,:3,3]
    assert np.allclose(predicted,actual,atol=2e-7)


def test_future_mutation_cannot_change_current_scene_or_known_goal():
    objects,hand,goal,canonical=fixture(); before=sample(objects,hand,goal,canonical)
    changed_obj,changed_hand=objects.copy(),hand.copy()
    changed_obj[11:35,2,:3,3]+=[.3,-.2,.1]
    changed_hand[11:35,2]+=[.2,.1,-.1]
    after=sample(changed_obj,changed_hand,goal,canonical)
    for key in ('xyz','features','scene_object','points','object_features','start_hand',
                'start_hand_history','start_hand_valid','start_object_points','goal_object_points','object_goal_valid'):
        assert np.array_equal(before[key],after[key]),key
    assert not np.array_equal(before['action'],after['action'])
    assert not np.array_equal(before['effect'],after['effect'])
    assert not after['trajectory_valid'].any()


def test_global_rigid_coordinate_change_preserves_world_sample():
    objects,hand,goal,canonical=fixture(); before=sample(objects,hand,goal,canonical)
    g=np.eye(4); g[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]; g[:3,3]=[1,2,3]
    after=sample(g@objects,hand@g[:3,:3].T+g[:3,3],g@goal,canonical)
    for key in ('xyz','features','start_hand','start_hand_history','goal_object_points','action','effect'):
        assert np.allclose(before[key],after[key],atol=2e-6),key


def test_startup_and_terminal_frame_bounds_and_own_row_isolation():
    objects,hand,goal,canonical=fixture()
    first=sample(objects,hand,goal,canonical,tick=0)
    assert not first['start_hand_history'].any()
    sample(objects,hand,goal,canonical,tick=25)
    for tick in (-1,26):
        with pytest.raises(ValueError):
            sample(objects,hand,goal,canonical,tick=tick)
    before=sample(objects,hand,goal,canonical)
    objects[:,3,:3,3]+=10; hand[:,3]+=10
    after=sample(objects,hand,goal,canonical)
    assert np.array_equal(before['action'],after['action'])
    assert np.array_equal(before['effect'],after['effect'])


def test_handoff_provenance_keeps_each_actual_causal_control():
    mask=np.zeros((192,2),bool); mask[32:,1]=True
    metadata=future_control_metadata(mask,16,3)
    assert np.array_equal(metadata['control_ticks'],np.arange(16,40))
    assert np.array_equal(metadata['source_expert_applied'],mask[16:40])
    assert not metadata['own_expert_applied'][:16].any()
    assert metadata['own_expert_applied'][16:].all()
    assert (metadata['executor_code'][:16]==3).all()
    assert (metadata['executor_code'][16:]==4).all()
    assert not future_control_metadata(mask,16,2)['own_expert_applied'].any()
    assert future_control_metadata(mask,168,3)['own_expert_applied'].all()
    with pytest.raises(ValueError):
        future_control_metadata(mask,169,3)


def test_rejects_invalid_future_SE3_and_nonfinite_future_hand():
    objects,hand,goal,canonical=fixture()
    objects[11,2,0,0]=10
    with pytest.raises(ValueError):
        sample(objects,hand,goal,canonical)
    objects,hand,goal,canonical=fixture(); hand[11,2,0,0]=np.nan
    with pytest.raises(ValueError):
        sample(objects,hand,goal,canonical)
