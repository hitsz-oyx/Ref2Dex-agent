import copy
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from torch import nn

TASK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v5.contracts import SCHEMA,STATS_SCHEMA,SPARSE11_IN_21
from ptv3_shared_wam_v5.data import sample_window,collate,V5Windows
from ptv3_shared_wam_v5.model import SharedWAMV5
from ptv3_shared_wam_v5.objectives import forecast_loss
from ptv3_shared_wam_v5.preparation import surface_contact


class Geometry(nn.Module):
    def __init__(self,width):super().__init__();self.project=nn.Linear(width,width)
    def forward(self,batch,hand,valid):
        tokens=self.project(hand.mean(1,keepdim=True).expand(-1,4,-1))
        return dict(tokens=tokens,xyz=hand.new_zeros(len(hand),4,3),mask=torch.ones(len(hand),4,dtype=torch.bool),hand=hand,
                    point_features=hand.mean(1,keepdim=True).expand(-1,batch['xyz'].shape[1],-1),point_valid=batch['point_valid'])


def config():
    return dict(schema=SCHEMA,hand_points=21,surface_points=2048,interaction_points=2048,interaction_dim=17,
        horizon=24,goal_encoding='current-point-horizon-delta.v1',task_conditioning='shared-scene-hand-goal.v1',
        point_width=8,shared_width=16,shared_heads=4,shared_layers=1,head_width=8,head_heads=2,
        position_scale_m=.05,velocity_scale_mps=.5,acceleration_scale_mps2=5.,interaction_edge_width=8,
        checkpoint_interaction=False,checkpoint_contact=True,contact_chunk_size=256,
        hand_weight=1.,scene_weight=1.,motion_tau_m=.002,motion_floor=.1)


def stats():
    z=np.zeros((24,3)).tolist();s=np.ones((24,3)).tolist()
    return dict(schema=STATS_SCHEMA,split='train',units='m',fps=30,history=4,horizon=24,hand_points=21,
                source_ids=[1],hand=dict(mean=z,std=s),scene=dict(mean=z,std=s))


def fixture():
    n=28;rng=np.random.default_rng(2)
    joints=rng.normal(0,.01,(n,2,21,3)).astype(np.float32)
    poses=np.broadcast_to(np.eye(4),(n,1,4,4)).copy().astype(np.float32)
    seq=dict(joints=joints,joint_valid=np.ones((n,2,21),bool),surface_offsets=np.zeros((n,2,778,3),np.float16),
        surface_valid=np.ones((n,2),bool),surface_parts=np.zeros((2,778),np.uint8),
        object_poses=poses,object_pose_valid=np.ones((n,1),bool))
    geo=dict(surface=rng.normal(0,.02,(2048,3)).astype(np.float32),normals=np.tile([0,0,1],(2048,1)).astype(np.float32),
             query=rng.normal(0,.02,(32,3)).astype(np.float32))
    contact=dict(contact=np.ones((n,1,2048),np.float16),part=np.full((n,1,2048),3,np.uint8),valid=np.ones((n,1),bool))
    return seq,geo,contact


def batch():
    return collate([sample_window(*fixture(),3,0,1,[0,0,3])])


def test_joint21_order_and_observed_history_do_not_read_future_hand():
    assert SPARSE11_IN_21==(0,1,5,9,13,17,4,8,12,16,20)
    seq,geo,contact=fixture();first=sample_window(seq,geo,contact,3,0,1,[0,0,3])
    seq['joints'][4:]+=10
    second=sample_window(seq,geo,contact,3,0,1,[0,0,3])
    for key in ('start_hand','start_hand_history','dense_hand','dense_history','features','xyz'):
        np.testing.assert_array_equal(first[key],second[key])
    assert not np.array_equal(first['trajectory_target'],second['trajectory_target'])


def test_model_joint21_contact16_and_future_label_invariance():
    c=config();m=SharedWAMV5(stats(),c,Geometry(8)).eval();b=batch()
    with torch.no_grad():first=m(b,return_shared=True)
    assert first['trajectory'].shape==(1,24,2,21,3)
    assert first['interaction'].shape==(1,1,24,2048,17)
    assert first['shared_tokens'].shape[1]==4+42+32
    for key in ('trajectory_target','contact_target','scene_target'):b[key].fill_(float('nan'))
    b['part_target'].fill_(-999);b['trajectory_valid'].zero_();b['interaction_valid'].zero_()
    with torch.no_grad():second=m(b)
    for key in ('trajectory','scene','interaction'):assert torch.equal(first[key],second[key])


def test_masked_missing_surface_is_not_negative_contact_and_backward_is_finite():
    c=config();m=SharedWAMV5(stats(),c,Geometry(8)).train();b=batch()
    b['interaction_valid'].zero_();b['contact_target'].fill_(float('nan'));b['part_target'].fill_(-999)
    p=m(b);loss,terms=forecast_loss(p,b,c,m.normalizer)
    assert terms['contact'].item()==0 and terms['part'].item()==0
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None)


def test_contact_query_chunking_preserves_predictions_and_gradients():
    c=config();m=SharedWAMV5(stats(),c,Geometry(8)).train();b=batch()
    p=m(b);loss,terms=forecast_loss(p,b,c,m.normalizer);loss.backward()
    ref=p['interaction'].detach();grad=m.interaction_head[-1].weight.grad.clone();m.zero_grad(set_to_none=True)
    c['contact_chunk_size']=2048;c['checkpoint_contact']=False
    p=m(b);loss,_=forecast_loss(p,b,c,m.normalizer);loss.backward()
    torch.testing.assert_close(ref,p['interaction']);torch.testing.assert_close(grad,m.interaction_head[-1].weight.grad)
    assert grad.abs().sum()>0


def test_surface_proximity_retains_part_and_zero_far_field():
    contact,part=surface_contact(np.array([[0,0,0],[1,0,0]]),np.array([3,15],np.uint8),np.array([[0,0,0],[1,0,0],[0,0,1]]))
    np.testing.assert_array_equal(contact,[1,1,0]);np.testing.assert_array_equal(part[:2],[3,15])


def test_v4_configuration_and_statistics_are_rejected():
    c=config();c['schema']='ref2dex.ptv3-shared-wam.v4'
    with pytest.raises(ValueError,match='v5'):SharedWAMV5(stats(),c,Geometry(8))
    c=config();s=stats();s['hand_points']=11
    with pytest.raises(ValueError,match='joint21'):SharedWAMV5(s,c,Geometry(8))


def test_contact_broad_phase_preserves_positive_parts_and_distance_field():
    rng=np.random.default_rng(17);vertices=rng.normal(0,.02,(100,3));parts=rng.integers(0,16,100,dtype=np.uint8)
    queries=np.concatenate((vertices[:10],rng.normal(0,.1,(100,3))))
    distance=np.linalg.norm(queries[:,None]-vertices[None],axis=-1);which=distance.argmin(-1);nearest=distance.min(-1)
    expected=np.exp(-np.maximum(nearest-.003,0)/.01);expected[nearest>.033]=0
    contact,labels=surface_contact(vertices,parts,queries)
    np.testing.assert_array_equal(contact,expected.astype(np.float16))
    np.testing.assert_array_equal(labels[contact>0],parts[which][contact>0])


def test_shadow_joint21_matches_recorded_fk_and_has_distinct_thumb_cmc_mcp():
    root=TASK.parents[2]
    asset=root/'outputs/video-data-preparation/unidexgrasp-policy-assets-20261014-r1/assets/mjcf/open_ai_assets/hand/robot.xml'
    recorded=root/'outputs/video-data-preparation/unidexgrasp-multi-replay-20261014-r2/trajectories/episode-00-env-000.npz'
    if not asset.is_file() or not recorded.is_file():pytest.skip('local recorded Shadow audit assets unavailable')
    from ptv3_shared_wam_v5.shadow import ShadowSurface
    with np.load(recorded,allow_pickle=False) as z:q,origin,expected=z['q'],z['hand_root'],z['hand']
    joints,valid,surface,sv,parts=ShadowSurface(asset).reconstruct(q,origin)
    actual=joints[:,0,[0,2,5,9,13,17,4,8,12,16,20]]
    np.testing.assert_allclose(actual,expected,atol=5e-5,rtol=0)
    assert np.linalg.norm(joints[:,0,1]-joints[:,0,2],axis=-1).min()>.02


def test_main3_sampling_keeps_motion_weights_and_native_missing_strata():
    dataset=object.__new__(V5Windows)
    dataset.pools={'0':np.arange(30),'1':np.arange(20)}
    dataset.stratum_pools={'0':[np.arange(10),np.arange(10,20),np.arange(20,30)],
                          '1':[np.arange(10),np.arange(10,20),np.empty(0,np.int64)]}
    chosen=dataset.draw_source('0',50000,np.random.default_rng(2))
    weights=np.bincount(chosen//10,minlength=3)/len(chosen)
    np.testing.assert_allclose(weights,[.6,.2,.2],atol=.01)
    chosen=dataset.draw_source('1',50000,np.random.default_rng(2))
    weights=np.bincount(chosen//10,minlength=2)/len(chosen)
    np.testing.assert_allclose(weights,[.75,.25],atol=.01)
