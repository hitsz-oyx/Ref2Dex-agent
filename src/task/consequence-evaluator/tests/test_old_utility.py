from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
from consequence_evaluator.old_utility import teacher,OldUtility,pw_sample,future_from_prediction,panel_metrics,PW_ORDER


def test_frozen_teacher_uses32future_states_and_post8_regions():
    h=np.full(32,.1,np.float32);pair=np.ones(32,bool)
    y,u=teacher(np.float32(.1),True,h,pair,np.float32(0))
    assert y.shape==(8,) and u==1.25
    pair[-1]=False;h[-1]=0
    assert teacher(np.float32(.1),True,h,pair,np.float32(0))[1]<u
    with pytest.raises(ValueError):teacher(np.float32(.1),True,h[:24],pair[:24],np.float32(0))


def test_geometry_adapter_preserves_native_hand_order_and_masks_left_hand():
    obj=np.tile(np.eye(4,dtype=np.float32),(4,1,1));hh=np.arange(33,dtype=np.float32).reshape(11,3)[None].repeat(4,0)
    future=hh[-1:]+np.arange(1,25,dtype=np.float32)[:,None,None]
    canonical=dict(points=np.zeros((512,3),np.float32),normals=np.ones((512,3),np.float32),radius=.1,center=np.zeros(3))
    sample=pw_sample(obj,hh,future,canonical)
    assert sample['features'].shape==(523,18)
    assert np.array_equal(sample['action'][:,:11,:3],future[:,PW_ORDER])
    assert not sample['action_valid'][:,11:].any() and not sample['action'][:,11:].any()
    rot=np.tile(np.eye(3),(24,1,1));trans=np.zeros((24,3))
    z=future_from_prediction(rot,trans,future)
    assert np.array_equal(z[:,12:].reshape(24,11,3),future)
    rot[:]=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    z=future_from_prediction(rot,trans,future)
    assert np.allclose(z[:,12:].reshape(24,11,3),future@rot[0])
    obj[-1,0,3]=1
    with pytest.raises(ValueError):pw_sample(obj,hh,future,canonical)


def test_rank_ties_have_half_credit_and_baseline_first_top1():
    y=np.array([[1.25]*7,[-1,0,.25,.5,.75,1,1.25]])
    equal=panel_metrics(y,np.zeros_like(y));perfect=panel_metrics(y,y)
    assert equal['informative_anchors']==1 and equal['pairwise_accuracy']==.5
    assert equal['choices']==[0,0] and perfect['mean_regret']==0
    assert perfect['pairwise_accuracy']==1 and perfect['top1_agreement']==1


def test_same_initial_ha_model_cannot_read_future():
    torch.manual_seed(8);model=OldUtility(9,width=16,layers=1).eval()
    h=torch.randn(2,9);a=torch.randn(2,24,18);z=torch.randn(2,24,45)
    with torch.no_grad():
        assert torch.equal(model(h,a,z,False),model(h,a,z.flip(0),False))
        assert not torch.equal(model(h,a,z,True),model(h,a,z.flip(0),True))
