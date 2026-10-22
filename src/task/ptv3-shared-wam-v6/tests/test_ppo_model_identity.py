"""Do not silently label a v4 PPO runtime as v6; preserve sparse/event semantics."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
TASK=Path(__file__).resolve().parents[1];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/wam-rl-policy-v3/src')]
from ptv3_shared_wam_v6 import paths


def test_official_v4_loader_refuses_v6_checkpoint_before_model_construction(tmp_path):
    from wam_rl_policy_v3.wam_v4_runtime import V4Runtime
    file=tmp_path/'v6.pt';torch.save(dict(schema='ref2dex.ptv3-shared-wam.v6'),file)
    with pytest.raises(ValueError,match='v4 runtime requires'):
        V4Runtime(file,[],tmp_path,device='cpu')


def test_v6_online_sparse_hand_goal_event_roundtrip():
    from ptv3_shared_wam_v6.online_runtime import online
    from ptv3_shared_wam_v5.contracts import SPARSE11_IN_21,JOINT_NAMES
    from wam_rl_policy_v3.inspire_contract import NATIVE_TO_WAM,WAM_TO_NATIVE
    poses=np.broadcast_to(np.eye(4),(4,4,4)).copy();poses[:,:3,3]=[1,2,3]
    hand=np.arange(33,dtype=np.float32).reshape(11,3)/100
    native=hand[WAM_TO_NATIVE];canonical=native[NATIVE_TO_WAM]
    joints=np.zeros((4,2,21,3),np.float32);indices=np.array(SPARSE11_IN_21);joints[:,0,indices]=canonical
    valid=np.zeros((2,21),bool);valid[0,indices]=True
    geo=dict(surface=np.zeros((2048,3),np.float32),normals=np.tile([0,0,1],(2048,1)),query=np.zeros((32,3),np.float32))
    goal=poses[-1].copy();goal[2,3]+=.08
    event=dict(event_type=2,event_delta_steps=8.,event_valid=True,event_hz=30)
    sample=online(geo,poses,joints,np.zeros((4,2,778,3)),valid,np.zeros((2,778),bool),np.zeros((2,778),int),goal,event)
    np.testing.assert_allclose(sample['start_hand'][0,indices]+poses[-1,:3,3],hand,atol=1e-6)
    np.testing.assert_allclose(sample['goal_object_delta'],np.tile([0,0,.08],(32,1)),atol=1e-6)
    assert [JOINT_NAMES[i] for i in indices]==['wrist','thumb_base','index_base','middle_base','ring_base','little_base','thumb_tip','index_tip','middle_tip','ring_tip','little_tip']
    assert sample['event_type']==2 and sample['event_delta_steps']==8 and sample['event_hz']==30
    assert not sample['start_hand_valid'][1].any() and sample['start_hand_valid'].sum()==11
    assert not sample['dense_hand_valid'].any()
