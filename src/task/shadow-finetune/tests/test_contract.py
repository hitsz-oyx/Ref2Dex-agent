import numpy as np
import pytest
import torch
from torch import nn
from shadow_finetune.data import current_inputs,sample_sequence,collate
from shadow_finetune.model import ShadowActor,ShadowPolicy,loss
from test_contracts import config,stats,Geometry


def fixture():
    t=80;poses=np.tile(np.eye(4,dtype=np.float32),(t,1,1));poses[:,0,3]=np.arange(t)*.001
    hand=np.zeros((t,11,3),np.float32);hand[:,:,0]=np.arange(t)[:,None]*.002
    hand[:,:,1]=np.arange(11)[None]*.001
    s=dict(poses=poses,hand=hand,goal=np.broadcast_to([1.,0.,1.],(t,3)).copy(),
        observation=np.zeros((t,300),np.float32),action=np.broadcast_to(np.arange(t-1)[:,None],(t-1,24)).astype(np.float32))
    g=dict(surface=np.random.default_rng(3).normal(size=(64,3)).astype(np.float32)*.02,
        normals=np.broadcast_to([0.,0.,1.],(64,3)).astype(np.float32).copy())
    return s,g


def test_60hz_commands_and_30hz_geometry_alignment():
    s,g=fixture();a=sample_sequence(s,g,10)
    assert a['action_target'][:,0].tolist()==list(range(10,18))
    np.testing.assert_allclose(a['trajectory_target'][:,0,0,0],np.arange(1,25)*.004,atol=1e-7)
    np.testing.assert_allclose(a['start_hand_history'][0,:,0],.12,atol=1e-6)
    np.testing.assert_allclose(a['goal_object_delta'][:,0],.99,atol=1e-7)
    assert not a['start_hand_valid'][1].any()
    # Semantic order is already WAM order (thumb/ff/mf/rf/lf then tips), no Inspire reordering.
    np.testing.assert_allclose(a['start_hand'][0,:,1],np.arange(11)*.001,atol=1e-7)


def test_future_mutation_cannot_change_any_policy_input():
    s,g=fixture();first=sample_sequence(s,g,10)
    s['poses'][11:,0,3]+=100;s['hand'][11:]+=1
    changed=sample_sequence(s,g,10)
    for key in ('xyz','features','start_hand','start_hand_history','goal_object_delta','policy_state'):
        np.testing.assert_array_equal(first[key],changed[key])
    assert not np.array_equal(first['scene_target'],changed['scene_target'])
    assert not np.array_equal(first['trajectory_target'],changed['trajectory_target'])
    startup=sample_sequence(s,g,0)
    assert not startup['start_hand_history'].any()


def test_full_wam_action_gradient_and_single_encode():
    c=config();c.update(shared_width=48,shared_heads=4)
    net=ShadowPolicy(stats(),c,Geometry(c['point_width'])).eval()
    nn.init.normal_(net.actor.head.weight,std=.01)
    s,g=fixture();batch=collate([sample_sequence(s,g,5),sample_sequence(s,g,8)])
    p=net(batch);total,terms=loss(p,batch,c,net)
    assert p['action'].shape==(2,8,24) and torch.isfinite(total)
    for group in (net.forecaster.geometry,net.forecaster.interaction,net.forecaster.goal):
        gradients=torch.autograd.grad(terms['action'],list(group.parameters()),retain_graph=True,allow_unused=True)
        assert sum(float(x.abs().sum()) for x in gradients if x is not None)>0
