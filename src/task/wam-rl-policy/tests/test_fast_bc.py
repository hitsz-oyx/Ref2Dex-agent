import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

TASK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TASK.parent/'shadow-finetune/src'))
from wam_rl_policy.fast_bc import batched_inputs,type_summaries
from wam_rl_policy.fast_bc import FastController
from shadow_finetune.data import current_inputs
from video_data_preparation.unidexgrasp_data import object_poses


def test_batched_h4_matches_original_all_features_and_ragged_geometry():
    rng=np.random.default_rng(20)
    obj=rng.normal(size=(3,4,7));obj[...,3:]=Rotation.random(12,random_state=20).as_quat().reshape(3,4,4)
    hand=rng.normal(size=(3,4,11,3));goal=rng.normal(size=(3,3))
    geometries=[dict(surface=rng.normal(size=(n,3)).astype(np.float32),normals=rng.normal(size=(n,3)).astype(np.float32)) for n in (32,45,60)]
    batch=batched_inputs(obj,hand,goal,geometries)
    for env,actual in enumerate(batch):
        expected=current_inputs(object_poses(obj[env]),hand[env],goal[env],geometries[env])
        assert actual.keys()==expected.keys()
        for key in actual:np.testing.assert_allclose(actual[key],expected[key],rtol=1e-6,atol=1e-6,err_msg=key)


def test_outside_support_zero_is_per_env_persistent_and_rejects_nan(monkeypatch):
    import torch
    import pytest
    from types import SimpleNamespace
    zeros=torch.zeros
    def cpu_zeros(*args,**kwargs):
        kwargs.pop('device',None);return zeros(*args,**kwargs)
    monkeypatch.setattr(torch,'zeros',cpu_zeros)
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self:self)
    c=FastController.__new__(FastController);c.history=[];c.invalid=set();c.allow_invalid_zero=True
    c.geometry=[dict(surface=np.zeros((32,3),np.float32),normals=np.ones((32,3),np.float32)) for _ in range(2)]
    c.model=SimpleNamespace(forecaster=SimpleNamespace(encode=lambda batch:(zeros(len(batch['policy_state']),1,4),torch.ones(len(batch['policy_state']),1,dtype=torch.bool),None)),
        actor=lambda state,shared,valid:torch.ones(len(state),8,24)*.25)
    def snapshot(bad=False):
        obj=np.zeros((2,7));obj[:,6]=1;hand=np.zeros((2,11,3))
        if bad:hand[1,:,0]=-3
        return dict(object=obj,hand=hand,goal=np.zeros((2,3)),observation=np.zeros((2,300),np.float32))
    action=c.act(snapshot(True),zeros(2,300),1)
    assert (action[0]==.25).all() and (action[1]==0).all() and c.invalid=={1}
    assert (c.act(snapshot(),zeros(2,300),2)[1]==0).all()
    assert (c.act(snapshot(),zeros(2,300),1)==.25).all() and not c.invalid
    invalid=snapshot();invalid['hand'][1,0,0]=np.nan
    with pytest.raises(FloatingPointError):c.act(invalid,zeros(2,300),1)


def test_wam_type_summary_masks_padding_and_keeps_semantic_groups():
    tokens=torch.ones(2,5,768,requires_grad=True)
    with torch.no_grad():
        tokens[:,2:4]*=2;tokens[:,4:]*=3;tokens[:,1]=float('nan')
    valid=torch.tensor([[True,False,True,True,True],[True,False,True,True,True]])
    z=type_summaries(tokens,valid,2,2)
    assert z.shape==(2,2304) and not z.requires_grad and torch.isfinite(z).all()
    assert torch.equal(z[:,:768],torch.ones(2,768))
    assert torch.equal(z[:,768:1536],torch.ones(2,768)*2)
    assert torch.equal(z[:,1536:],torch.ones(2,768)*3)
