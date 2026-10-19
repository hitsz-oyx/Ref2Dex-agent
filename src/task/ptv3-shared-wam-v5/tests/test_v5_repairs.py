import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from test_v5_contract import TASK,Geometry,config,stats,batch
from ptv3_shared_wam_v5.model import SharedWAMV5
from ptv3_shared_wam_v5.window_contract import history_admitted,history_joint_valid
from ptv3_shared_wam_v5.shadow import centered_capsule
from ptv3_shared_wam_v5.cache_contract import bind_cache,validate_binding,array_identity
from ptv3_shared_wam_v5.checkpoints import checkpoint_due,retained_steps
from ptv3_shared_wam_v5.metrics import forecast_metrics


def test_capsule_is_centered_for_both_trimesh_origin_conventions(monkeypatch):
    import trimesh
    original=trimesh.creation.capsule
    for shift in (0.,.02):
        def factory(*args,**kwargs):
            mesh=original(*args,**kwargs);mesh.apply_translation([0,0,shift]);return mesh
        monkeypatch.setattr(trimesh.creation,'capsule',factory)
        np.testing.assert_allclose(centered_capsule(.008,.02).bounds[:,2],[-.028,.028],atol=1e-7)


def test_alternating_hand_history_is_rejected_by_both_admission_and_reader_predicate():
    valid=np.zeros((30,2,21),bool)
    valid[np.arange(30),np.arange(30)%2]=True
    assert valid[:4].any((1,2)).all()  # The former loose predicate accepted this.
    assert not history_admitted(valid,np.array([3,4])).any()
    assert not history_joint_valid(valid,3).any()
    valid[:,0,7]=True
    assert history_admitted(valid,np.array([3,4])).all()


def test_cache_resume_rejects_protocol_and_upstream_drift_and_shadow_inheritance(tmp_path):
    root=tmp_path/'cache';root.mkdir();contract=bind_cache(root,'hand',dict(protocol='v2',upstream='A'))
    assert bind_cache(root,'hand',dict(protocol='v2',upstream='A'))==contract
    with pytest.raises(ValueError,match='fresh'):bind_cache(root,'hand',dict(protocol='v2',upstream='B'))
    old=tmp_path/'old';old.mkdir();(old/'s4-00000').mkdir();(old/'s1-00000').mkdir()
    shadow=root/'s4-00000';shadow.symlink_to(old/'s4-00000')
    human=root/'s1-00000';human.symlink_to(old/'s1-00000')
    with pytest.raises(ValueError,match='identity'):validate_binding(shadow,{},contract,old)
    validate_binding(human,{},contract,old)
    assert array_identity(dict(x=np.zeros(3)))!=array_identity(dict(x=np.ones(3)))


def test_local_contact_reads_same_id_feature_and_backpropagates_without_query_mixing():
    class LocalGeometry(Geometry):
        def __init__(self):super().__init__(8);self.local=nn.Parameter(torch.zeros(1,2048,8))
        def forward(self,*args):
            result=super().forward(*args);result['point_features']=self.local;return result
    m=SharedWAMV5(stats(),config(),LocalGeometry()).eval();b=batch()
    before=m(b)['interaction_logits'].detach()
    with torch.no_grad():m.geometry.local[:,7,0]=2
    p=m(b);after=p['interaction_logits']
    assert not torch.equal(before[:,:,:,7],after[:,:,:,7])
    torch.testing.assert_close(before[:,:,:,8:],after[:,:,:,8:],rtol=0,atol=0)
    after[:,:,:,7].sum().backward()
    assert m.geometry.local.grad[:,7].abs().sum()>0
    assert m.geometry.local.grad[:,8:].abs().sum()==0
    b['interaction_points']=b['interaction_points'].flip(2)
    with pytest.raises(ValueError,match='point order'):m(b)


def test_hand_in_ptv3_false_excludes_both_joint_and_surface_voxels():
    from ptv3_shared_wam_v5.geometry import DenseJointGeometry
    class Backbone(nn.Module):
        def forward(self,values):return SimpleNamespace(feat=values['feat'])
    c=dict(config(),scene_tokens=4,voxel_m=.005,voxel_origin_m=[-2.,-2.,-2.],hand_in_ptv3=False)
    geometry=DenseJointGeometry(c,backbone=Backbone()).eval();b=batch()
    hand=torch.randn(1,42,8);valid=b['start_hand_valid'].flatten(1)
    first=geometry(b,hand,valid)
    b['dense_hand'].fill_(.5);b['start_hand'].fill_(.5)
    second=geometry(b,hand*10,valid)
    torch.testing.assert_close(first['point_features'],second['point_features'],rtol=0,atol=0)
    assert first['hand'].abs().sum()==0
    assert first['point_features'].shape==(1,2048,8)


def test_first_and_milestone_checkpoints_survive_recent_rotation():
    c=dict(checkpoint_interval=1000,checkpoint_milestone_interval=40000,save_first_step=True,checkpoint_keep_recent=2)
    assert checkpoint_due(1,80000,c) and checkpoint_due(40000,80000,c)
    assert retained_steps([1,39000,40000,41000,42000,43000],c)=={1,40000,42000,43000}


def test_root_relative_metric_separates_translation_and_masks_unknown_contact_phase():
    b=batch();p=dict(trajectory=b['trajectory_target']+.01,scene=b['scene_target'].clone(),
        interaction=torch.zeros(1,1,24,2048,17),interaction_prediction_valid=b['interaction_valid'].clone())
    r=forecast_metrics(p,b)
    assert r['hand/root']['sum']/r['hand/root']['count']==pytest.approx(17.320508,rel=1e-5)
    assert r['hand/root_relative']['sum']<.001
    assert sum(sum(row) for row in r['part/confusion_gt025']['matrix'])==24*2048
    b['interaction_valid'].zero_();r=forecast_metrics(p,b)
    assert r['hand/contact_phase']['count']==0 and r['hand/noncontact_phase']['count']==0
