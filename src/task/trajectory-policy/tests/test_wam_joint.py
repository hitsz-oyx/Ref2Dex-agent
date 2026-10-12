"""Joint supervision, differentiability and paired deployment regressions."""
import copy

import pytest
import torch
from torch import nn

from trajectory_policy.wam_joint import (JOINT_SCHEMA,PATCH_SCHEMA,POLICY_MODULES,TRAIN_MODULES,
    action_gradient_evidence,joint_losses,load_policy_patch,paired_manifest_admission,
    policy_patch_admission,state_hash,buffer_hash)


def test_action_gradient_reaches_every_policy_module_and_R():
    modules={k:nn.Linear(3,3) for k in TRAIN_MODULES}
    value=torch.ones(2,3)
    for net in modules.values():value=net(value).tanh()
    evidence=action_gradient_evidence(value.square().mean(),modules)
    assert set(evidence)==set(TRAIN_MODULES)
    assert all(v>0 for v in evidence.values())


def test_detached_backbone_is_rejected():
    modules={k:nn.Linear(3,3) for k in TRAIN_MODULES}
    value=torch.ones(2,3)
    for name,net in modules.items():
        value=net(value).tanh()
        if name=='goal':value=value.detach()
    with pytest.raises(ValueError,match='scene'):
        action_gradient_evidence(value.square().mean(),modules)


def test_pure_PD_rows_do_not_supply_tau_labels_but_do_train_actions():
    tau=torch.ones(2,24,33,requires_grad=True)
    true=torch.zeros_like(tau);true[1]=float('nan')
    pd=torch.ones(2,12,requires_grad=True)
    geometry,action=joint_losses(tau,true,torch.tensor([True,False]),pd,torch.zeros_like(pd),torch.ones(12))
    assert torch.isfinite(geometry+action)
    (geometry+action).backward()
    assert tau.grad[0].abs().sum()>0
    assert tau.grad[1].abs().sum()==0
    assert pd.grad[1].abs().sum()>0


def fixture_pair():
    model=nn.Module()
    for name in POLICY_MODULES:
        net=nn.Sequential(nn.Linear(3,3),nn.BatchNorm1d(3)).eval()
        setattr(model,name,net)
    states={k:copy.deepcopy(getattr(model,k).state_dict()) for k in POLICY_MODULES}
    result=dict(status='COMPLETED',offline_screen_pass=True,head_sha256='h',checkpoint_sha256='r',
        train_initial_hash={k:'initial' for k in TRAIN_MODULES},
        train_final_hash={k:state_hash(v) for k,v in states.items()},
        action_gradient_checks={k:1. for k in TRAIN_MODULES},
        normalization_buffers_unchanged=True,frozen_initial_hash={'world':'w'},frozen_final_hash={'world':'w'})
    result['train_final_hash']['R']='finalR'
    result['normalization_final_hash']={'WAM':buffer_hash(model)}
    manifest=dict(schema=JOINT_SCHEMA,status='COMPLETED',head_sha256='h',checkpoint_sha256='r',
        head_training_R_sha256='r',train_modules=list(TRAIN_MODULES),policy_patch_schema=PATCH_SCHEMA,input_sha256={'source':'s'})
    packet=dict(schema=JOINT_SCHEMA,head={k:states[k] for k in ('trajectory','hand_query')},
        policy_patch={k:states[k] for k in ('scene','goal')},policy_patch_schema=PATCH_SCHEMA,
        joint_R_sha256='r',input_sha256=manifest['input_sha256'])
    return model,packet,manifest,result


def test_exact_pair_and_full_patch_load():
    model,packet,manifest,result=fixture_pair()
    paired_manifest_admission(manifest,result,manifest,'h','r')
    identity=policy_patch_admission(packet,manifest,result)
    load_policy_patch(model,packet,identity)
    assert state_hash(model.scene.state_dict())==identity['module_sha256']['scene']


@pytest.mark.parametrize('mismatch',['old_R','other_head','detached','frozen_drift'])
def test_pair_mismatches_are_rejected(mismatch):
    _,_,manifest,result=fixture_pair()
    head_sha,r_sha='h','r'
    if mismatch=='old_R':r_sha='old'
    if mismatch=='other_head':head_sha='other'
    if mismatch=='detached':result['action_gradient_checks']['scene']=0.
    if mismatch=='frozen_drift':result['frozen_final_hash']['world']='changed'
    with pytest.raises(ValueError):paired_manifest_admission(manifest,result,manifest,head_sha,r_sha)


def test_buffer_mutation_rejected_before_any_parameter_application():
    model,packet,manifest,result=fixture_pair()
    packet['policy_patch']['goal']['1.running_mean'][0]=1.
    result['train_final_hash']['goal']=state_hash(packet['policy_patch']['goal'])
    packet['policy_patch']['scene']['0.weight'].add_(1.)
    result['train_final_hash']['scene']=state_hash(packet['policy_patch']['scene'])
    initial=state_hash(model.state_dict())
    identity=policy_patch_admission(packet,manifest,result)
    with pytest.raises(ValueError,match='buffer changed'):load_policy_patch(model,packet,identity)
    assert state_hash(model.state_dict())==initial


def test_byte_exact_buffers_include_negative_zero():
    a=nn.Module();a.register_buffer('scale',torch.tensor([0.]))
    b=nn.Module();b.register_buffer('scale',torch.tensor([-0.]))
    assert torch.equal(a.scale,b.scale)
    assert buffer_hash(a)!=buffer_hash(b)


def test_nonpersistent_voxel_origin_is_checked_without_loading_a_missing_key():
    model,packet,manifest,result=fixture_pair()
    model.scene.register_buffer('voxel_origin_m',torch.tensor([-1.,-1.,-1.]),persistent=False)
    result['normalization_final_hash']={'WAM':buffer_hash(model)}
    identity=policy_patch_admission(packet,manifest,result)
    load_policy_patch(model,packet,identity)
    model.scene.voxel_origin_m.add_(1.)
    with pytest.raises(ValueError,match='config buffers'):
        load_policy_patch(model,packet,identity)
