import copy
import torch
import pytest
from ptv3_shared_wam_v6.flow_matching import (
    FLOW_ENCODING, OVERNIGHT_CAMPAIGN, pack, unpack, linear_path, euler_sample,
)
from ptv3_shared_wam_v6.model import SharedWAMV6
from ptv3_shared_wam_v6.data import V6Windows, collate
from ptv3_shared_wam_v6.objectives import forecast_loss
from ptv3_shared_wam_v6.local_subset import validate_local_execution, EXPANDED_SCOPE, initialize_flow_weights
from test_v6_training import packet
from test_v5_contract import Geometry, stats


def flow_config(base):
    return dict(base,flow_matching=True,flow_encoding=FLOW_ENCODING,flow_steps=4,
        flow_noise_std=.1,flow_matching_weight=1.,flow_width=128,flow_layers=1,
        flow_eval_seed=172,flow_lr=.0001,flow_checkpoint=True)


def test_linear_path_and_euler_endpoint_and_gradient_are_exact():
    hand=torch.randn(2,24,2,21,3);scene=torch.randn(2,1,24,32,3)
    recovered=unpack(pack(hand,scene))
    torch.testing.assert_close(recovered[0],hand);torch.testing.assert_close(recovered[1],scene)
    noise=torch.randn(2,24,222);target=torch.randn_like(noise)
    mixed,v=linear_path(target,noise,torch.tensor([0.,1.]))
    torch.testing.assert_close(mixed[0],noise[0]);torch.testing.assert_close(mixed[1],target[1])
    for steps in (1,4,8):torch.testing.assert_close(euler_sample(lambda x,t:v,noise,steps),target,atol=2e-6,rtol=1e-5)
    velocity=torch.randn_like(noise,requires_grad=True)
    euler_sample(lambda x,t:velocity,noise,4).sum().backward()
    torch.testing.assert_close(velocity.grad,torch.ones_like(noise))


def test_flow_generated_coordinates_do_not_read_future_targets_and_eval_rng_is_fixed(packet):
    root,_,_,base=packet;config=flow_config(base)
    model=SharedWAMV6(stats(),config,Geometry(8));batch=collate([V6Windows(root/'manifest.json','dev')[1]])
    model.train();torch.manual_seed(6);a=model(batch)
    changed=copy.deepcopy(batch);changed['trajectory_target']+=10;changed['scene_target']-=10
    changed['geometry_object_future_gt']=torch.full((1,1,24,2048,3),float('nan'))
    torch.manual_seed(6);b=model(changed)
    for key in ('trajectory','scene','interaction'):
        torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    assert not torch.equal(a['flow_hand_target_velocity'],b['flow_hand_target_velocity'])
    loss,terms=forecast_loss(a,batch,config,model.normalizer);loss.backward()
    assert terms['flow_hand']>0 and torch.isfinite(model.flow_future.output.weight.grad).all()
    assert model.flow_future.output.weight.grad.norm()>0
    model.eval();state=torch.get_rng_state()
    with torch.no_grad():x=model(batch);y=model(changed)
    assert torch.equal(state,torch.get_rng_state())
    for key in ('trajectory','scene'):torch.testing.assert_close(x[key],y[key],atol=0,rtol=0)
    groups=model.parameter_groups(.00001,.00001)
    assert [g['name'] for g in groups]==['joint_geometry','shared_and_heads','conditional_flow']
    ids=[id(p) for g in groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(list(model.parameters()))


def test_flow_expansion_preserves_base_weights_and_explicit_budget_exception(packet):
    root,_,_,base=packet;cfg=flow_config(base)
    original=SharedWAMV6(stats(),base,Geometry(8));model=SharedWAMV6(stats(),cfg,Geometry(8))
    saved=dict(schema=base['schema'],step=4371,world_size=3,data_identity='same',config=base,model=original.state_dict())
    binding=initialize_flow_weights(model,saved,root/'source.pt',cfg,'same')
    assert binding['new_parameters'] and not binding['optimizer_restored']
    for key,value in original.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key],atol=0,rtol=0)
    long=dict(cfg,strict_fp32=True,geometry_object_source='gt_dense2048',
        local_subset_scope=EXPANDED_SCOPE,execution_campaign=OVERNIGHT_CAMPAIGN,
        updates=30000,max_seconds=43200,batch_size=96,artifact_budget_bytes=8*2**30)
    recipe=dict(local_subset_scope=EXPANDED_SCOPE)
    validate_local_execution(long,recipe,3,30000,96)
    with pytest.raises(ValueError):validate_local_execution(dict(long,execution_campaign=None),recipe,3,30000,96)
    with pytest.raises(ValueError):validate_local_execution(dict(long,flow_matching=False),recipe,3,30000,96)
