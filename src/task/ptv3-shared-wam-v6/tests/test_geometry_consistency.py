import copy
import torch
import pytest
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.geometry_consistency import (
    BONES, GEOMETRY_ENCODING, affine_query_weights, geometry_components, skin_attachment, skin_future,
)
from ptv3_shared_wam_v6.objectives import forecast_loss


def geometry_config():
    return dict(geometry_consistency=True,geometry_encoding=GEOMETRY_ENCODING,
        geometry_weight=.05,geometry_clearance_weight=.25,geometry_warmup_updates=100,
        geometry_queries=32,geometry_query_chunk=16,geometry_scale_m=.01,
        geometry_clearance_tolerance_m=.002,geometry_proxy_max_gt_gap_m=.05,
        geometry_ticks=[1,8,24])


def hand():
    j=torch.zeros(21,3)
    for base,x,y in [(1,-.035,.02),(5,-.02,.05),(9,0.,.06),(13,.02,.055),(17,.04,.045)]:
        for k in range(4):j[base+k]=torch.tensor([x,y+.018*k,0.])
    vertices=torch.stack([(j[a]+j[b])/2+torch.tensor([0.,0.,.004]) for a,b in BONES])
    return j,vertices


def packet():
    torch.manual_seed(3);j,v=hand();q=v[0]+torch.tensor([0.,0.,.002])
    queries=torch.cat((q[None],torch.rand(31,3)*.03+.15))
    scene=torch.randn(32,3)*.02+q
    confidence=torch.zeros(1,1,24,32);confidence[...,0]=1.
    b=dict(start_hand=j[None,None],start_hand_valid=torch.ones(1,1,21,dtype=torch.bool),
        dense_hand=v[None,None],dense_hand_valid=torch.ones(1,1,len(v),dtype=torch.bool),
        dense_hand_part=torch.zeros(1,1,len(v),dtype=torch.long),
        trajectory_target=torch.zeros(1,24,1,21,3),trajectory_valid=torch.ones(1,24,1,21,dtype=torch.bool),
        interaction_points=queries[None,None],scene_points=scene[None,None],
        scene_target=torch.zeros(1,1,24,32,3),scene_valid=torch.ones(1,1,24,32,dtype=torch.bool),
        object_valid=torch.ones(1,1,dtype=torch.bool),contact_target=confidence,
        part_target=torch.zeros(1,1,24,32,dtype=torch.long),interaction_valid=torch.ones(1,1,24,32,dtype=torch.bool))
    p=dict(trajectory=torch.zeros_like(b['trajectory_target'],requires_grad=True),
           scene=torch.zeros_like(b['scene_target'],requires_grad=True),
           interaction_prediction_valid=torch.ones_like(b['interaction_valid']))
    return b,p


def total(components):
    return sum(n/d.clamp_min(1) for n,d in components.values())


def test_observed_skinning_is_rigid_equivariant_and_object_flow_is_exact():
    j,v=hand();j=j[None,None];v=v[None,None]
    attach=skin_attachment(j,v,torch.ones_like(j[...,0],dtype=torch.bool),torch.ones_like(v[...,0],dtype=torch.bool))
    r=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]);t=torch.tensor([.2,-.1,.05])
    result=skin_future((j@r.T+t)[:,None],attach)
    torch.testing.assert_close(result[:,0],v@r.T+t,rtol=0,atol=1e-6)
    torch.manual_seed(7);p=torch.randn(2,1,32,3)*.03;q=torch.randn(2,1,3,12,3)*.04
    weights=affine_query_weights(p,q);flow=(p@r.T+t-p)[:,:,None].expand(-1,-1,3,-1,-1)
    torch.testing.assert_close(q+weights@flow,q@r.T+t,rtol=0,atol=3e-6)
    assert torch.isfinite(affine_query_weights(torch.zeros_like(p),q)).all()


def test_exact_gt_has_zero_loss_and_common_translation_preserves_contact():
    b,p=packet();c=geometry_config();terms,diag=geometry_components(p,b,c,torch.float64)
    assert terms['geometry_contact'][1]>0 and terms['geometry_clearance'][1]>0
    assert float(total(terms))<1e-10
    offset=torch.tensor([.2,-.1,.3])
    shifted=dict(p,trajectory=p['trajectory']+offset,scene=p['scene']+offset)
    terms,_=geometry_components(shifted,b,c)
    assert float(total(terms))<1e-8
    assert diag['geometry/contact_relative']['sum']<1e-5


def test_far_separation_pulls_both_coordinate_outputs_without_cutoff():
    b,p=packet();p['trajectory']=torch.full_like(p['trajectory'],0.);p['trajectory'][...,0]=.2;p['trajectory'].requires_grad_()
    terms,_=geometry_components(p,b,geometry_config());loss=total(terms)
    assert loss>1
    hand_grad,object_grad=torch.autograd.grad(loss,(p['trajectory'],p['scene']))
    assert torch.isfinite(hand_grad).all() and torch.isfinite(object_grad).all()
    assert hand_grad[...,0].sum()>0 and object_grad[...,0].sum()<0


def test_noncontact_does_not_pull_distant_hand_and_overlap_has_separating_gradient():
    b,p=packet();b['contact_target'].zero_()
    p['trajectory']=torch.full_like(p['trajectory'],.5,requires_grad=True)
    terms,_=geometry_components(p,b,geometry_config());assert float(total(terms))==0
    # Move a known separated surface directly onto the query; unsigned norm at
    # zero has no direction, so clearance uses the detached GT gap direction.
    j,v=hand();b['interaction_points'][:]=v[0]+torch.tensor([0.,0.,.02])
    p['trajectory']=torch.zeros_like(p['trajectory']);p['trajectory'][...,2]=.02;p['trajectory'].requires_grad_()
    terms,_=geometry_components(p,b,geometry_config());loss=total(terms)
    assert loss>0
    gradient=torch.autograd.grad(loss,p['trajectory'])[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0


def test_unknown_geometry_is_masked_without_nan_or_false_negative():
    b,p=packet();b['start_hand_valid'].zero_();b['dense_hand_valid'].zero_();b['trajectory_valid'].zero_();b['scene_valid'].zero_();b['interaction_valid'].zero_()
    b['trajectory_target'].fill_(float('nan'));b['scene_target'].fill_(float('nan'));b['contact_target'].fill_(float('nan'))
    p['trajectory']=torch.full_like(p['trajectory'],float('nan'),requires_grad=True)
    p['scene']=torch.full_like(p['scene'],float('nan'),requires_grad=True)
    terms,_=geometry_components(p,b,geometry_config());loss=total(terms)
    assert torch.isfinite(loss) and float(loss)==0
    gradients=torch.autograd.grad(loss,(p['trajectory'],p['scene']))
    assert all(torch.isfinite(g).all() and g.abs().sum()==0 for g in gradients)


def test_geometry_loss_reaches_actual_hand_and_object_readout_parameters():
    from test_v5_contract import Geometry,config,stats,batch
    from ptv3_shared_wam_v6.model import SharedWAMV6
    from ptv3_shared_wam_v6.contracts import SCHEMA
    from ptv3_shared_wam_v6.events import EVENT_ENCODING
    c=dict(config(),**geometry_config());c.update(schema=SCHEMA,event_conditioning=True,event_encoding=EVENT_ENCODING,checkpoint_contact=False)
    m=SharedWAMV6(stats(),c,Geometry(8)).train();b=batch();b['dense_hand_part'].fill_(3)
    b.update(event_type=torch.tensor([1]),event_delta_steps=torch.tensor([8.]),event_valid=torch.tensor([True]),event_hz=torch.tensor([30]))
    p=m(b);_,terms=forecast_loss(p,b,c,m.normalizer)
    gradients=torch.autograd.grad(terms['geometry_contact'],(m.hand_future.mlp[-1].weight,m.scene_future.mlp[-1].weight))
    assert all(torch.isfinite(g).all() and g.norm()>0 for g in gradients)


def test_geometry_initialization_preserves_weights_and_rejects_scale_or_data_change(tmp_path):
    from torch import nn
    from ptv3_shared_wam_v6.local_subset import initialize_geometry_weights
    from ptv3_shared_wam_v6.contracts import SCHEMA
    model=nn.Module();model.normalizer=nn.Module();model.normalizer.register_buffer('hand_std',torch.ones(24,3));model.head=nn.Linear(3,3)
    state=copy.deepcopy(model.state_dict());state['head.weight'].fill_(2.)
    saved=dict(schema=SCHEMA,world_size=3,data_identity='same',initialization_binding={'kind':'baseline'},
               config={},model=state,step=4000)
    binding=initialize_geometry_weights(model,saved,tmp_path/'latest.pt',geometry_config(),'same')
    assert torch.equal(model.head.weight,state['head.weight']) and binding['source_step']==4000
    assert binding['optimizer_restored'] is False and binding['training_step_reset']
    with pytest.raises(ValueError,match='same-data'):
        initialize_geometry_weights(model,saved,tmp_path/'latest.pt',geometry_config(),'other')
    model.normalizer.hand_std.fill_(2.)
    with pytest.raises(ValueError,match='normalization'):
        initialize_geometry_weights(model,saved,tmp_path/'latest.pt',geometry_config(),'same')
