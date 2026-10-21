import copy
import torch
import pytest
from test_geometry_consistency import packet, geometry_config, total
from ptv3_shared_wam_v6.geometry_consistency import (
    GT_OBJECT_ENCODING, geometry_components, validate_geometry_config,
)
from ptv3_shared_wam_v6.geometry_gt_contact import proximity_bce


def config():
    return dict(geometry_config(), geometry_object_source='gt_dense2048',
        geometry_encoding=GT_OBJECT_ENCODING, geometry_queries=2048,
        geometry_contact_radius_m=.003, geometry_contact_falloff_m=.010,
        geometry_contact_smoothing_m=.001)


def dense_packet():
    b,p=packet()
    for key in ('interaction_points','interaction_normals'):
        if key in b:b[key]=b[key].repeat(1,1,64,1)
    for key in ('contact_target','part_target','interaction_valid'):
        b[key]=b[key].repeat(1,1,1,64)
    p['interaction_prediction_valid']=torch.ones_like(b['interaction_valid'])
    b['geometry_object_future_gt']=b['interaction_points'][:,:,None].expand(-1,-1,24,-1,-1).clone()
    return b,p


def test_dense_gt_contact_has_hand_gradients_and_no_direct_scene_or_gt_gradients():
    b,p=dense_packet();p['trajectory']=torch.full_like(p['trajectory'],.5,requires_grad=True)
    b['geometry_object_future_gt'].requires_grad_()
    terms,_=geometry_components(p,b,config());loss=total(terms)
    h,o,g=torch.autograd.grad(loss,(p['trajectory'],p['scene'],b['geometry_object_future_gt']),allow_unused=True)
    assert torch.isfinite(loss) and torch.isfinite(h).all() and h.sum()>0
    assert o is None and g is None
    shifted=dict(p,scene=p['scene']+10)
    changed,_=geometry_components(shifted,b,config())
    torch.testing.assert_close(total(changed),loss,atol=0,rtol=0)


def test_object_does_not_cancel_and_chunking_preserves_complete_field_and_gradients():
    b,p=dense_packet();p['trajectory']=torch.full_like(p['trajectory'],.04,requires_grad=True)
    c=config();c['geometry_query_chunk']=32
    first,_=geometry_components(p,b,c);a=total(first);ga=torch.autograd.grad(a,p['trajectory'])[0]
    c['geometry_query_chunk']=64
    second,_=geometry_components(p,b,c);z=total(second);gz=torch.autograd.grad(z,p['trajectory'])[0]
    torch.testing.assert_close(a,z);torch.testing.assert_close(ga,gz)
    assert first['geometry_contact'][1]==3*2048
    b['geometry_object_future_gt']=b['geometry_object_future_gt']+.15
    changed,_=geometry_components(p,b,c)
    assert abs(float(total(changed)-a))>.01


def test_stable_contact_bce_has_no_distant_positive_gradient_cutoff():
    distance=torch.tensor([.01,.1,1.,10.],requires_grad=True)
    loss,prob=proximity_bce(distance,torch.ones_like(distance),config())
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(distance.grad).all()
    assert (distance.grad>90).all() and prob[-1]==0
    near=torch.tensor([0.,.002,.01,.03],requires_grad=True)
    labels=torch.tensor([1.,0.,.5,0.])
    loss,p=proximity_bce(near,labels,config())
    torch.testing.assert_close(loss,torch.nn.functional.binary_cross_entropy(p,labels,reduction='none'))


def test_missing_labels_are_masked_and_dense_contract_is_explicit():
    b,p=dense_packet();b['interaction_valid'].zero_();b['trajectory_valid'].zero_()
    b['contact_target'].fill_(float('nan'));b['geometry_object_future_gt'].fill_(float('nan'))
    p['trajectory']=torch.full_like(p['trajectory'],float('nan'),requires_grad=True)
    terms,_=geometry_components(p,b,config());loss=total(terms)
    assert torch.isfinite(loss) and loss==0
    loss.backward();assert torch.isfinite(p['trajectory'].grad).all()
    with pytest.raises(ValueError,match='complete2048'):
        validate_geometry_config(dict(config(),geometry_queries=128))
    del b['geometry_object_future_gt']
    with pytest.raises(ValueError,match='no32-flow fallback'):
        geometry_components(p,b,config())
