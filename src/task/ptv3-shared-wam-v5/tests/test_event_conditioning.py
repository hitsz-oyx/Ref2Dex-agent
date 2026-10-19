"""Event contract and full WAM forward equality at zero-residual migration."""
import copy
import pytest
import torch
from test_v5_contract import Geometry,config,stats,batch
from ptv3_shared_wam_v5.model import SharedWAMV5
from ptv3_shared_wam_v5.events import EVENT_ENCODING,event_packet,event_features,initialize_event_model


def condition(b,kind=1,delta=6.,valid=True):
    b.update(event_type=torch.tensor([kind]),event_delta_steps=torch.tensor([delta]),
             event_valid=torch.tensor([valid]),event_hz=torch.tensor([30]))
    return b


def test_full_model_event_expansion_preserves_all_old_predictions():
    c=config();base=SharedWAMV5(stats(),c,Geometry(8)).eval()
    target=SharedWAMV5(stats(),dict(c,event_conditioning=True,event_encoding=EVENT_ENCODING),Geometry(8)).eval()
    migration=initialize_event_model(target,base.state_dict())
    assert not migration['optimizer_restored']
    b=batch()
    with torch.no_grad():expected=base(b,return_shared=True)
    for kind,delta,valid in [(1,6.,True),(2,100.,True),(0,0.,True),(999,float('nan'),False)]:
        with torch.no_grad():actual=target(condition(copy.deepcopy(b),kind,delta,valid),return_shared=True)
        for k in ('trajectory','scene','interaction','shared_tokens'):
            torch.testing.assert_close(actual[k],expected[k],rtol=0,atol=0)


def test_event_branch_gets_gradients_and_learns_condition_dependence():
    c=dict(config(),event_conditioning=True,event_encoding=EVENT_ENCODING)
    model=SharedWAMV5(stats(),c,Geometry(8)).eval();b=condition(batch())
    prediction=model(b);prediction['trajectory'].sum().backward()
    grad=model.event_goal.encoder[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
    with torch.no_grad():model.event_goal.encoder[-1].weight.add_(grad,alpha=-.01)
    with torch.no_grad():
        a=model(b)['trajectory'];b['event_type'].fill_(2);z=model(b)['trajectory']
    assert not torch.equal(a,z)


def test_events_outside_horizon_are_not_none_and_unknown_padding_is_clean():
    b=condition({},2,100.)
    features,valid=event_features(b)
    assert features[0,2]==1 and features[0,4]==1 and valid.item()
    assert not event_features(condition({},999,float('nan'),False))[0].any()
    with pytest.raises(ValueError,match='30Hz'):
        b['event_hz'].fill_(60);event_features(b)
    with pytest.raises(ValueError):event_packet('CONTACT',-1,True)
    assert event_packet('RELEASE',5,True)['event_type']==2


def test_event_variant_requires_fields_and_rejects_partial_base_state():
    m=SharedWAMV5(stats(),dict(config(),event_conditioning=True,event_encoding=EVENT_ENCODING),Geometry(8))
    with pytest.raises(ValueError,match='explicit type'):m(batch())
    with pytest.raises(ValueError,match='every original'):initialize_event_model(m,{})
    b=condition(batch());b['xyz']=b['xyz'].repeat(2,1,1)
    with pytest.raises(ValueError,match='no broadcast'):event_features(b)
