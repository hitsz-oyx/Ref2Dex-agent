import torch
import pytest
from types import SimpleNamespace
from ptv3_shared_wam_v2.capacity_actor import CapacityActor
from ptv3_shared_wam_v2.shared_actor import SharedQueryActor


def test_goal_controls_share_initial_weights_and_original_zero_start():
    torch.manual_seed(221)
    off = CapacityActor(128, False)
    torch.manual_seed(221)
    on = CapacityActor(128, True)
    assert all(torch.equal(v,on.state_dict()[k]) for k,v in off.state_dict().items())
    old = SharedQueryActor('shared', current_observation='delta0')
    old.load_state_dict({k:v for k,v in off.state_dict().items() if k in old.state_dict()}, strict=True)
    state=torch.randn(2,87);current=torch.randn(2,721);z=torch.randn(2,5,768)
    valid=torch.ones(2,5,dtype=torch.bool);goal=torch.randn(2,3)
    assert torch.equal(off(state,z,valid,current,goal),old(state,shared=z,shared_valid=valid,current_obs=current))
    assert torch.equal(off(state,z,valid,current,goal),on(state,z,valid,current,goal))


def test_explicit_goal_and_current_bypass_are_effective_and_masked():
    model=CapacityActor(128,True,True).eval()
    with torch.no_grad():
        model.head.weight.normal_()
        model.goal_encoder[-1].weight.normal_()
        model.bypass.weight.fill_(.001)
    state=torch.randn(2,87);current=torch.randn(2,721);goal=torch.randn(2,3)
    z=torch.randn(2,5,768);valid=torch.ones(2,5,dtype=torch.bool);valid[:,-1]=False
    first=model(state,z,valid,current,goal)
    assert not torch.allclose(first,model(state,z,valid,current,goal+1))
    z[:,-1]=float('nan')
    assert torch.allclose(first,model(state,z,valid,current,goal))
    model.bypass_enabled=False
    base=model(state,z,valid,current,goal)
    direct=model.bypass(current).reshape(2,8,15)*model.action_scale
    assert torch.allclose(first-base,direct,atol=1e-4)


def test_native_capacity_bridge_requires_explicit_own_goal():
    from ptv3_shared_wam_v2.native_execution import predict_shared_action
    class Forecaster:
        def __call__(self,batch,return_shared=False):
            return dict(trajectory=torch.zeros(2,24,2,11,3),shared_tokens=torch.ones(2,5,768),
                        shared_valid=torch.ones(2,5,dtype=torch.bool))
    actor=CapacityActor(128,True).eval()
    with torch.no_grad():
        actor.head.weight.normal_()
        actor.goal_encoder[-1].weight.normal_()
    model=SimpleNamespace(action_interface='shared',capacity_probe={},forecaster=Forecaster(),retargeter=actor)
    batch=dict(start_hand_valid=torch.ones(2,2,11,dtype=torch.bool))
    state=torch.randn(2,87);current=torch.randn(2,721)
    with pytest.raises(ValueError,match='planned t\\+24'):
        predict_shared_action(model,batch,state,current)
    a,tau=predict_shared_action(model,batch,state,current,explicit_goal=torch.zeros(2,3))
    b,_=predict_shared_action(model,batch,state,current,explicit_goal=torch.ones(2,3))
    assert a.shape==(2,8,15) and tau.shape==(2,24,33) and not torch.allclose(a,b)
