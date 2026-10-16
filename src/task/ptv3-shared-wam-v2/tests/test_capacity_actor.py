import torch
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
