import numpy as np
import torch
from wam_rl_policy_v2.inspire_shadow import (
    CURRENT_DIM,INPUT_DIM,CURRENT_SCHEMA,CONTRACT,expanded_model,initialize_scheduled,
    current_state,measured_from_legacy,object_state,tau_world,WAM_TO_NATIVE,
)


def test_shared_shadow_capacity_and_action_normalization_expansion():
    model=expanded_model();scheduled=expanded_model(True)
    linear=[m for m in model.net if isinstance(m,torch.nn.Linear)]
    assert [m.out_features for m in linear]==[1024,1024,512,512,12]
    assert sum(isinstance(m,torch.nn.ELU) for m in model.net)==4
    assert CURRENT_DIM==312 and INPUT_DIM==577 and not CONTRACT['direct_hand_reference']
    with torch.no_grad():
        model.net[-1].weight.normal_(std=.01);model.x_mean.normal_();model.x_scale.uniform_(.1,1);model.y_mean.normal_();model.y_scale.uniform_(.1,1)
        scheduled.x_scale[CURRENT_DIM:-1]=.05
    initialize_scheduled(scheduled,dict(schema=CURRENT_SCHEMA,model=model.state_dict()))
    x=torch.randn(5,CURRENT_DIM)*10;tail=torch.randn(5,265)
    torch.testing.assert_close(scheduled(torch.cat((x,tail),1)),model(x),atol=2e-6,rtol=2e-6)
    assert torch.equal(scheduled.x_scale[CURRENT_DIM:-1],torch.full((264,),.05))
    scheduled(torch.cat((x,tail),1)).square().mean().backward()
    assert scheduled.net[0].weight.grad[:,CURRENT_DIM:].abs().sum()>0


def test_actor_input_ignores_all_reference_error_fields():
    rng=np.random.default_rng(4);legacy=rng.normal(size=(2,721)).astype(np.float32)
    changed=legacy.copy();changed[:,:9]=10000;changed[:,257:]=np.nan
    q=np.zeros((2,18));dq=np.ones((2,18));pose=np.repeat(np.eye(4)[None],2,0)
    state=object_state(pose,np.zeros((2,6)));goal=np.array([[0,0,.1],[0,0,.2]]);previous=np.zeros((2,12))
    before=current_state(q,dq,measured_from_legacy(legacy),state,goal,previous)
    after=current_state(q,dq,measured_from_legacy(changed),state,goal,previous)
    np.testing.assert_array_equal(before,after)
    np.testing.assert_array_equal(after[:,-15:-12],goal.astype(np.float32))


def test_native_tau_is_30hz_world_displacement_not_absolute_position():
    prediction=np.zeros((2,24,2,11,3));prediction[:,:8,0,:,0]=np.arange(8)[:,None]+np.arange(11)[None,:]*.1
    rotations=np.repeat(np.eye(3)[None],2,0);rotations[1]=[[0,-1,0],[1,0,0],[0,0,1]]
    tau=tau_world(prediction,rotations)
    np.testing.assert_allclose(tau[0,:,:,0],prediction[0,:8,0][:,WAM_TO_NATIVE,0],atol=1e-6)
    np.testing.assert_allclose(tau[1,:,:,1],tau[0,:,:,0],atol=1e-6)
    np.testing.assert_array_equal(tau[1,:,:,0],0)
