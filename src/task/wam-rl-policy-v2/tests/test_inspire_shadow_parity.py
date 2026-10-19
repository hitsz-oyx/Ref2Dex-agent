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


def test_actual_official_shadow_ppo_recipe_and_bc_mean(tmp_path):
    import ast
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from gym import spaces
    from wam_rl_policy_v2.inspire_ppo_bridge import make_shadow_ppo
    from wam_rl_policy_v2.inspire_shadow import SCHEMA
    torch.set_num_threads(2)
    model=expanded_model(True)
    with torch.no_grad():model.y_mean.copy_(torch.linspace(-.3,.3,12))
    env=SimpleNamespace(observation_space=spaces.Box(-np.inf,np.inf,shape=(577,),dtype=np.float32),
        state_space=spaces.Box(-np.inf,np.inf,shape=(0,),dtype=np.float32),action_space=spaces.Box(-1,1,shape=(12,),dtype=np.float32),
        runtime=SimpleNamespace(device=torch.device('cpu')),task=SimpleNamespace(cfg={}),rl_device='cpu',num_envs=2,num_states=0)
    root=Path(__file__).resolve().parents[4]
    ppo,cfg=make_shadow_ppo(root,tmp_path,env,dict(schema=SCHEMA,model=model.state_dict()),246)
    try:
        assert (ppo.num_transitions_per_env,ppo.num_learning_epochs,ppo.num_mini_batches)==(8,5,4)
        assert (ppo.gamma,ppo.lam,ppo.clip_param,ppo.desired_kl)==(.96,.95,.2,.016)
        assert cfg['policy']['pi_hid_sizes']==cfg['policy']['vf_hid_sizes']==[1024,1024,512,512]
        x=torch.randn(2,577)
        torch.testing.assert_close(ppo.actor_critic.act_inference(x),model(x))
        actions,logp,values,mean,sigma=ppo.actor_critic.act(x,None)
        replay=ppo.actor_critic.evaluate(x,None,actions)
        torch.testing.assert_close(logp,replay[0]);torch.testing.assert_close(mean,replay[3])
        torch.testing.assert_close(ppo.actor_critic.log_std.exp().square(),torch.full((12,),.64))
        def methods(path):
            tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PPO')
            return {n.name:ast.dump(n) for n in cls.body if isinstance(n,ast.FunctionDef)}
        original=methods(root/'third_party/UniDexGrasp2-official-ppo/dexgrasp/algorithms/rl/ppo/ppo.py')
        actual=methods(tmp_path/'runtime-official/dexgrasp/algorithms/rl/ppo/ppo.py')
        assert original['run']==actual['run']
        assert original.get('compute_returns')==actual.get('compute_returns')
    finally:ppo.writer.close()


def test_complete_goal_cannot_be_paid_before_ordered_path_completion():
    from wam_rl_policy_v2.inspire_ppo_bridge import shadow_path_reward
    objects=np.zeros((1,13));hands=np.zeros((1,11,3));actions=np.zeros((1,12))
    decision=dict(reference_error_m=0.,confirmed_progress=0.,complete=False)
    pending,_=shadow_path_reward(objects,hands,actions,np.zeros(1),[decision],np.zeros(1))
    complete=dict(decision,confirmed_progress=1.,complete=True)
    finished,_=shadow_path_reward(objects,hands,actions,np.zeros(1),[complete],np.zeros(1))
    assert finished[0]>pending[0]+2.9


def test_zero_condition_bc_elision_preserves_actions_and_workspace_termination():
    from wam_rl_policy_v2.inspire_shadow import BatchedInspireInputs
    class Runtime:
        device='cpu'
        config=dict(voxel_origin_m=[-2]*3,voxel_m=.01)
        calls=0
        def predict(self,batch):
            self.calls+=1
            return dict(trajectory=torch.full((len(batch['start_hand']),24,2,11,3),.01))
    surface=np.random.default_rng(246).normal(size=(512,3)).astype(np.float32)*.02
    normals=np.zeros_like(surface);normals[:,2]=1
    geometry=(surface,normals,surface[:32],surface[:128],normals[:128])
    full=Runtime();elided=Runtime();full.geometry=elided.geometry=[geometry]*2
    start=np.repeat(np.eye(4)[None],2,axis=0);goal=start.copy();goal[:,2,3]=.2
    references=[np.stack((s,g)) for s,g in zip(start,goal)]
    a=BatchedInspireInputs(full,references);b=BatchedInspireInputs(elided,references);b.predict_trajectory=False
    model=expanded_model(True)
    with torch.no_grad():
        model.net[0].weight[:,CURRENT_DIM:]=0
        model.net[-1].weight.normal_(std=.01)
    for tick in range(5):
        hand=np.zeros((2,11,3))
        if tick==4:hand[1]=-20
        first=a.observation(np.full((2,CURRENT_DIM),.02),start,hand,np.full(2,tick))
        second=b.observation(np.full((2,CURRENT_DIM),.02),start,hand,np.full(2,tick))
        np.testing.assert_array_equal(first['workspace_invalid'],second['workspace_invalid'])
        torch.testing.assert_close(model(torch.tensor(first['observation'])),model(torch.tensor(second['observation'])),rtol=0,atol=0)
        if tick==3:
            assert first['observation'][:,-1].tolist()==second['observation'][:,-1].tolist()==[1,1]
        if tick==4:assert second['workspace_invalid'].tolist()==[False,True]
    assert full.calls==2 and elided.calls==0
