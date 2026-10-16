import torch
import pytest
from wam_rl_policy.ppo import Actor,Critic,advantages,apply_residual,observation,update


@pytest.mark.parametrize('minibatches',[1,2])
def test_scaled_value_keeps_physical_bootstrap_gae_and_normalized_optimizer_units(minibatches):
    import copy
    torch.manual_seed(184)
    actor=Actor().double();critic=Critic().double();scaled_actor=copy.deepcopy(actor);scaled_critic=copy.deepcopy(critic)
    scaled_critic.value_scale=100.
    obs=torch.randn(4,4,327,dtype=torch.float64);obs[:,:,-24:]=0;next_obs=torch.randn(4,327,dtype=torch.float64)
    with torch.no_grad():
        dist=actor.distribution(obs);latent=dist.sample()
        batch=dict(observation=obs,latent=latent,logp=dist.log_prob(latent).sum(-1),mean=dist.loc,std=dist.scale,
            value=critic(obs),reward=torch.randn(4,4,dtype=torch.float64),done=torch.zeros(4,4,dtype=torch.bool),bootstrap=critic(next_obs))
        scaled=dict(batch,value=scaled_critic(obs),reward=batch['reward']*100.,bootstrap=scaled_critic(next_obs))
    adv,target=advantages(batch['reward'],batch['value'],batch['done'],batch['bootstrap'])
    scaled_adv,scaled_target=advantages(scaled['reward'],scaled['value'],scaled['done'],scaled['bootstrap'])
    torch.testing.assert_close(scaled_adv,adv*100.,rtol=2e-6,atol=2e-5)
    torch.testing.assert_close(scaled_target,target*100.,rtol=2e-6,atol=2e-5)
    results=[]
    for policy,value,data in [(actor,critic,batch),(scaled_actor,scaled_critic,scaled)]:
        results.append(update(policy,value,data,torch.optim.Adam(policy.parameters(),lr=1e-5),
            torch.optim.Adam(value.parameters(),lr=3e-4),epochs=2,minibatches=minibatches,
            shuffle_generator=torch.Generator().manual_seed(45)))
    for original,other in [(actor,scaled_actor),(critic,scaled_critic)]:
        for key,val in original.state_dict().items():torch.testing.assert_close(val,other.state_dict()[key],rtol=1e-5,atol=1e-6)
    assert results[1]['return_mean']==pytest.approx(results[0]['return_mean']*100.,rel=2e-5)
    assert results[1]['normalized_value_loss']==pytest.approx(results[0]['normalized_value_loss'],rel=2e-5)


def test_shadow_zero_residual_and_clipping_contract():
    torch.manual_seed(1)
    base=torch.linspace(-1,1,48).reshape(2,24)
    obs=observation(torch.zeros(2,300),torch.ones(2,3)*.1,base)
    actor=Actor()
    assert obs.shape==(2,327)
    assert torch.equal(obs[:,-24:],base)
    applied,intended,delta=apply_residual(base,actor.distribution(obs).loc)
    assert torch.equal(applied,base) and torch.equal(intended,base) and not delta.any()
    applied,intended,delta=apply_residual(base,torch.ones_like(base)*100)
    assert torch.allclose(intended,base+.15) and applied.max()==1
    assert torch.equal(base,obs[:,-24:])  # simulator mutation must never affect base


def test_gae_respects_real_terminal_and_final_bootstrap():
    reward=torch.tensor([[1.,1.],[2.,2.]])
    value=torch.zeros_like(reward)
    done=torch.tensor([[False,True],[True,False]])
    adv,ret=advantages(reward,value,done,torch.tensor([10.,10.]),gamma=.9,lam=1.)
    assert torch.allclose(ret,torch.tensor([[2.8,1.],[2.,11.]]))
    assert torch.equal(adv,ret)


@pytest.mark.parametrize('obs_dim,action_kind,network_kind,critic_activation',[(327,'residual','small','tanh'),(539,'residual','small','tanh'),
    (2631,'residual','small','tanh'),(327,'residual','large','tanh'),(327,'reference-gaussian','large','tanh'),
    (540,'reference-gaussian','large','elu')])
def test_real_ppo_update_replays_behavior_and_changes_only_optimized_heads(obs_dim,action_kind,network_kind,critic_activation):
    torch.manual_seed(2)
    actor=Actor(-4.3971199849 if action_kind=='reference-gaussian' else -2.5,obs_dim=obs_dim,
                network_kind=network_kind,action_kind=action_kind)
    critic=Critic(obs_dim=obs_dim,network_kind=network_kind,activation=critic_activation)
    obs=torch.randn(4,3,obs_dim);obs[:,:,-24:]=0
    with torch.no_grad():
        dist=actor.distribution(obs);latent=dist.sample()
        batch=dict(observation=obs,latent=latent,logp=dist.log_prob(latent).sum(-1),
            mean=dist.loc,std=dist.scale,value=critic(obs),reward=torch.randn(4,3),
            done=torch.zeros(4,3,dtype=torch.bool),bootstrap=torch.zeros(3))
        batch['done'][-1]=True
    before=actor.net[-1].weight.detach().clone()
    stats=update(actor,critic,batch,torch.optim.Adam(actor.parameters(),lr=1e-4),
                 torch.optim.Adam(critic.parameters(),lr=3e-4))
    assert stats['behavior_logprob_error']<1e-4 and 1<=stats['actor_steps']<=4
    assert stats['joint_kl']<=.02 and torch.isfinite(torch.tensor(stats['value_loss']))
    assert not torch.equal(before,actor.net[-1].weight)


def test_feature_observation_is_detached_and_preserves_native_goal_and_reference():
    features=torch.randn(3,512,requires_grad=True);base=torch.randn(3,24)
    obs=observation(torch.zeros(3,300),torch.ones(3,3)*.2,base,features)
    assert obs.shape==(3,539) and not obs.requires_grad
    assert torch.equal(obs[:,-24:],base) and torch.equal(obs[:,512:515],torch.ones(3,3)*2)
    torch.testing.assert_close(obs[:,:512].mean(-1),torch.zeros(3),atol=1e-6,rtol=0)


def test_exploration_arm_changes_only_initial_latent_scale():
    torch.manual_seed(31);reference=Actor()
    torch.manual_seed(31);strong=Actor(-1.5)
    x=torch.randn(7,327)
    torch.testing.assert_close(reference.distribution(x).loc,strong.distribution(x).loc,rtol=0,atol=0)
    torch.testing.assert_close(strong.distribution(x).scale/reference.distribution(x).scale,torch.full((7,24),torch.exp(torch.tensor(1.)).item()))
    for key,value in reference.state_dict().items():
        if key!='log_std':assert torch.equal(value,strong.state_dict()[key])


def test_segmented_rollout_bootstraps_actual_next_state_and_replays_new_policy(monkeypatch):
    from wam_rl_policy import ppo
    torch.manual_seed(61);actor=Actor();critic=Critic()
    actor_optimizer=torch.optim.Adam(actor.parameters(),lr=1e-4)
    critic_optimizer=torch.optim.Adam(critic.parameters(),lr=3e-4)
    observed=[];original=ppo.update
    def capture(*args,**kwargs):
        observed.append(args[2]['bootstrap'].clone())
        return original(*args,**kwargs)
    monkeypatch.setattr(ppo,'update',capture)
    storage={key:[] for key in ('observation','latent','mean','std','logp','value','reward','done')}
    def collect(count,terminal):
        with torch.no_grad():
            for t in range(count):
                obs=torch.randn(3,327);obs[:,-24:]=0;dist=actor.distribution(obs);latent=dist.sample()
                values=dict(observation=obs,latent=latent,mean=dist.loc,std=dist.scale,logp=dist.log_prob(latent).sum(-1),
                    value=critic(obs),reward=torch.randn(3),done=torch.full((3,),terminal and t==count-1))
                for key,value in values.items():storage[key].append(value)
    collect(4,False)
    with pytest.raises(ValueError,match='real terminal'):ppo.update_rollout(actor,critic,storage,actor_optimizer,critic_optimizer)
    next_obs=torch.randn(3,327);expected=critic(next_obs).detach().clone()
    first=ppo.update_rollout(actor,critic,storage,actor_optimizer,critic_optimizer,next_obs)
    assert torch.equal(observed[0],expected) and all(not values for values in storage.values())
    collect(3,True)
    second=ppo.update_rollout(actor,critic,storage,actor_optimizer,critic_optimizer)
    assert torch.equal(observed[1],torch.zeros(3)) and all(not values for values in storage.values())
    assert first['rollout_steps']==4 and second['rollout_steps']==3
    assert first['behavior_logprob_error']<1e-4 and second['behavior_logprob_error']<1e-4


def test_full_episode_rollout_retains_previous_optimizer_result():
    import copy
    from wam_rl_policy.ppo import update_rollout
    torch.manual_seed(71);actor=Actor();critic=Critic();other_actor=copy.deepcopy(actor);other_critic=copy.deepcopy(critic)
    obs=torch.randn(5,3,327);obs[:,:,-24:]=0
    with torch.no_grad():
        dist=actor.distribution(obs);latent=dist.sample()
        batch=dict(observation=obs,latent=latent,logp=dist.log_prob(latent).sum(-1),mean=dist.loc,std=dist.scale,
            value=critic(obs),reward=torch.randn(5,3),done=torch.zeros(5,3,dtype=torch.bool),bootstrap=torch.zeros(3))
        batch['done'][-1]=True
    update(actor,critic,batch,torch.optim.Adam(actor.parameters(),lr=1e-4),torch.optim.Adam(critic.parameters(),lr=3e-4))
    storage={key:list(value.unbind(0)) for key,value in batch.items() if key!='bootstrap'}
    update_rollout(other_actor,other_critic,storage,torch.optim.Adam(other_actor.parameters(),lr=1e-4),torch.optim.Adam(other_critic.parameters(),lr=3e-4))
    for original,other in [(actor,other_actor),(critic,other_critic)]:
        for key,value in original.state_dict().items():assert torch.equal(value,other.state_dict()[key])


def test_kl_rejection_reports_retained_policy_after_rollback():
    torch.manual_seed(91);actor=Actor();critic=Critic();obs=torch.randn(4,3,327);obs[:,:,-24:]=0
    with torch.no_grad():
        dist=actor.distribution(obs);latent=dist.sample()
        batch=dict(observation=obs,latent=latent,logp=dist.log_prob(latent).sum(-1),mean=dist.loc,std=dist.scale,
            value=critic(obs),reward=torch.randn(4,3),done=torch.zeros(4,3,dtype=torch.bool),bootstrap=torch.zeros(3))
        batch['done'][-1]=True
    before={key:value.clone() for key,value in actor.state_dict().items()}
    stats=update(actor,critic,batch,torch.optim.Adam(actor.parameters(),lr=1000),torch.optim.Adam(critic.parameters(),lr=3e-4))
    assert stats['actor_steps']==0 and stats['joint_kl']==0
    for key,value in actor.state_dict().items():assert torch.equal(value,before[key])


def test_wam_observation_keeps_full_state_goal_and_reference():
    state=torch.randn(3,300);goal=torch.randn(3,3);base=torch.randn(3,24);z=torch.randn(3,2304,requires_grad=True)
    ordinary=observation(state,goal,base);with_z=observation(state,goal,base,wam_features=z)
    assert with_z.shape==(3,2631) and not with_z.requires_grad
    assert torch.equal(with_z[:,:300],ordinary[:,:300]) and torch.equal(with_z[:,-27:],ordinary[:,-27:])
    zero=observation(state,goal,base,wam_features=torch.zeros_like(z))
    assert not zero[:,300:2604].any()


def test_reference_gaussian_starts_at_bc_and_can_exceed_residual_limit():
    from wam_rl_policy.ppo import apply_policy_action
    obs=torch.randn(3,327);base=torch.ones(3,24)*.2;obs[:,-24:]=base
    actor=Actor(-4.3971199849,network_kind='large',action_kind='reference-gaussian')
    assert torch.equal(actor.distribution(obs).loc,base)
    assert torch.equal(apply_policy_action(base,actor.distribution(obs).loc,'reference-gaussian')[0],base)
    action,intended,_=apply_policy_action(base,base+.5,'reference-gaussian')
    torch.testing.assert_close(action-base,torch.full_like(base,.5))
    assert torch.equal(action,intended)
    residual=apply_policy_action(base,torch.ones_like(base)*100,'residual')[0]
    assert (residual-base).max()<.151


def test_minibatch_shuffle_does_not_consume_environment_rng():
    torch.manual_seed(99);actor=Actor();critic=Critic();obs=torch.randn(4,4,327);obs[:,:,-24:]=0
    with torch.no_grad():
        dist=actor.distribution(obs);latent=dist.sample()
        batch=dict(observation=obs,latent=latent,logp=dist.log_prob(latent).sum(-1),mean=dist.loc,std=dist.scale,
            value=critic(obs),reward=torch.randn(4,4,dtype=torch.float64),done=torch.zeros(4,4,dtype=torch.bool),bootstrap=torch.zeros(4))
        batch['done'][-1]=True
    environment_rng=torch.get_rng_state().clone();shuffle=torch.Generator().manual_seed(1247);old_shuffle=shuffle.get_state().clone()
    stats=update(actor,critic,batch,torch.optim.Adam(actor.parameters(),lr=1e-4),torch.optim.Adam(critic.parameters(),lr=3e-4),
        minibatches=4,epochs=5,shuffle_generator=shuffle)
    assert torch.equal(environment_rng,torch.get_rng_state()) and not torch.equal(old_shuffle,shuffle.get_state())
    assert stats['critic_steps']==20 and 0<=stats['actor_steps']<=20 and stats['joint_kl']<=.02


def test_episode_clock_preserves_reference_and_only_encodes_known_time():
    from wam_rl_policy.ppo import append_episode_clock
    obs=torch.randn(3,539)
    assert append_episode_clock(obs,1,199) is obs
    zero=append_episode_clock(obs,1,199,'zero');first=append_episode_clock(obs,1,199,'remaining')
    last=append_episode_clock(obs,199,199,'remaining')
    assert zero.shape==(3,540) and torch.equal(zero[:,-24:],obs[:,-24:])
    assert torch.equal(first[:,:-25],obs[:,:-24]) and not zero[:,-25].any()
    assert (first[:,-25]==1).all() and (last[:,-25]==1/199).all()
