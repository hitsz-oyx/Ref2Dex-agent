import torch
from wam_rl_policy.ppo import Actor,Critic,advantages,apply_residual,observation,update


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


def test_real_ppo_update_replays_behavior_and_changes_only_optimized_heads():
    torch.manual_seed(2)
    actor=Actor();critic=Critic()
    obs=torch.randn(4,3,327);obs[:,:,-24:]=0
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


def test_exploration_arm_changes_only_initial_latent_scale():
    torch.manual_seed(31);reference=Actor()
    torch.manual_seed(31);strong=Actor(-1.5)
    x=torch.randn(7,327)
    torch.testing.assert_close(reference.distribution(x).loc,strong.distribution(x).loc,rtol=0,atol=0)
    torch.testing.assert_close(strong.distribution(x).scale/reference.distribution(x).scale,torch.full((7,24),torch.exp(torch.tensor(1.)).item()))
    for key,value in reference.state_dict().items():
        if key!='log_std':assert torch.equal(value,strong.state_dict()[key])
