"""Shadow Stage-0 PPO: frozen BC plus bounded 24D canonical residuals."""
import copy
import torch
from torch import nn

SCHEMA = 'ref2dex.shadow-bc-residual-ppo.v1'
OBS_DIM = 327  # native current300, explicit relative goal3, BC reference24
ACTION_DIM = 24


def observation(current, relative_goal, base):
    if current.shape != (len(current),300) or relative_goal.shape != (len(current),3) or base.shape != (len(current),24):
        raise ValueError('Shadow current300/relative goal3/reference24 required')
    result=torch.cat((current.clamp(-5,5),relative_goal/.1,base),-1)
    if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite RL observation')
    return result


def apply_residual(base, latent, alpha=.15):
    if base.shape != latent.shape or base.shape[-1] != ACTION_DIM or not 0<=alpha<=1:
        raise ValueError('Matched canonical24D actions and bounded alpha required')
    delta=torch.tanh(latent)
    intended=base+alpha*delta
    return intended.clamp(-1,1),intended,delta


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(OBS_DIM,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,ACTION_DIM))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.log_std=nn.Parameter(torch.full((ACTION_DIM,),-2.5))

    def distribution(self,obs):
        return torch.distributions.Normal(self.net(obs),self.log_std.clamp(-5,-1.2).exp())


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(OBS_DIM,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,1))

    def forward(self,obs):return self.net(obs).squeeze(-1)


def advantages(rewards,values,dones,bootstrap,gamma=.99,lam=.95):
    if rewards.shape != values.shape or rewards.shape != dones.shape or bootstrap.shape != rewards.shape[1:]:
        raise ValueError('T,N rollout and actual final-state bootstrap required')
    running=torch.zeros_like(bootstrap);adv=torch.empty_like(rewards);next_value=bootstrap
    for t in range(len(rewards)-1,-1,-1):
        live=(~dones[t]).float()
        delta=rewards[t]+gamma*live*next_value-values[t]
        running=delta+gamma*lam*live*running
        adv[t]=running;next_value=values[t]
    return adv,adv+values


def update(actor,critic,batch,actor_optimizer,critic_optimizer,bc_weight=.1,target_kl=.02,epochs=4):
    adv,returns=advantages(batch['reward'],batch['value'],batch['done'],batch['bootstrap'])
    obs=batch['observation'].flatten(0,1);latent=batch['latent'].flatten(0,1)
    old_logp=batch['logp'].flatten();old=torch.distributions.Normal(batch['mean'].flatten(0,1),batch['std'].flatten(0,1))
    adv=adv.flatten();adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6);returns=returns.flatten()
    with torch.no_grad():replay=float((actor.distribution(obs).log_prob(latent).sum(-1)-old_logp).abs().max())
    if replay>1e-4:raise ValueError('Behavior latent log probability replay drift: '+str(replay))
    steps=0;kl=0.;reg=0.
    for _ in range(epochs):
        dist=actor.distribution(obs);logp=dist.log_prob(latent).sum(-1)
        ratio=(logp-old_logp).exp()
        surrogate=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
        # Penalize executed deterministic canonical deviation, not latent units.
        base=obs[:,-24:];reference=(base+.15*torch.tanh(dist.loc)).clamp(-1,1)
        reg=(reference-base).square().mean();loss=surrogate+bc_weight*reg
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite actor objective')
        actor_optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),.5)
        before=copy.deepcopy(actor.state_dict());optim_before=copy.deepcopy(actor_optimizer.state_dict())
        accepted=False
        for _retry in range(4):
            actor_optimizer.step()
            with torch.no_grad():kl=float(torch.distributions.kl_divergence(old,actor.distribution(obs)).sum(-1).mean())
            if kl<=target_kl:accepted=True;break
            rates=[group['lr']*.25 for group in actor_optimizer.param_groups]
            actor.load_state_dict(before);actor_optimizer.load_state_dict(optim_before)
            for group,rate in zip(actor_optimizer.param_groups,rates):group['lr']=rate
        if not accepted:break
        steps+=1
    for _ in range(epochs):
        value_loss=(critic(obs)-returns).square().mean()
        if not torch.isfinite(value_loss):raise FloatingPointError('Nonfinite critic objective')
        critic_optimizer.zero_grad(set_to_none=True);value_loss.backward();nn.utils.clip_grad_norm_(critic.parameters(),1.)
        critic_optimizer.step()
    return dict(actor_steps=steps,joint_kl=kl,bc_regularization=float(reg),value_loss=float(value_loss),
                behavior_logprob_error=replay,actor_lr=actor_optimizer.param_groups[0]['lr'],
                return_mean=float(returns.mean()),advantage_std=float(adv.std(unbiased=False)))
