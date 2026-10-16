"""Shadow Stage-0 PPO: frozen BC plus bounded 24D canonical residuals."""
import copy
import torch
from torch import nn
import torch.distributed as distributed

SCHEMA = 'ref2dex.shadow-bc-residual-ppo.v1'
OBS_DIM = 327  # native current300, explicit relative goal3, BC reference24
ACTION_DIM = 24


def distributed_ready():
    return distributed.is_available() and distributed.is_initialized()


def reduce(value,op='mean'):
    """Equal-sized rank batches; every update decision is shared by all ranks."""
    value=value.detach().clone()
    if distributed_ready():
        distributed.all_reduce(value,op=distributed.ReduceOp.MAX if op=='max' else distributed.ReduceOp.SUM)
        if op=='mean':value/=distributed.get_world_size()
    return value


def synchronize_gradients(module):
    if not distributed_ready():return
    parameters=list(module.parameters())
    if any(p.grad is None for p in parameters):raise RuntimeError('Expected dense actor/critic gradients on every rank')
    flat=torch.cat([p.grad.reshape(-1) for p in parameters])
    distributed.all_reduce(flat);flat/=distributed.get_world_size()
    offset=0
    for p in parameters:
        p.grad.copy_(flat[offset:offset+p.numel()].view_as(p));offset+=p.numel()


def observation(current, relative_goal, base, features=None,wam_features=None):
    if current.shape != (len(current),300) or relative_goal.shape != (len(current),3) or base.shape != (len(current),24):
        raise ValueError('Shadow current300/relative goal3/reference24 required')
    prefix=current.clamp(-5,5)
    if features is not None:
        if features.shape!=(len(current),512):raise ValueError('Frozen current-only 512D feature contract')
        prefix=torch.nn.functional.layer_norm(features.detach(),(512,))
    if wam_features is not None:
        if features is not None or wam_features.shape!=(len(current),2304):raise ValueError('Three frozen WAM type summaries required')
        z=torch.nn.functional.layer_norm(wam_features.detach().reshape(-1,3,768),(768,)).flatten(1)
        prefix=torch.cat((prefix,z),-1)
    result=torch.cat((prefix,relative_goal/.1,base),-1)
    if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite RL observation')
    return result


def apply_residual(base, latent, alpha=.15):
    if base.shape != latent.shape or base.shape[-1] != ACTION_DIM or not 0<=alpha<=1:
        raise ValueError('Matched canonical24D actions and bounded alpha required')
    delta=torch.tanh(latent)
    intended=base+alpha*delta
    return intended.clamp(-1,1),intended,delta


def append_episode_clock(obs,tick,controls,kind='none'):
    if kind=='none':return obs
    if kind not in ('zero','remaining') or not 1<=tick<=controls:raise ValueError('Known finite-horizon control clock required')
    value=(controls-tick+1)/controls if kind=='remaining' else 0.
    clock=obs.new_full((*obs.shape[:-1],1),value)
    return torch.cat((obs[...,:-24],clock,obs[...,-24:]),-1)


def dead_critic_streak(curves,episodes):
    """Unknown EV is not evidence of collapse; require all 13 actual chunks."""
    if episodes<=0 or len(curves)<episodes:return False
    for row in curves[-episodes:]:
        segments=row.get('ppo_segments',[])
        if len(segments)!=13:return False
        if any(s.get('value_explained_variance') is None or abs(s['value_explained_variance'])>=1e-4 for s in segments):return False
    return True


def apply_policy_action(base,latent,kind='residual'):
    if kind=='residual':return apply_residual(base,latent)
    if kind not in ('reference-gaussian','independent-gaussian') or base.shape!=latent.shape:raise ValueError('Unknown policy action contract')
    return latent.clamp(-1,1),latent,latent-base


def control_mlp(obs_dim,output_dim,kind,activation='tanh'):
    if kind not in ('small','large'):raise ValueError('Unknown control capacity')
    if activation not in ('tanh','elu'):raise ValueError('Unknown control activation')
    widths=[obs_dim]+([256,256] if kind=='small' else [1024,1024,1024,512])+[output_dim]
    layers=[]
    for index,(left,right) in enumerate(zip(widths,widths[1:])):
        layers.append(nn.Linear(left,right))
        if index<len(widths)-2:layers.append(nn.Tanh() if activation=='tanh' else nn.ELU())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self,initial_log_std=-2.5,obs_dim=OBS_DIM,network_kind='small',action_kind='residual'):
        super().__init__()
        if not -5<=initial_log_std<=-1.2:raise ValueError('Initial latent log_std outside admitted clamp')
        if action_kind not in ('residual','reference-gaussian','independent-gaussian'):raise ValueError('Unknown policy action kind')
        self.action_kind=action_kind
        self.net=control_mlp(obs_dim,ACTION_DIM,network_kind)
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.log_std=nn.Parameter(torch.full((ACTION_DIM,),float(initial_log_std)))

    def distribution(self,obs):
        mean=self.net(obs)
        if self.action_kind=='reference-gaussian':mean=mean+obs[...,-24:]
        return torch.distributions.Normal(mean,self.log_std.clamp(-5,-1.2).exp())


class Critic(nn.Module):
    def __init__(self,obs_dim=OBS_DIM,network_kind='small',activation='tanh',value_scale=1.):
        super().__init__()
        if value_scale not in (1.,100.):raise ValueError('Admitted fixed value scale required')
        self.value_scale=float(value_scale)
        self.net=control_mlp(obs_dim,1,network_kind,activation)

    def forward(self,obs):return self.net(obs).squeeze(-1)*self.value_scale


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


def update_rollout(actor,critic,storage,actor_optimizer,critic_optimizer,next_observation=None,**options):
    """Consume one on-policy chunk; only real terminals have zero bootstrap."""
    batch={key:torch.stack(values) for key,values in storage.items()}
    if next_observation is None:
        if not batch['done'][-1].all():raise ValueError('Zero bootstrap requires real terminal states')
        batch['bootstrap']=torch.zeros_like(batch['value'][-1])
    else:
        if batch['done'][-1].any():raise ValueError('Interior chunk cannot contain episode reset')
        with torch.no_grad():batch['bootstrap']=critic(next_observation).detach()
    result=update(actor,critic,batch,actor_optimizer,critic_optimizer,**options)
    result.update(rollout_steps=len(batch['reward']),bootstrap_mean=float(batch['bootstrap'].mean()),
                  bootstrap_kind='terminal-zero' if next_observation is None else 'actual-next-state-before-update')
    for values in storage.values():values.clear()
    return result


def update(actor,critic,batch,actor_optimizer,critic_optimizer,bc_weight=.1,target_kl=.02,epochs=4,minibatches=1,shuffle_generator=None):
    if minibatches>1:
        from .recipe_ppo import update_minibatches
        return update_minibatches(actor,critic,batch,actor_optimizer,critic_optimizer,bc_weight,target_kl,epochs,minibatches,shuffle_generator)
    adv,returns=advantages(batch['reward'],batch['value'],batch['done'],batch['bootstrap'])
    obs=batch['observation'].flatten(0,1);latent=batch['latent'].flatten(0,1)
    old_logp=batch['logp'].flatten();old=torch.distributions.Normal(batch['mean'].flatten(0,1),batch['std'].flatten(0,1))
    adv=adv.flatten();moments=reduce(torch.stack((adv.double().mean(),adv.double().square().mean())))
    scale=(moments[1]-moments[0].square()).clamp_min(0).sqrt().clamp_min(1e-6)
    adv=((adv.double()-moments[0])/scale).float();returns=returns.flatten()
    with torch.no_grad():replay=float(reduce((actor.distribution(obs).log_prob(latent).sum(-1)-old_logp).abs().max(),'max'))
    if replay>1e-4:raise ValueError('Behavior latent log probability replay drift: '+str(replay))
    steps=0;kl=0.;reg=0.
    for _ in range(epochs):
        dist=actor.distribution(obs);logp=dist.log_prob(latent).sum(-1)
        ratio=(logp-old_logp).exp()
        surrogate=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
        # Penalize executed deterministic canonical deviation, not latent units.
        base=obs[:,-24:];reference=apply_policy_action(base,dist.loc,actor.action_kind)[0]
        reg=(reference-base).square().mean() if actor.action_kind!='independent-gaussian' else dist.loc.sum()*0
        loss=surrogate+bc_weight*reg
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite actor objective')
        actor_optimizer.zero_grad(set_to_none=True);loss.backward();synchronize_gradients(actor);nn.utils.clip_grad_norm_(actor.parameters(),.5)
        before=copy.deepcopy(actor.state_dict());optim_before=copy.deepcopy(actor_optimizer.state_dict())
        accepted=False
        for _retry in range(4):
            actor_optimizer.step()
            with torch.no_grad():kl=float(reduce(torch.distributions.kl_divergence(old,actor.distribution(obs)).sum(-1).mean()))
            if kl<=target_kl:accepted=True;break
            rates=[group['lr']*.25 for group in actor_optimizer.param_groups]
            actor.load_state_dict(before);actor_optimizer.load_state_dict(optim_before)
            for group,rate in zip(actor_optimizer.param_groups,rates):group['lr']=rate
        if not accepted:break
        steps+=1
    # A rejected candidate has been rolled back: report the actual retained policy.
    with torch.no_grad():kl=float(reduce(torch.distributions.kl_divergence(old,actor.distribution(obs)).sum(-1).mean()))
    for _ in range(epochs):
        value_loss=(critic(obs)-returns).square().mean()
        if not torch.isfinite(value_loss):raise FloatingPointError('Nonfinite critic objective')
        normalized_value_loss=value_loss/getattr(critic,'value_scale',1.)**2
        critic_optimizer.zero_grad(set_to_none=True);normalized_value_loss.backward();synchronize_gradients(critic);nn.utils.clip_grad_norm_(critic.parameters(),1.)
        critic_optimizer.step()
    with torch.no_grad():
        error=returns-critic(obs)
        moments=reduce(torch.stack((returns.double().mean(),returns.double().square().mean(),error.double().mean(),error.double().square().mean())))
        target_var=(moments[1]-moments[0].square()).clamp_min(0);error_var=(moments[3]-moments[2].square()).clamp_min(0)
        explained=None if target_var<1e-8 else float(1-error_var/target_var)
    return dict(actor_steps=steps,joint_kl=kl,bc_regularization=float(reg),value_loss=float(value_loss),normalized_value_loss=float(normalized_value_loss),value_explained_variance=explained,
                behavior_logprob_error=replay,actor_lr=actor_optimizer.param_groups[0]['lr'],
                return_mean=float(returns.mean()),advantage_std=float(adv.std(unbiased=False)))
