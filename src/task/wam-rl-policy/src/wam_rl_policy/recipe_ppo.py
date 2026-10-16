"""Minibatch PPO qualification path; legacy full-batch updates remain unchanged."""
import copy
import torch
from torch import nn
from .ppo import advantages,reduce,synchronize_gradients,apply_policy_action


def update_minibatches(actor,critic,batch,actor_optimizer,critic_optimizer,bc_weight,target_kl,epochs,minibatches,shuffle_generator=None):
    adv,returns=advantages(batch['reward'],batch['value'],batch['done'],batch['bootstrap'])
    obs=batch['observation'].flatten(0,1);latent=batch['latent'].flatten(0,1)
    if shuffle_generator is None:shuffle_generator=torch.Generator(device=obs.device).manual_seed(0)
    old_logp=batch['logp'].flatten();old=torch.distributions.Normal(batch['mean'].flatten(0,1),batch['std'].flatten(0,1))
    adv=adv.flatten();moments=reduce(torch.stack((adv.double().mean(),adv.double().square().mean())))
    scale=(moments[1]-moments[0].square()).clamp_min(0).sqrt().clamp_min(1e-6)
    adv=((adv.double()-moments[0])/scale).float();returns=returns.flatten()
    with torch.no_grad():replay=float(reduce((actor.distribution(obs).log_prob(latent).sum(-1)-old_logp).abs().max(),'max'))
    if replay>1e-4:raise ValueError('Behavior latent log probability replay drift: '+str(replay))
    if len(obs)%minibatches:raise ValueError('Equal minibatch sizes required across ranks')
    steps=0;rejected=0;stop=False;reg=torch.zeros((),device=obs.device)
    for _ in range(epochs):
        for indices in torch.randperm(len(obs),device=obs.device,generator=shuffle_generator).chunk(minibatches):
            dist=actor.distribution(obs[indices]);ratio=(dist.log_prob(latent[indices]).sum(-1)-old_logp[indices]).exp()
            surrogate=-torch.minimum(ratio*adv[indices],ratio.clamp(.8,1.2)*adv[indices]).mean()
            base=obs[indices,-24:];reference=apply_policy_action(base,dist.loc,actor.action_kind)[0]
            reg=(reference-base).square().mean();loss=surrogate+bc_weight*reg
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite actor objective')
            actor_optimizer.zero_grad(set_to_none=True);loss.backward();synchronize_gradients(actor);nn.utils.clip_grad_norm_(actor.parameters(),.5)
            before=copy.deepcopy(actor.state_dict());optim_before=copy.deepcopy(actor_optimizer.state_dict());accepted=False
            for retry in range(4):
                actor_optimizer.step()
                with torch.no_grad():kl=float(reduce(torch.distributions.kl_divergence(old,actor.distribution(obs)).sum(-1).mean()))
                if kl<=target_kl:accepted=True;break
                rates=[group['lr'] for group in actor_optimizer.param_groups]
                actor.load_state_dict(before);actor_optimizer.load_state_dict(optim_before);rejected+=1
                # Calibrate an oversized first step. Once this rollout has progressed,
                # stop at its trust-region boundary instead of shrinking LR indefinitely.
                if steps or retry==3:break
                for group,rate in zip(actor_optimizer.param_groups,rates):group['lr']=rate*.25
                optim_before=copy.deepcopy(actor_optimizer.state_dict())
            if not accepted:stop=True;break
            steps+=1
        if stop:break
    value_steps=0
    for _ in range(epochs):
        for indices in torch.randperm(len(obs),device=obs.device,generator=shuffle_generator).chunk(minibatches):
            value_loss=(critic(obs[indices])-returns[indices]).square().mean()
            if not torch.isfinite(value_loss):raise FloatingPointError('Nonfinite critic objective')
            critic_optimizer.zero_grad(set_to_none=True);value_loss.backward();synchronize_gradients(critic);nn.utils.clip_grad_norm_(critic.parameters(),1.)
            critic_optimizer.step();value_steps+=1
    with torch.no_grad():
        kl=float(reduce(torch.distributions.kl_divergence(old,actor.distribution(obs)).sum(-1).mean()))
        error=returns-critic(obs)
        moments=reduce(torch.stack((returns.double().mean(),returns.double().square().mean(),error.double().mean(),error.double().square().mean())))
        target_var=(moments[1]-moments[0].square()).clamp_min(0);error_var=(moments[3]-moments[2].square()).clamp_min(0)
        explained=None if target_var<1e-8 else float(1-error_var/target_var)
    return dict(actor_steps=steps,critic_steps=value_steps,rejected_candidates=rejected,joint_kl=kl,bc_regularization=float(reg),
        value_loss=float(value_loss),value_explained_variance=explained,behavior_logprob_error=replay,
        actor_lr=actor_optimizer.param_groups[0]['lr'],return_mean=float(returns.mean()),advantage_std=float(adv.std(unbiased=False)),
        optimizer='equal-size shuffled minibatches; global rollout KL; first-step LR calibration, then rollback/early-stop')
