"""Task-only reward and variable-duration high-level PPO contracts."""
import copy

import torch
from torch import nn

from .history import INPUT_DIM


def task_reward(gap, lift, supported, previous_held, clipped, action_delta):
    """Only actual present geometry/control; no reference/force/phase argument."""
    near = gap <= .01
    held = near & ~supported & (lift >= .03)
    terms = dict(proximity=.2*torch.exp(-2*gap.clamp_min(0)),
        lift=.5*(lift/.2).clamp(0, 1)*near,
        held=held.float(), loss=-.5*(previous_held & ~held).float(),
        clipping=-.1*clipped.float(), action_rate=-.01*action_delta.square().mean(-1))
    return sum(terms.values()), held, terms


def advantages(reward, value, done, duration, bootstrap, gamma=.99, lam=.95):
    """R_k=sum gamma^j r_j; bootstrap gamma^k and per-control GAE lambda^k.

    A sampled trajectory has one log probability regardless of its duration.
    Done marks the declared task episode end, not a rollout truncation.
    """
    if reward.shape != value.shape or done.shape != value.shape or duration.shape != value.shape or bootstrap.shape != value.shape[1:]:
        raise ValueError('high-level transition shapes mismatch')
    if (duration < 1).any() or (duration > 8).any():
        raise ValueError('executed prefix duration must be in1:8')
    advantage = torch.zeros_like(value)
    carry = torch.zeros_like(bootstrap)
    for tick in reversed(range(len(value))):
        continuation = (~done[tick]).to(value.dtype)
        next_value = bootstrap if tick == len(value)-1 else value[tick+1]
        delta = reward[tick]+gamma**duration[tick]*continuation*next_value-value[tick]
        carry = delta+(gamma*lam)**duration[tick]*continuation*carry
        advantage[tick] = carry
    return advantage, advantage+value


class HistoryValue(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer('mean', mean.detach().clone())
        self.register_buffer('scale', scale.detach().clone())
        self.net = nn.Sequential(nn.Linear(INPUT_DIM, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1))

    def forward(self, history):
        return self.net((history-self.mean)/self.scale).squeeze(-1)


def update(actor, value_net, batch, bootstrap, actor_optimizer, value_optimizer):
    advantage, returns = advantages(batch['reward'], batch['value'], batch['done'], batch['duration'], bootstrap)
    h, c, old_logp, old_mean, old_std, target = (batch[key].flatten(0, 1) for key in
        ('history', 'c', 'logp', 'mean', 'std', 'value'))
    target = returns.flatten()
    a = advantage.flatten()
    a = (a-a.mean())/a.std(unbiased=False).clamp_min(1e-6)
    with torch.no_grad():
        replay_error = float(abs(actor.distribution(h).log_prob(c).sum(-1)-old_logp).max())
    if replay_error > .02:
        raise ValueError('behavior logprob replay drift: '+str(replay_error))
    old_distribution = torch.distributions.Normal(old_mean, old_std)
    steps, kl, policy_loss = 0, 0., 0.
    for epoch in range(4):
        distribution = actor.distribution(h)
        logp = distribution.log_prob(c).sum(-1)
        ratio = torch.exp((logp-old_logp).clamp(-20, 20))
        objective = -torch.minimum(ratio*a, ratio.clamp(.8, 1.2)*a).mean()
        if not torch.isfinite(objective):
            raise FloatingPointError('nonfinite PPO objective')
        actor_optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), .5)
        state = {key: value.detach().clone() for key, value in actor.state_dict().items()}
        optimizer_state = copy.deepcopy(actor_optimizer.state_dict())
        accepted = False
        for trial in range(4):
            actor_optimizer.step()
            with torch.no_grad():
                kl = float(torch.distributions.kl_divergence(old_distribution, actor.distribution(h)).sum(-1).mean())
            if kl <= .02:
                accepted = True
                break
            next_lr = actor_optimizer.param_groups[0]['lr']*.25
            actor.load_state_dict(state)
            actor_optimizer.load_state_dict(optimizer_state)
            actor_optimizer.param_groups[0]['lr'] = next_lr
        if not accepted:
            # A failed trust-region proposal leaves actor and Adam moments intact.
            with torch.no_grad():
                kl = float(torch.distributions.kl_divergence(old_distribution, actor.distribution(h)).sum(-1).mean())
            break
        steps += 1
        policy_loss = float(objective)
    for epoch in range(4):
        value_loss = (value_net(h)-target).square().mean()
        if not torch.isfinite(value_loss):
            raise FloatingPointError('nonfinite value objective')
        value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.)
        value_optimizer.step()
    return dict(advantage=advantage.detach(), returns=returns.detach(),
        actor_steps=steps, policy_loss=policy_loss, value_loss=float(value_loss),
        joint_action_kl=kl, behavior_logprob_error=replay_error, actor_lr=actor_optimizer.param_groups[0]['lr'])
