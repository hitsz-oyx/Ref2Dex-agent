"""12-D residual PPO on a frozen native base action.

The base action already comes from the frozen WAM and R. This module only
adds a clipped residual and a task reward. It does not read an expert command
or a world latent.
"""
import copy

import torch
from torch import nn

from .ppo import advantages
from .retargeter_data import ACTIVE

ALPHA = 0.15
OBS_DIM = 393
RESIDUAL_DIM = 12
TAU_STEPS = 8


def apply_residual(base, delta, alpha=ALPHA):
    """Add the residual on the 12 native active coordinates and clip."""
    if base.shape[-1] != 18 or delta.shape != (*base.shape[:-1], RESIDUAL_DIM):
        raise ValueError('18-D base action and 12-D residual required')
    if not torch.isfinite(base).all() or not torch.isfinite(delta).all():
        raise ValueError('finite base action and residual required')
    out = base.clone()
    out[..., list(ACTIVE)] = (base[..., list(ACTIVE)] + alpha * delta).clamp(-1, 1)
    return out


def observation(q, dq, previous, obj, velocity, hand, tau, reference_poses):
    """Measured state, the WAM's short plan, and three planned object poses."""
    if (q.shape[-1] != 18 or dq.shape != q.shape or previous.shape != q.shape
            or obj.shape[-2:] != (4, 4) or velocity.shape[-1] != 6
            or hand.shape[-2:] != (11, 3) or tau.shape[-2:] != (24, 33)
            or reference_poses.shape[-3:] != (3, 4, 4) or len(q) != len(reference_poses)):
        raise ValueError('residual observation tensors do not match the contract')
    rotation = obj[:, :3, :3]
    center = obj[:, :3, 3]
    local_hand = torch.einsum('bpj,bjk->bpk', hand - center[:, None], rotation)
    relative_rotation = torch.einsum('bij,bkjl->bkil', rotation.transpose(-1, -2), reference_poses[:, :, :3, :3])
    relative_position = torch.einsum('bkj,bji->bki', reference_poses[:, :, :3, 3] - center[:, None], rotation)
    goal = torch.cat((relative_position, relative_rotation[:, :, :, 0], relative_rotation[:, :, :, 1]), -1)
    value = torch.cat((
        q, dq, previous, center, rotation[:, :, 0], rotation[:, :, 1], velocity,
        local_hand.reshape(len(q), -1), tau[:, :TAU_STEPS].reshape(len(q), -1), goal.reshape(len(q), -1),
    ), -1)
    if value.shape != (len(q), OBS_DIM) or not torch.isfinite(value).all():
        raise ValueError('finite 393-D residual observation required')
    return value


def object_distance(obj, reference_obj):
    if obj.shape[-2:] != (4, 4) or reference_obj.shape != obj.shape:
        raise ValueError('current and reference object poses required')
    return torch.linalg.vector_norm(obj[:, :3, 3] - reference_obj[:, :3, 3], dim=-1)


def residual_reward(distance, previous_distance, gap, lift, supported, previous_held, delta):
    """Progress, contact, hold, and a penalty for leaving the object in the air.

    Throwing the object up while the hand opens is a negative slip term.
    Matching an expert command is not part of this reward.
    """
    if delta.shape[-1] != RESIDUAL_DIM or distance.shape != gap.shape:
        raise ValueError('residual reward batch mismatch')
    progress = (previous_distance - distance).clamp(-0.05, 0.05)
    near = gap <= 0.01
    held = near & ~supported & (lift >= 0.03)
    contact = torch.exp(-gap.clamp_min(0) / 0.01)
    slip = ((lift > 0.03) & (gap > 0.05)).to(distance.dtype)
    lost = (previous_held & ~held).to(distance.dtype)
    terms = dict(progress=progress, contact=0.2 * contact, held=held.to(distance.dtype),
                 slip=-slip, lost=-0.5 * lost, residual=-0.05 * delta.square().mean(-1))
    reward = sum(terms.values())
    if not torch.isfinite(reward).all():
        raise FloatingPointError('nonfinite residual reward')
    return reward, held, terms


def rollout_bootstrap(value_net, final_observation, done):
    """Value the actual post-control state; only real terminals have zero value."""
    with torch.no_grad():
        value = value_net(final_observation)
    if value.shape != done.shape or not torch.isfinite(value).all():
        raise ValueError('finite final value for each environment required')
    return torch.where(done.bool(), torch.zeros_like(value), value)


def residual_execution_stats(base, delta, applied, alpha=ALPHA):
    active = base[..., list(ACTIVE)]
    intended = active + alpha * delta
    executed = applied[..., list(ACTIVE)]
    axes = tuple(range(delta.ndim - 1))
    return dict(
        saturation_by_dof=(intended - executed).abs().gt(1e-6).float().mean(axes).tolist(),
        positive_margin_by_dof=(1 - active.clamp(-1, 1)).mean(axes).tolist(),
        negative_margin_by_dof=(1 + active.clamp(-1, 1)).mean(axes).tolist(),
        applied_residual_rms=float((executed - active.clamp(-1, 1)).square().mean().sqrt()),
    )


class ResidualActor(nn.Module):
    """Zero-mean start, so the first sample stays near the frozen base action."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM, 256), nn.Tanh(),
                                 nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, RESIDUAL_DIM))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.log_std = nn.Parameter(torch.full((RESIDUAL_DIM,), -2.))

    def distribution(self, obs):
        if obs.shape[-1] != OBS_DIM:
            raise ValueError('393-D residual observation required')
        std = self.log_std.clamp(-5., -1.2).exp()
        return torch.distributions.Normal(self.net(obs), std)


class ResidualValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM, 256), nn.Tanh(),
                                 nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1))

    def forward(self, obs):
        if obs.shape[-1] != OBS_DIM:
            raise ValueError('393-D residual observation required')
        return self.net(obs).squeeze(-1)


def update(actor, value_net, batch, bootstrap, actor_optimizer, value_optimizer,
           gamma=0.99, gae_lambda=1.0, target_kl=0.02, epochs=4):
    """PPO on the 12 residual coordinates. One control is one action."""
    reward, value, done = (batch[key] for key in ('reward', 'value', 'done'))
    duration = torch.ones_like(done, dtype=torch.long)
    advantage, returns = advantages(reward, value, done, duration, bootstrap, gamma=gamma, lam=gae_lambda)
    obs, delta, old_logp, old_mean, old_std = (
        batch[key].flatten(0, 1) for key in ('observation', 'delta', 'logp', 'mean', 'std'))
    target = returns.flatten()
    normalized = advantage.flatten()
    normalized = (normalized - normalized.mean()) / normalized.std(unbiased=False).clamp_min(1e-6)
    with torch.no_grad():
        replay = actor.distribution(obs).log_prob(delta).sum(-1)
        replay_error = float((replay - old_logp).abs().max())
    if replay_error > 0.02:
        raise ValueError('behavior log probability replay drift: ' + str(replay_error))
    old_distribution = torch.distributions.Normal(old_mean, old_std)
    actor_steps, kl, policy_loss = 0, 0., 0.
    for _ in range(epochs):
        distribution = actor.distribution(obs)
        logp = distribution.log_prob(delta).sum(-1)
        ratio = torch.exp((logp - old_logp).clamp(-20, 20))
        objective = -torch.minimum(ratio * normalized, ratio.clamp(0.8, 1.2) * normalized).mean()
        if not torch.isfinite(objective):
            raise FloatingPointError('nonfinite residual PPO objective')
        actor_optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        state = {key: value.detach().clone() for key, value in actor.state_dict().items()}
        optimizer_state = copy.deepcopy(actor_optimizer.state_dict())
        accepted = False
        for _attempt in range(4):
            actor_optimizer.step()
            with torch.no_grad():
                current = actor.distribution(obs)
                kl = float(torch.distributions.kl_divergence(
                    old_distribution, torch.distributions.Normal(current.loc, current.scale)).sum(-1).mean())
            if kl <= target_kl:
                accepted = True
                break
            next_rates = [group['lr'] * 0.25 for group in actor_optimizer.param_groups]
            actor.load_state_dict(state)
            actor_optimizer.load_state_dict(optimizer_state)
            for group, rate in zip(actor_optimizer.param_groups, next_rates):
                group['lr'] = rate
        if not accepted:
            break
        actor_steps += 1
        policy_loss = float(objective.detach())
    for _ in range(epochs):
        value_loss = (value_net(obs) - target).square().mean()
        if not torch.isfinite(value_loss):
            raise FloatingPointError('nonfinite residual value objective')
        value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.)
        value_optimizer.step()
    with torch.no_grad():
        target_variance = target.var(unbiased=False)
        explained_variance = (1 - (target - value_net(obs)).var(unbiased=False) / target_variance
                              if target_variance > 1e-8 else torch.zeros_like(target_variance))
    return dict(actor_steps=actor_steps, policy_loss=policy_loss, value_loss=float(value_loss.detach()),
                joint_action_kl=kl, behavior_logprob_error=replay_error,
                value_explained_variance=float(explained_variance),
                bootstrap_mean=float(bootstrap.mean()), gae_lambda=gae_lambda,
                actor_lr=actor_optimizer.param_groups[0]['lr'])
