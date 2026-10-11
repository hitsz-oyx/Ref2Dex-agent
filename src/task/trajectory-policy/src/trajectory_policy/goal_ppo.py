"""Low-level goal-conditioned PPO update and measured tracking reward."""

from __future__ import annotations

import copy

import torch

from .goal_policy import ACTIVE_ACTION
from .ppo import advantages, task_reward


def goal_reward(hand, obj, hand_goal, obj_goal, gap, lift, supported,
                previous_held, clipped, action_delta):
    """Measured task reward plus hand/object goal tracking; no q oracle.

    A weak tracking component supplies startup signal while the physical
    proximity/lift/held/loss terms remain the execution target.  Current
    hand/object measurements and the corresponding goal are the only reference
    quantities accepted here.
    """
    if (hand.shape != hand_goal.shape or obj.shape != obj_goal.shape
            or hand.ndim != 3 or hand.shape[1:] != (11, 3)
            or obj.shape[1:] != (4, 4)):
        raise ValueError("tracking reward hand/object shape mismatch")
    hand_error = (hand - hand_goal).square().mean((1, 2)).sqrt()
    object_error = (obj[:, :3, 3] - obj_goal[:, :3, 3]).norm(dim=-1)
    task, held, terms = task_reward(gap, lift, supported, previous_held, clipped, action_delta)
    tracking_terms = dict(
        hand_tracking=.25 * torch.exp(-hand_error / .05),
        object_tracking=.25 * torch.exp(-object_error / .05),
    )
    reward = task + tracking_terms["hand_tracking"] + tracking_terms["object_tracking"]
    if not torch.isfinite(reward).all():
        raise FloatingPointError("nonfinite goal PPO reward")
    return reward, held, dict(terms, **tracking_terms)


def update(actor, value_net, batch, bootstrap, actor_optimizer, value_optimizer,
           gamma=.99, gae_lambda=1., target_kl=.02, epochs=4):
    """PPO on active Gaussian latents with a bounded joint KL proposal."""
    reward, value, done = (batch[key] for key in ("reward", "value", "done"))
    duration = torch.ones_like(done, dtype=torch.long)
    advantage, returns = advantages(reward, value, done, duration, bootstrap,
                                    gamma=gamma, lam=gae_lambda)
    features, latent, old_logp, old_mean, old_std = (
        batch[key].flatten(0, 1) for key in ("features", "latent", "logp", "mean", "std")
    )
    target = returns.flatten()
    normalized = advantage.flatten()
    normalized = (normalized - normalized.mean()) / normalized.std(unbiased=False).clamp_min(1e-6)
    active = list(ACTIVE_ACTION)
    with torch.no_grad():
        replay = actor.distribution(features).log_prob(latent)[:, active].sum(-1)
        replay_error = float((replay - old_logp).abs().max())
    if replay_error > .02:
        raise ValueError("behavior log probability replay drift: " + str(replay_error))
    old_distribution = torch.distributions.Normal(old_mean[:, active], old_std[:, active])
    actor_steps, kl, policy_loss = 0, 0., 0.
    for _ in range(epochs):
        distribution = actor.distribution(features)
        logp = distribution.log_prob(latent)[:, active].sum(-1)
        ratio = torch.exp((logp - old_logp).clamp(-20, 20))
        objective = -torch.minimum(ratio * normalized,
                                   ratio.clamp(.8, 1.2) * normalized).mean()
        if not torch.isfinite(objective):
            raise FloatingPointError("nonfinite goal PPO objective")
        actor_optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), .5)
        state = {key: value.detach().clone() for key, value in actor.state_dict().items()}
        optimizer_state = copy.deepcopy(actor_optimizer.state_dict())
        accepted = False
        for _ in range(4):
            actor_optimizer.step()
            with torch.no_grad():
                current = actor.distribution(features)
                current_active = torch.distributions.Normal(current.loc[:, active], current.scale[:, active])
                kl = float(torch.distributions.kl_divergence(old_distribution, current_active).sum(-1).mean())
            if kl <= target_kl:
                accepted = True
                break
            next_lr = actor_optimizer.param_groups[0]["lr"] * .25
            actor.load_state_dict(state)
            actor_optimizer.load_state_dict(optimizer_state)
            actor_optimizer.param_groups[0]["lr"] = next_lr
        if not accepted:
            break
        actor_steps += 1
        policy_loss = float(objective.detach())
    for _ in range(epochs):
        raw_features = batch["raw_features"].flatten(0, 1) if "raw_features" in batch else features
        value_loss = (value_net(raw_features) - target).square().mean()
        if not torch.isfinite(value_loss):
            raise FloatingPointError("nonfinite goal PPO value objective")
        value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.)
        value_optimizer.step()
    return dict(advantage=advantage.detach(), returns=returns.detach(),
                actor_steps=actor_steps, policy_loss=policy_loss, value_loss=float(value_loss.detach()),
                joint_action_kl=kl, behavior_logprob_error=replay_error,
                actor_lr=actor_optimizer.param_groups[0]["lr"])
