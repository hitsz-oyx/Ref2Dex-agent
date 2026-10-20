"""Measured 421/686 actor and critic contracts for the PPO bridge.

The online executor already emits a flat measured observation after the shape
encoder and optional spatial-tau adapter.  This module keeps PPO on that exact
flat boundary and deliberately leaves TaskController/TaskReward outside the
network: the collector supplies the causal reward and the actual next-state
bootstrap.
"""
import copy

import torch
from torch import nn

from .bc_mlp import NormalizedCommandMLP
from .ppo import advantages
from .measured_reward_bridge import MEASURED_GAMMA
from .measured_execution import GOAL_CONTRACT


class MeasuredGaussianActor(nn.Module):
    """A BC-initialized 12D Gaussian actor over a 421D or 686D observation."""

    def __init__(self, observation_dim, *, action_dim=12, initial_log_std=-2.5):
        super().__init__()
        if observation_dim not in (421, 686) or action_dim != 12:
            raise ValueError("measured PPO requires 421/686 observation and 12 actions")
        if not -5 <= initial_log_std <= -1.2:
            raise ValueError("initial latent log_std outside admitted clamp")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.mean = NormalizedCommandMLP(observation_dim, 312, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(initial_log_std)))

    @classmethod
    def from_bc_packet(cls, packet, *, with_tau):
        expected_schema = (
            "ref2dex.inspire-measured-scheduled-bc.v2"
            if with_tau
            else "ref2dex.inspire-measured-current-bc.v1"
        )
        if packet.get("schema") != expected_schema:
            raise ValueError("measured BC checkpoint schema mismatch")
        if with_tau and packet.get('config', {}).get('goal_contract') != GOAL_CONTRACT:
            raise ValueError('measured Scheduled686 requires the shared TaskController goal')
        actor = cls(686 if with_tau else 421)
        source = packet["model"]
        prefix = "actor."
        state = {key[len(prefix):]: value for key, value in source.items() if key.startswith(prefix)}
        if set(state) != set(actor.mean.state_dict()):
            raise ValueError("measured BC actor state does not match 421/686 PPO actor")
        actor.mean.load_state_dict(state, strict=True)
        return actor

    def distribution(self, observation):
        if observation.shape[-1] != self.observation_dim or not torch.isfinite(observation).all():
            raise ValueError("finite measured PPO observation with the declared dimension required")
        mean = self.mean(observation)
        std = self.log_std.clamp(-5, -1.2).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def act_mean(self, observation):
        return self.distribution(observation).loc.clamp(-1, 1)

    def sample(self, observation):
        """Unclipped behavior sample; clip only the command sent to physics."""
        distribution = self.distribution(observation)
        action = distribution.sample()
        return action, distribution.log_prob(action).sum(-1), distribution.loc, distribution.scale


class MeasuredCritic(nn.Module):
    """Fresh value network kept separate from the BC actor checkpoint."""

    def __init__(self, observation_dim, *, value_scale=1.0):
        super().__init__()
        if observation_dim not in (421, 686) or value_scale not in (1.0, 100.0):
            raise ValueError("measured PPO requires 421/686 observation and admitted value scale")
        self.observation_dim = observation_dim
        self.value_scale = float(value_scale)
        self.net = NormalizedCommandMLP(observation_dim, 312, 1)

    def forward(self, observation):
        if observation.shape[-1] != self.observation_dim or not torch.isfinite(observation).all():
            raise ValueError("finite measured PPO observation with the declared dimension required")
        return self.net(observation).squeeze(-1) * self.value_scale


def update_measured(actor, critic, batch, actor_optimizer, critic_optimizer, *, gamma=MEASURED_GAMMA, lam=0.95,
                    target_kl=0.02, epochs=1):
    """Run one bounded PPO update on a T x N measured collector batch.

    ``batch['bootstrap']`` must be the value of the actual next measured state
    unless the final row contains only real terminal transitions.  The action
    is the unclipped Gaussian sample; the collector clips only the command sent
    to the native host and records that boundary separately.
    """
    required = {"observation", "action", "logp", "mean", "std", "value", "reward", "done", "bootstrap"}
    if set(batch) != required:
        raise ValueError("measured PPO batch fields mismatch")
    obs = batch["observation"]
    if obs.ndim != 3 or obs.shape[-1] != actor.observation_dim:
        raise ValueError("T x N x measured observation required")
    adv, returns = advantages(batch["reward"], batch["value"], batch["done"], batch["bootstrap"], gamma, lam)
    flat_obs = obs.flatten(0, 1)
    action = batch["action"].flatten(0, 1)
    if torch.equal(action, batch['mean'].flatten(0, 1)):
        raise ValueError('PPO requires sampled behavior actions, not deterministic means')
    old_logp = batch["logp"].flatten()
    old = torch.distributions.Normal(batch["mean"].flatten(0, 1), batch["std"].flatten(0, 1))
    flat_adv = adv.flatten()
    scale = flat_adv.std(unbiased=False).clamp_min(1e-6)
    flat_adv = (flat_adv - flat_adv.mean()) / scale
    flat_returns = returns.flatten()
    with torch.no_grad():
        replay_error = float((actor.distribution(flat_obs).log_prob(action).sum(-1) - old_logp).abs().max())
    if replay_error > 1e-4:
        raise ValueError("behavior log probability replay drift: " + str(replay_error))
    actor_steps = 0
    for _ in range(epochs):
        dist = actor.distribution(flat_obs)
        logp = dist.log_prob(action).sum(-1)
        ratio = (logp - old_logp).exp()
        loss = -torch.minimum(ratio * flat_adv, ratio.clamp(0.8, 1.2) * flat_adv).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite measured actor objective")
        actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        before = copy.deepcopy(actor.state_dict())
        optimizer_before = copy.deepcopy(actor_optimizer.state_dict())
        actor_optimizer.step()
        with torch.no_grad():
            kl = float(torch.distributions.kl_divergence(old, actor.distribution(flat_obs)).sum(-1).mean())
        if kl > target_kl:
            actor.load_state_dict(before)
            actor_optimizer.load_state_dict(optimizer_before)
            kl = float(torch.distributions.kl_divergence(old, actor.distribution(flat_obs)).sum(-1).mean())
        else:
            actor_steps += 1
    value_loss = (critic(flat_obs) - flat_returns).square().mean()
    if not torch.isfinite(value_loss):
        raise FloatingPointError("nonfinite measured critic objective")
    critic_optimizer.zero_grad(set_to_none=True)
    (value_loss / critic.value_scale ** 2).backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    critic_optimizer.step()
    with torch.no_grad():
        error = flat_returns - critic(flat_obs)
        target_var = flat_returns.var(unbiased=False)
        explained = None if target_var < 1e-8 else float(1 - error.var(unbiased=False) / target_var)
    return dict(actor_steps=actor_steps, joint_kl=kl, behavior_logprob_error=replay_error,
                value_loss=float(value_loss), value_explained_variance=explained,
                bootstrap_mean=float(batch["bootstrap"].mean()),
                return_mean=float(flat_returns.mean()), advantage_std=float(adv.std(unbiased=False)))
