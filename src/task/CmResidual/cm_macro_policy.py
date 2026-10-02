"""Small binary policy for choosing a short Cm macro or the fixed Cup action.

The policy is deliberately independent of the frozen selector.  Cm is exposed as
physical consequence features; it is not added as a logit prior and it is never
used to manufacture the training reward.
"""
import torch
from torch import nn


class CmMacroPolicy(nn.Module):
    """Shared state encoder with a two-option categorical actor and value head."""

    def __init__(self, input_dim=69 * 10 + 18 + 4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
        )
        self.actor = nn.Linear(64, 2)
        self.value = nn.Linear(64, 1)
        # Start as an unbiased binary policy.  There is no frozen selector
        # prior to overcome and no action option is privileged by construction.
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.body[0].in_features:
            raise ValueError("invalid Cm macro policy feature shape")
        x = self.body(features)
        return torch.distributions.Categorical(logits=self.actor(x)), self.value(x).squeeze(-1)


def macro_features(history, action_difference, physical, cm_on):
    """Create matched on/off inputs from the pre-decision state.

    ``physical`` is score/std/retention/release for the raw top candidate.  The
    off arm receives an exactly shaped zero channel while retaining the shared
    state and candidate difference channel.
    """
    if history.ndim != 3 or history.shape[1:] != (10, 69):
        raise ValueError("history must be [batch,10,69]")
    if action_difference.shape != (len(history), 18) or physical.shape != (len(history), 4):
        raise ValueError("invalid macro feature channels")
    # Fixed scales keep state, actions, and mm/probability channels numerically
    # comparable without learning a new normalizer from either arm.
    state = history.clone()
    state[..., :36] = state[..., :36].clamp(-5, 5) / 2
    state[..., 36:49] = state[..., 36:49].clamp(-5, 5) / 0.5
    state[..., 49:51] = state[..., 49:51].clamp(0, 1)
    state[..., 51:] = state[..., 51:].clamp(-1, 1)
    action = action_difference.clamp(-1, 1)
    effect = physical if cm_on else torch.zeros_like(physical)
    effect = effect.clone()
    effect[..., :2] = effect[..., :2].clamp(-100, 100) / 50
    effect[..., 2:] = effect[..., 2:].clamp(0, 1)
    result = torch.cat((state.flatten(1), action, effect), -1)
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite Cm macro features")
    return result


def ppo_loss(distribution, value, selected, old_logprob, advantage, returns):
    ratio = (distribution.log_prob(selected) - old_logprob).exp()
    clipped = ratio.clamp(0.8, 1.2)
    actor = -torch.minimum(ratio * advantage, clipped * advantage).mean()
    value_loss = (value - returns).square().mean()
    entropy = distribution.entropy().mean()
    loss = actor + 0.5 * value_loss - 0.01 * entropy
    return loss, dict(ratio=ratio, actor=actor, value=value_loss, entropy=entropy)
