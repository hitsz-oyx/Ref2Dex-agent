"""Experimental reference-conditioned PD-target tracking, independent of Gym.

The robot-state reference is an explicit oracle for the first control Probe.
No commanded teacher action is an input or supervision target.
"""
import torch
from torch import nn

SCHEMA = "ref2dex.reference-tracker.v1"
ACTIVE = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15)
RESIDUAL_LIMITS = (.06, .06, .06, .6, .6, .6, .5, .5, .5, .5, .35, .35)
INPUT_DIM = 36 + 33 + 24 * 33 + 18 + 12 + 12 + 6


def reference_velocity(q, dt):
    """Position-derived generalized velocity, with one-sided episode endpoints."""
    if q.ndim != 2 or q.shape[1] != 18 or len(q) < 2 or dt <= 0:
        raise ValueError("finite episode q and positive control dt required")
    delta = q[1:] - q[:-1]
    delta = delta.clone()
    delta[:, 3:6] = torch.atan2(torch.sin(delta[:, 3:6]), torch.cos(delta[:, 3:6]))
    velocity = torch.cat((delta[:1], (delta[:-1] + delta[1:]) / 2, delta[-1:])) / dt
    if not torch.isfinite(velocity).all():
        raise FloatingPointError("nonfinite reference velocity")
    return velocity


def future_reference_velocity(q, dt):
    """Nominal future velocity must not depend on the live calibration q[0].

    q[0] is measured feedback, whereas q[1:] is the future geometric plan.
    The first future target uses a one-sided derivative; later targets retain
    central differences. Index0 is unused and returned as zero.
    """
    if q.ndim != 2 or q.shape[1] != 18 or len(q) < 3:
        raise ValueError("current calibration and at least two future targets required")
    future = reference_velocity(q[1:], dt)
    return torch.cat((torch.zeros_like(future[:1]), future))


def wrist_feedforward(q, velocity, damping_over_stiffness):
    """Compensate native position-drive damping using kinematic reference velocity.

    This yields Kp*(q_ref-q)+Kd*(dq_ref-dq), without pretending to invert
    acceleration, contact, gravity or solver dynamics. Finger preload remains
    learned; its velocity is not treated as a wrist kinematic reference.
    """
    if q.shape != velocity.shape or q.shape[-1] != 18:
        raise ValueError("matching 18-D reference position/velocity required")
    target = q.clone()
    target[..., :6] += velocity[..., :6] * damping_over_stiffness
    return target


def apply_coupling(target):
    target = target.clone()
    for distal, parent, ratio in ((7, 6, 1.05), (9, 8, 1.05), (11, 10, 1.05),
                                  (13, 12, 1.05), (16, 15, .6), (17, 15, .8)):
        target[..., distal] = target[..., parent] * ratio
    return target


def native_action(target, current_q, offset, scale):
    """Invert the actual Inspire positional-target adapter before coupling.

    Wrist targets are relative to live q; fingers use the native [0,1] scale.
    Dependent coordinates are zero because native coupling replaces them.
    """
    if target.shape != current_q.shape or target.shape[-1] != 18:
        raise ValueError("target/current q must share an 18-D suffix")
    if offset.shape != (18,) or scale.shape != (18,) or (scale <= 0).any():
        raise ValueError("positive native scale and 18-D offset required")
    value = (target - offset) / scale
    value[..., :6] = (target[..., :6] - current_q[..., :6] - offset[:6]) / scale[:6]
    value[..., 6:] = value[..., 6:] * 2 - 1
    mask = torch.zeros(18, device=value.device, dtype=value.dtype)
    mask[list(ACTIVE)] = 1
    value = value * mask
    if not torch.isfinite(value).all():
        raise FloatingPointError("nonfinite native tracking command")
    return value


def features(q, dq, hand, obj, obj_velocity, future_hand, next_q, next_obj, previous):
    """Preserve absolute reference/live error, expressed in the live object frame.

    Future hand coordinates are fixed reference states, never recentered to the
    live hand. The reference q is deliberately privileged in this first Probe.
    """
    n = len(q)
    if (q.shape != (n, 18) or dq.shape != q.shape or hand.shape != (n, 11, 3)
            or future_hand.shape != (n, 24, 11, 3) or obj.shape != (n, 4, 4)
            or next_obj.shape != obj.shape or next_q.shape != q.shape
            or previous.shape != (n, 12) or obj_velocity.shape != (n, 6)):
        raise ValueError("reference tracker feature shape mismatch")
    rotation = obj[:, :3, :3]
    center = obj[:, :3, 3]
    local_hand = (hand - center[:, None]) @ rotation
    local_future = (future_hand - center[:, None, None]) @ rotation[:, None]
    ref_error = next_q - q
    ref_error = ref_error.clone()
    ref_error[:, 3:6] = torch.atan2(torch.sin(ref_error[:, 3:6]), torch.cos(ref_error[:, 3:6]))
    obj_error = torch.cat(((next_obj[:, :3, 3] - center) / .1,
                           (next_obj[:, :3, :3] - rotation).flatten(1)), -1)
    values = torch.cat((q, dq / 8, local_hand.flatten(1) / .1,
                        local_future.flatten(1) / .1, ref_error, obj_error,
                        previous, obj_velocity / 2), -1).clamp(-20, 20)
    if values.shape != (n, INPUT_DIM) or not torch.isfinite(values).all():
        raise FloatingPointError("invalid reference tracker features")
    return values


def tracking_reward(hand, obj, reference_hand, reference_obj, pair, initial_height, latent):
    """Object/interaction tracking plus a weak native force-pair hold proxy.

    There is no joint-error penalty: actual finger q need not equal PD preload.
    The force-pair term is a net-force proxy, not a collision-pair/force target.
    Dense surface proximity is evaluated separately in full native episodes.
    """
    position = (obj[:, :3, 3] - reference_obj[:, :3, 3]).norm(dim=-1)
    orientation = (obj[:, :3, :3] - reference_obj[:, :3, :3]).square().mean((1, 2)).sqrt()
    relative_hand = hand - obj[:, None, :3, 3]
    reference_relative = reference_hand - reference_obj[:, None, :3, 3]
    interaction = (relative_hand - reference_relative).square().mean((1, 2)).sqrt()
    wants_lift = reference_obj[:, 2, 3] - initial_height > .03
    actual_lift = obj[:, 2, 3] - initial_height > .03
    hold_proxy = (pair & actual_lift & wants_lift).float()
    reward = (.55 * torch.exp(-position / .05) + .10 * torch.exp(-orientation / .15)
              + .30 * torch.exp(-interaction / .04) + .15 * hold_proxy
              - .005 * torch.tanh(latent).square().mean(-1))
    if not torch.isfinite(reward).all():
        raise FloatingPointError("nonfinite tracking reward")
    return reward


def advantages(rewards, values, dones, bootstrap, gamma=.99, lam=.95):
    """GAE with terminal transitions explicitly cutting cross-reset bootstrap."""
    result = torch.zeros_like(rewards)
    tail = torch.zeros_like(bootstrap)
    following = bootstrap
    for step in reversed(range(len(rewards))):
        alive = (~dones[step]).float()
        delta = rewards[step] + gamma * following * alive - values[step]
        tail = delta + gamma * lam * alive * tail
        result[step] = tail
        following = values[step]
    return result, result + values


class ReferenceTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(INPUT_DIM, 128), nn.Tanh(),
                                   nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 12))
        self.critic = nn.Sequential(nn.Linear(INPUT_DIM, 128), nn.Tanh(),
                                    nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.full((12,), -1.5))
        self.register_buffer("limits", torch.tensor(RESIDUAL_LIMITS))

    def distribution(self, observation):
        return torch.distributions.Normal(self.actor(observation), self.log_std.clamp(-3, -.3).exp())

    def value(self, observation):
        return self.critic(observation).squeeze(-1)

    def target(self, reference_q, latent):
        target = reference_q.clone()
        target[:, list(ACTIVE)] += torch.tanh(latent) * self.limits
        return apply_coupling(target)
