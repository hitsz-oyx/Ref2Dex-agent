"""Physical-value data and reward contracts, independent of Isaac Gym."""
from __future__ import annotations

from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = "ref2dex.physical_value.v1"
STATE_DIM = 55
ACTION_DIM = 18
HISTORY = 16
OBJECT_QUAT = slice(39, 43)


@contextmanager
def private_initialization(seed):
    # manual_seed on the global torch API also touches CUDA. Only install a
    # private CPU generator state while CPU modules initialize.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        yield


def canonical_quaternion(q):
    q = F.normalize(q, dim=-1, eps=1e-8)
    # Choose the largest component, not w: this also resolves the w==0 case.
    pivot = q.gather(-1, q.abs().argmax(-1, keepdim=True))
    return q * torch.where(pivot < 0, -1.0, 1.0)


def quaternion_loss(predicted, actual):
    p, a = F.normalize(predicted, dim=-1), F.normalize(actual, dim=-1)
    return torch.minimum((p - a).square().mean(-1), (p + a).square().mean(-1))


def advance_events(events, height, contact, initial_height):
    """Events are normalized run, ever-held, normalized lost-contact, dropped."""
    run, ever, lost, dropped = events.unbind(-1)
    held = (height - initial_height >= .03) & contact.bool()
    run = torch.where(held, (run + 1 / 30).clamp_max(1.5), 0.0)
    ever = torch.maximum(ever, (run >= 1 - 1e-6).float())
    lost = torch.where(contact.bool(), 0.0, (lost + 1 / 6).clamp_max(1.0))
    falling = (ever > .5) & ((height - initial_height < .02) | (lost >= 1 - 1e-6))
    drop_event = falling & (dropped < .5)
    dropped = torch.maximum(dropped, falling.float())
    result = torch.stack((run, ever, lost, dropped), -1)
    reward = .5 * run.clamp_max(1) - drop_event.float()
    return result, reward, drop_event


class HoldTracker:
    def __init__(self, num_envs, device):
        self.events = torch.zeros(num_envs, 4, device=device)
        self.initial_height = torch.zeros(num_envs, device=device)
        self.max_run = torch.zeros(num_envs, device=device)
        self.run_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.stable = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.drop_after_success = torch.zeros_like(self.stable)

    def reset(self, ids, height):
        self.events[ids] = 0
        self.initial_height[ids] = height
        self.max_run[ids] = 0
        self.run_steps[ids] = 0
        self.stable[ids] = False
        self.drop_after_success[ids] = False

    def step(self, height, contact):
        previous_success = self.stable.clone()
        self.events, reward, drop = advance_events(
            self.events, height, contact, self.initial_height)
        held = (height - self.initial_height >= .03) & contact.bool()
        self.run_steps = torch.where(held, self.run_steps + 1, 0)
        self.max_run = torch.maximum(self.max_run, self.run_steps.float() / 30)
        self.stable |= self.run_steps >= 45
        # Falling after first reaching 30 steps and falling after primary
        # success are different metrics. Do not conflate their events.
        falling = (height - self.initial_height < .02) | (self.events[:, 2] >= 1 - 1e-6)
        self.drop_after_success |= previous_success & falling
        return reward, drop


class HistoryBuffer:
    def __init__(self, batch, device):
        self.state = torch.zeros(batch, HISTORY, STATE_DIM, device=device)
        self.action = torch.zeros(batch, HISTORY, ACTION_DIM, device=device)
        self.mask = torch.zeros(batch, HISTORY, 1, device=device)

    def reset(self, ids):
        self.state[ids] = 0
        self.action[ids] = 0
        self.mask[ids] = 0

    def append(self, state, previous_action):
        self.state = torch.cat((self.state[:, 1:], state[:, None]), 1)
        self.action = torch.cat((self.action[:, 1:], previous_action[:, None]), 1)
        self.mask = torch.cat((self.mask[:, 1:], torch.ones_like(self.mask[:, :1])), 1)

    def tensors(self):
        return self.state.clone(), self.action.clone(), self.mask.clone()


def make_candidates(mean, dose=.1):
    """37 slots; invalid duplicates are masked, preserving original first."""
    base = mean.clamp(-1, 1)
    perturb = torch.cat((torch.zeros(1, ACTION_DIM, device=mean.device),
                         dose * torch.eye(ACTION_DIM, device=mean.device),
                         -dose * torch.eye(ACTION_DIM, device=mean.device)))
    candidates = (base[:, None] + perturb).clamp(-1, 1)
    valid = torch.ones(candidates.shape[:2], dtype=torch.bool, device=mean.device)
    for index in range(1, candidates.shape[1]):
        duplicate = (candidates[:, :index] == candidates[:, index:index + 1]).all(-1).any(-1)
        valid[:, index] = ~duplicate
    return candidates, valid


def teacher_label(mean, scores, candidates, valid):
    scores = scores.masked_fill(~valid, -torch.inf)
    if not torch.isfinite(scores[:, 0]).all():
        raise FloatingPointError("original action score must be finite")
    index = scores.argmax(-1)  # lowest-index tie break
    selected = candidates[torch.arange(len(mean), device=mean.device), index]
    active = scores.gather(1, index[:, None]).squeeze(1) > scores[:, 0]
    return mean + .1 * (selected - mean), active


def actor_supervision(mean, sigma, label, active):
    rows = ((mean - label.detach()) / sigma.detach().clamp_min(1e-6)).square().mean(-1)
    return rows * active.float()


def validate_rows(data):
    if data.get("schema") != SCHEMA:
        raise ValueError("physical-value schema mismatch")
    n = len(data["state"])
    required = ("state", "next_state", "context", "next_context", "action", "reward",
                "done", "terminate", "timeout", "episode_id", "env_id", "step")
    for key in required:
        value = data[key]
        if len(value) != n or not torch.isfinite(value).all():
            raise ValueError(f"invalid {key}")
    if data["state"].shape != (n, STATE_DIM) or data["next_state"].shape != (n, STATE_DIM):
        raise ValueError("physical state shape mismatch")
    if data["action"].shape != (n, ACTION_DIM):
        raise ValueError("action shape mismatch")
    if (data["terminate"].bool() & ~data["done"].bool()).any():
        raise ValueError("termination without done")
    if (data["timeout"].bool() & ~data["done"].bool()).any():
        raise ValueError("timeout without done")
    if ((data["terminate"].bool() | data["timeout"].bool()) != data["done"].bool()).any():
        raise ValueError("done reason is incomplete")
    norms = data["next_state"][:, OBJECT_QUAT].norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
        raise ValueError("nonunit object quaternion")
    if not torch.allclose(data["reward"], data["reward_components"].sum(-1), atol=1e-5):
        raise ValueError("reward components do not sum to actual reward")
    return n
