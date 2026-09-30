"""Short physical dynamics, value networks, and model-derived action teacher."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from src.task.CmResidual.physical_value_contract import (
    STATE_DIM, ACTION_DIM, OBJECT_QUAT, canonical_quaternion, private_initialization,
    advance_events, make_candidates, teacher_label, quaternion_loss,
)
from src.task.CmResidual.v118_planner import TorchInspireKinematics, QUERY_LINKS, TIP_LINKS

URDF = Path(__file__).resolve().parents[3] / "third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf"


class Features(nn.Module):
    def __init__(self, state_mean, state_std, context_mean, context_std, device="cpu"):
        super().__init__()
        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_std", state_std.clamp_min(.01))
        self.register_buffer("context_mean", context_mean)
        self.register_buffer("context_std", context_std.clamp_min(.01))
        self.fk = TorchInspireKinematics(URDF, torch.device(device))
        self.indices = [QUERY_LINKS.index(n) for n in ("hand_base_link", *TIP_LINKS)]

    def state(self, states):
        # FK uses exactly the same native joint ordering for real/predicted states.
        shape = states.shape[:-1]
        flat = states.reshape(-1, STATE_DIM)
        links = self.fk.forward(flat[:, None, :18])[:, 0, self.indices, :3, 3]
        relative = ((links - flat[:, None, 36:39]) / .1).reshape(*shape, 18)
        normalized = ((states - self.state_mean) / self.state_std).clamp(-20, 20)
        return torch.cat((normalized, relative.clamp(-20, 20)), -1)

    def history(self, states, actions, mask):
        return torch.cat((self.state(states), actions, mask), -1) * mask

    def context(self, context):
        return ((context - self.context_mean) / self.context_std).clamp(-20, 20)


class OutcomeNetwork(nn.Module):
    def __init__(self, context_dim, action_dim, output_dim, seed):
        super().__init__()
        with private_initialization(seed):
            self.encoder = nn.GRU(STATE_DIM + 18 + ACTION_DIM + 1, 128, batch_first=True)
            self.head = nn.Sequential(nn.Linear(128 + context_dim + action_dim, 256), nn.ReLU(),
                                      nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, output_dim))
        self.action_dim = action_dim

    def forward(self, history, context, action=None):
        hidden = self.encoder(history)[0][:, -1]
        items = [hidden, context]
        if self.action_dim:
            if action is None:
                raise ValueError("action-conditioned network requires action")
            items.append(action)
        return self.head(torch.cat(items, -1))


def dynamics_output(model, features, states, actions, mask, context, candidate):
    raw = model(features.history(states, actions, mask), features.context(context), candidate)
    current = states[:, -1]
    next_state = current.clone()
    # q, dq and object position/velocities use normalized residuals.
    next_state[:, :49] += raw[:, :49] * features.state_std[:49]
    next_state[:, OBJECT_QUAT] = canonical_quaternion(raw[:, OBJECT_QUAT])
    return next_state, raw[:, 49:51], raw[:, 51], raw[:, 52]


def dynamics_loss(predicted, contact_logits, reward, terminal_logits, target, actual_reward,
                  terminal, features, reward_scale):
    groups = []
    for part in (slice(0, 18), slice(18, 36), slice(36, 39), slice(43, 49)):
        groups.append(((predicted[:, part] - target[:, part]) / features.state_std[part]).square().mean(-1))
    groups.append(quaternion_loss(predicted[:, OBJECT_QUAT], target[:, OBJECT_QUAT]))
    groups.append(F.binary_cross_entropy_with_logits(contact_logits, target[:, 49:51], reduction="none").mean(-1))
    groups.append(((reward - actual_reward) / reward_scale.clamp_min(1)).square())
    groups.append(F.binary_cross_entropy_with_logits(terminal_logits, terminal.float(), reduction="none"))
    return torch.stack(groups).mean(0).mean()


class Teacher:
    def __init__(self, features, dynamics, value, direct_q, gamma, seed, reward_scale):
        self.features, self.dynamics = features, dynamics
        self.value, self.direct_q, self.gamma = value, direct_q, gamma
        device = features.state_mean.device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.reward_scale = reward_scale

    @torch.no_grad()
    def labels(self, arm, mean, states, actions, mask, context, next_context):
        candidates, valid = make_candidates(mean)
        batch, count = candidates.shape[:2]
        def repeat(x):
            return x[:, None].expand(batch, count, *x.shape[1:]).reshape(batch * count, *x.shape[1:])
        s, a, m, c, nc = map(repeat, (states, actions, mask, context, next_context))
        candidate = candidates.reshape(-1, ACTION_DIM)
        if arm == "direct_q":
            scores = self.direct_q(self.features.history(s, a, m), self.features.context(c), candidate).reshape(batch, count)
        elif arm == "cm_value":
            samples = []
            for model in self.dynamics:
                predicted, logits, reward, terminal_logits = dynamics_output(model, self.features, s, a, m, c, candidate)
                for _ in range(4):
                    predicted = predicted.clone()
                    # Common random numbers across a state's candidates avoid
                    # selecting actions solely because of Monte Carlo noise.
                    uniform = torch.rand(batch, 1, 2, generator=self.generator, device=mean.device)
                    predicted[:, 49:51] = (uniform.expand(batch, count, 2).reshape(-1, 2) < logits.sigmoid()).float()
                    predicted[:, 51:55], _, _ = advance_events(s[:, -1, 51:55], predicted[:, 38],
                                                               predicted[:, 49:51].bool().all(-1), c[:, 6])
                    hs = torch.cat((s[:, 1:], predicted[:, None]), 1)
                    ha = torch.cat((a[:, 1:], candidate[:, None]), 1)
                    hm = torch.cat((m[:, 1:], torch.ones_like(m[:, :1])), 1)
                    continuation = self.value(self.features.history(hs, ha, hm), self.features.context(nc)).squeeze(-1)
                    uniform = torch.rand(batch, 1, generator=self.generator, device=mean.device)
                    terminated = (uniform.expand(batch, count).reshape(-1) < terminal_logits.sigmoid()).float()
                    samples.append(reward + self.gamma * (1 - terminated) * continuation)
            draws = torch.stack(samples)
            scores = (draws.mean(0) - draws.std(0, unbiased=False)).reshape(batch, count)
        else:
            raise ValueError("teacher arm must be direct_q or cm_value")
        label, active = teacher_label(mean, scores, candidates, valid)
        return label, active, scores
