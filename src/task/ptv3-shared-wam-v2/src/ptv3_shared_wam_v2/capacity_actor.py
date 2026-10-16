"""Matched query-head width, explicit planned-goal and raw-current bypass probes."""
import torch
from torch import nn
from .shared_actor import SharedQueryActor


class CapacityActor(SharedQueryActor):
    def __init__(self, width=128, direct_goal=False, bypass=False):
        super().__init__('shared', width=width, action_dim=15, current_observation='delta0')
        self.direct_goal = direct_goal
        self.bypass_enabled = bypass
        # Identical architecture and initial weights for goal on/off at each width.
        self.goal_encoder = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        nn.init.zeros_(self.goal_encoder[-1].weight)
        nn.init.zeros_(self.goal_encoder[-1].bias)
        self.register_buffer('goal_mean', torch.zeros(3))
        self.register_buffer('goal_scale', torch.ones(3))
        # Direct current-to-output path avoids the width-D state token. Always allocated.
        self.bypass = nn.Linear(721, 8 * 15, bias=False)
        nn.init.zeros_(self.bypass.weight)

    def forward(self, state, shared, valid, current, goal):
        normalized = (state - self.state_mean) / self.state_scale
        current = (current - self.current_mean) / self.current_scale
        goal = (goal - self.goal_mean) / self.goal_scale
        if not self.direct_goal:
            goal = torch.zeros_like(goal)
        token = self.state_encoder(normalized) + self.current_encoder(current) + self.goal_encoder(goal)
        memory = torch.cat((token[:, None] + self.type_embedding[0],
                            self.memory_encoder(torch.where(valid[..., None], shared, 0.)) + self.type_embedding[2]), 1)
        valid = torch.cat((torch.ones(len(state), 1, dtype=torch.bool, device=state.device), valid), 1)
        z = self.decoder(self.action_queries[None].expand(len(state), -1, -1), memory,
                         memory_key_padding_mask=~valid)
        out = self.head(z)
        if self.bypass_enabled:
            out = out + self.bypass(current).reshape(-1, 8, 15)
        return out * self.action_scale + self.action_mean
