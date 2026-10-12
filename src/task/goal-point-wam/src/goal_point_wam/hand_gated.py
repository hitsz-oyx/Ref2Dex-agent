"""Warm-start preserving PTv3 hand residual, independently of scene sampling."""
import torch
from torch import nn

from .hand_anchored import HandAnchoredWAM, gather_hand_tokens
from .model import HIDDEN


class GatedHandResidual(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.projection = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                        nn.Linear(hidden, hidden))
        # Zero only the gate: zeroing both gate and projection would block learning.
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, hand, valid):
        projected = self.projection(self.norm(hand))
        return self.alpha * projected.masked_fill(~valid[..., None], 0)


class GatedHandAnchoredWAM(HandAnchoredWAM):
    def __init__(self, stats, config):
        super().__init__(stats, config)
        # Adding a module must not change the historical initialization/draw stream.
        with torch.random.fork_rng(devices=[]):
            self.hand_residual = GatedHandResidual()

    def actor_from_scene(self, scene, batch):
        goal, mask = self.goal(batch['start_object_points'], batch['goal_object_points'],
                               batch['object_goal_valid'])
        hand, valid = gather_hand_tokens(scene, batch)
        raw = self.observed_hand_condition(batch)
        if raw is None:
            raise ValueError('gated adaptation requires the existing raw-hand query')
        residual = self.hand_residual(hand, valid).to(raw.dtype)
        return self.trajectory(scene['scene_tokens'], scene['scene_mask'], goal, mask,
                               raw + residual)
