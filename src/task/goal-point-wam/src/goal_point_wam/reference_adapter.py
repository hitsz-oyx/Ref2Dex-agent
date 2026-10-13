"""Demonstration hand/object anchors for the robot trajectory query.

The 408-D reference is the successful baseline contract: human hand pose and
object pose at t+1 and t+16, in the current object frame. It is not a sparse
wrist point and not a measured future robot state.
"""
import torch
from torch import nn

from .model import GoalPointWAM

REFERENCE_DIM = 408


class ReferenceAdapter(nn.Module):
    def __init__(self, hidden=128, reference_dim=REFERENCE_DIM):
        super().__init__()
        self.prop = nn.Sequential(nn.Linear(48, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.reference = nn.Sequential(nn.Linear(reference_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.alpha_p = nn.Parameter(torch.zeros(()))
        self.alpha_g = nn.Parameter(torch.zeros(()))

    def forward(self, prop, reference_goal, robot_mask):
        if prop.ndim != 2 or prop.shape[-1] != 48 or reference_goal.shape != (len(prop), REFERENCE_DIM):
            raise ValueError('measured proprio Bx48 and demonstration reference Bx408 required')
        if robot_mask.shape != (len(prop),) or robot_mask.dtype != torch.bool:
            raise ValueError('one boolean robot mask per sample required')
        prop = torch.where(robot_mask[:, None], prop, 0.)
        reference_goal = torch.where(robot_mask[:, None], reference_goal, 0.)
        residual = self.alpha_p * self.prop(prop) + self.alpha_g * self.reference(reference_goal)
        return torch.where(robot_mask[:, None], residual, 0.)


class ReferenceStudentWAM(GoalPointWAM):
    """Three-domain initialization. Adapter and trajectory learn; World stays frozen and unused."""

    def __init__(self, stats, config):
        super().__init__(stats, config)
        with torch.random.fork_rng(devices=[]):
            self.reference_adapter = ReferenceAdapter()
        self.requires_grad_(False)
        self.reference_adapter.requires_grad_(True)
        self.trajectory.requires_grad_(True)
        if self.trajectory.goal_query is not None and self.trajectory.goal_injection != 'query':
            self.trajectory.goal_query.requires_grad_(False)
        self.eval()

    def contexts(self, batch):
        with torch.no_grad():
            scene = self.scene(batch)
            goal, mask = self.goal(batch['start_object_points'], batch['goal_object_points'],
                                   batch['object_goal_valid'])
            raw = self.observed_hand_condition(batch)
        return scene, goal, mask, raw

    def actor(self, context, batch):
        scene, goal, mask, raw = context
        if raw is None:
            raise ValueError('reference adapter requires the raw-hand query')
        residual = self.reference_adapter(
            batch['proprio'], batch['reference_goal'], batch['robot_mask']).to(raw.dtype)
        return self.trajectory(scene['scene_tokens'], scene['scene_mask'], goal, mask,
                               raw + residual[:, None, None])
