"""Predict hand displacement through the fixed, goal-blind executed World."""
import torch

from .model import GoalPointWAM


def action_from_delta(delta, start, valid):
    """9D = future XYZ, current-relative displacement, previous-step increment."""
    previous = torch.cat((torch.zeros_like(delta[:, :1]), delta[:, :-1]), 1)
    previous_valid = torch.cat((valid[:, :1], valid[:, :-1]), 1)
    mask = valid & previous_valid
    action = torch.cat((start[:, None] + delta, delta, delta - previous), -1)
    return torch.where(mask[..., None], action, 0.), mask


class FrozenEffectWAM(GoalPointWAM):
    """C freezes B's interface and the complete oracle; only decoder learns."""
    def __init__(self, stats, config):
        super().__init__(stats, config)
        self.requires_grad_(False)
        self.trajectory.requires_grad_(True)
        self.eval()

    def contexts(self, batch):
        with torch.no_grad():
            scene = self.scene(batch)
            goal, mask = self.goal(batch['start_object_points'], batch['goal_object_points'],
                                   batch['object_goal_valid'])
            raw = self.observed_hand_condition(batch)
        return scene, goal, mask, raw

    def actor(self, context):
        scene, goal, mask, raw = context
        return self.trajectory(scene['scene_tokens'], scene['scene_mask'], goal, mask, raw)

    def induced(self, delta, batch, scene):
        # Validity is a supervision mask, never an actor/query input.
        action, valid = action_from_delta(delta, batch['start_hand'], batch['trajectory_valid'])
        tokens, mask = self.action(action, valid)
        return self.world(scene['scene_tokens'].detach(), scene['scene_mask'],
                          scene['object_tokens'].detach(), scene['object_mask'], tokens, mask)

    @torch.no_grad()
    def oracle(self, batch, scene):
        # Preserve exact recorded teacher actions, including their own masks.
        tokens, mask = self.action(batch['action'].reshape(-1, 24, 2, 11, 9),
                                   batch['action_valid'].reshape(-1, 24, 2, 11))
        return self.world(scene['scene_tokens'].detach(), scene['scene_mask'],
                          scene['object_tokens'].detach(), scene['object_mask'], tokens, mask)

    def forward(self, batch):
        context = self.contexts(batch)
        trajectory = self.actor(context)
        world = self.induced(trajectory, batch, context[0])
        return dict(trajectory=trajectory, **world)
