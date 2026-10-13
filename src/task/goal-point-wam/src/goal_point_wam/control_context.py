"""Robot-only zero-gated conditioning of the existing raw-hand decoder query."""
import torch
from torch import nn

from .effect_consistency import FrozenEffectWAM


class RobotControlAdapter(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.prop = nn.Sequential(nn.Linear(48, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.robot_goal = nn.Sequential(nn.Linear(27, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.alpha_p = nn.Parameter(torch.zeros(()))
        self.alpha_g = nn.Parameter(torch.zeros(()))

    def forward(self, prop, robot_goal, robot_mask):
        if prop.ndim != 2 or prop.shape[-1] != 48 or robot_goal.shape != (len(prop), 27):
            raise ValueError('robot context must be Bx48 and Bx27')
        if robot_mask.shape != (len(prop),):
            raise ValueError('one robot mask per sample required')
        # Sanitize masked values before the MLP too: absent human inputs cannot
        # leak or create NaN gradients even if represented by sentinels.
        prop = torch.where(robot_mask[:, None], prop, 0.)
        robot_goal = torch.where(robot_mask[:, None], robot_goal, 0.)
        residual = self.alpha_p * self.prop(prop) + self.alpha_g * self.robot_goal(robot_goal)
        return torch.where(robot_mask[:, None], residual, 0.)


def conditioned_actor(model, context, adapter, batch):
    scene, goal, mask, raw = context
    if raw is None:
        raise ValueError('robot conditioner requires the preserved raw-hand interface')
    residual = adapter(batch['proprio'], batch['robot_goal'], batch['robot_mask']).to(raw.dtype)
    # The same residual is broadcast to all existing hand/keypoint/time queries.
    # No module, token, parameter key or memory in the old decoder is replaced.
    return model.trajectory(scene['scene_tokens'], scene['scene_mask'], goal, mask,
                            raw + residual[:, None, None])


class ControlContextWAM(FrozenEffectWAM):
    def __init__(self, stats, config):
        super().__init__(stats, config)
        with torch.random.fork_rng(devices=[]):
            self.control_adapter = RobotControlAdapter()
        self.set_stage(1)

    def set_stage(self, stage):
        if stage not in (1, 2):
            raise ValueError('adapter stage1 or decoder stage2 required')
        self.requires_grad_(False)
        self.control_adapter.requires_grad_(True)
        if stage == 2:
            self.trajectory.requires_grad_(True)
            if self.trajectory.goal_query is not None and self.trajectory.goal_injection != 'query':
                self.trajectory.goal_query.requires_grad_(False)
        self.eval()

    def actor(self, context, batch):
        return conditioned_actor(self, context, self.control_adapter, batch)

    def forward(self, batch):
        context = self.contexts(batch)
        trajectory = self.actor(context, batch)
        return dict(trajectory=trajectory, **self.induced(trajectory, batch, context[0]))
