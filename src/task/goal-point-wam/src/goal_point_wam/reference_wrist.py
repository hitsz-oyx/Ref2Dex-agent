"""Optional planned human wrist conditioning, not a robot position command."""
import numpy as np
import torch
from torch import nn


def wrist_context(current_object, positions, rotations):
    """Three planned world poses expressed in the actual current object frame."""
    obj, pos, rot = [np.asarray(x, np.float32) for x in (current_object, positions, rotations)]
    if obj.shape != (4, 4) or pos.shape != (3, 3) or rot.shape != (3, 3, 3):
        raise ValueError('current object and three planned wrist poses required')
    local = (pos - obj[:3, 3]) @ obj[:3, :3]
    orient = obj[:3, :3].T @ rot
    result = np.concatenate((local, orient[:, :, 0], orient[:, :, 1]), -1).reshape(27)
    if not np.isfinite(result).all():
        raise ValueError('finite planned wrist context required')
    return result.astype(np.float32)


class ReferenceWristAdapter(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(27, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, goal, robot_mask):
        if goal.shape != (len(robot_mask), 27) or robot_mask.ndim != 1:
            raise ValueError('Bx27 wrist goal and B robot mask required')
        safe = torch.where(robot_mask[:, None], goal, 0.)
        return torch.where(robot_mask[:, None], self.alpha * self.net(safe), 0.)


def wrist_actor(model, adapter, context, batch, mode='planned'):
    scene, goal, mask, raw = context
    control = model.control_adapter(batch['proprio'], batch['robot_goal'], batch['robot_mask'])
    if mode == 'planned':
        wrist = batch['wrist_goal']
    elif mode == 'shift16':
        wrist = batch['wrist_shift16']
    elif mode == 'blank':
        wrist = torch.zeros_like(batch['wrist_goal'])
    else:
        raise ValueError('unknown reference intervention')
    residual = adapter(wrist, batch['robot_mask']).to(raw.dtype)
    # Right is WAM hand0. The human wrist is a learned condition only; no
    # literal anatomical-to-robot landmark correspondence is asserted.
    right_only = torch.stack((residual, torch.zeros_like(residual)), 1)[:, :, None]
    return model.trajectory(scene['scene_tokens'], scene['scene_mask'], goal, mask,
                            raw + control.to(raw.dtype)[:, None, None] + right_only)
