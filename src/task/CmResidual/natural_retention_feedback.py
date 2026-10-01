"""Current-state-only natural slip trigger and bounded native target feedback."""
import torch
from src.task.CmResidual.finger_preload import preload_target


def groups(motion, seed):
    rng = torch.Generator(device='cpu').manual_seed(seed+15000)
    out = torch.full((len(motion),), -1, dtype=torch.long)
    for m in range(3):
        ids = (motion.cpu() == m).nonzero().flatten()
        if len(ids) != 256:
            raise ValueError('balanced768env required')
        out[ids] = torch.arange(4).repeat_interleave(64)[torch.randperm(256,generator=rng)]
    return out


class NaturalRetentionFeedback:
    def __init__(self, group, initial_height, lift_start, early_step):
        self.group = group
        self.initial_height = initial_height
        self.lift_start = lift_start
        self.early_step = early_step
        self.run = torch.zeros_like(group)
        self.acquired = torch.zeros_like(group, dtype=torch.bool)
        self.event_tick = torch.full_like(group, -1)
        self.arrest_translation = torch.zeros((len(group),3),device=group.device)

    def target(self, base, q, dq, object_root, clearance, progress, lower, upper):
        valid = ((object_root[:,2]-self.initial_height >= .03) & (clearance >= .02)
                 & (progress >= self.lift_start))
        self.run = torch.where(valid, self.run+1, torch.zeros_like(self.run))
        self.acquired |= self.run >= 5
        trigger = (self.event_tick < 0) & self.acquired & valid & (object_root[:,9]-dq[:,2] <= -.02)
        self.event_tick[trigger] = progress[trigger]
        self.arrest_translation[trigger] = q[trigger,:3]
        curled = ((self.group == 1) & (progress >= self.early_step)) | ((self.group == 2) & (self.event_tick >= 0))
        goal = preload_target(base, curled.to(base.dtype)*.15, lower, upper)
        arrested = (self.group == 3) & (self.event_tick >= 0)
        goal[arrested,:3] = self.arrest_translation[arrested]
        return goal
