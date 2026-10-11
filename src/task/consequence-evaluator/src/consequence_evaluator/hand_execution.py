"""Decision-time hand execution bridge, with explicit common wrist motion."""
import numpy as np
import torch
from torch import nn

SCHEMA = 'ref2dex.hand-execution.v1'


def current_frame(object_history, hand_history):
    """Only four measured past/current states enter this operation."""
    obj = np.asarray(object_history)
    hand = np.asarray(hand_history)
    inverse = np.linalg.inv(obj[-1])
    objects = inverse[None] @ obj
    hands = (hand - obj[-1, :3, 3]) @ obj[-1, :3, :3]
    return objects.astype('float32'), hands.astype('float32')


def transform_future(object_current, world_hand):
    return ((world_hand-object_current[:3, 3]) @ object_current[:3, :3]).astype('float32')


def split_motion(current, future):
    wrist = future[:, 0] - current[0]
    relative = future[:, 1:] - future[:, :1] - (current[1:] - current[:1])[None]
    return np.concatenate((wrist, relative.reshape(24, 30)), -1).astype('float32')


def compose_motion(current, motion):
    wrist = current[..., None, 0, :] + motion[..., :3]
    finger = wrist[..., None, :] + (current[..., None, 1:, :] - current[..., None, :1, :])
    finger = finger + motion[..., 3:].reshape(*motion.shape[:-1], 10, 3)
    return torch.cat((wrist[..., None, :], finger), -2)


class HandExecution(nn.Module):
    """Common translation and wrist-relative point changes have separate heads.

    Rotation of the wrist/finger shape is represented by relative point changes;
    no future measured root, object, force, or executed action is an input.
    """
    def __init__(self, width=256):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(1442+132, width), nn.SiLU(),
                                   nn.Linear(width, width), nn.SiLU())
        self.plan = nn.Sequential(nn.Linear(432, width//2), nn.SiLU())
        self.join = nn.Sequential(nn.Linear(width+width//2, width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU())
        self.wrist = nn.Linear(width, 24*3)
        self.relative = nn.Linear(width, 24*30)

    def forward(self, state, plan):
        if state.shape[1:] != (1574,) or plan.shape[1:] != (24, 18):
            raise ValueError('current H/geometry and known residual plan required')
        z = self.join(torch.cat((self.state(state), self.plan(plan.flatten(1))), -1))
        return torch.cat((self.wrist(z).reshape(-1, 24, 3),
                          self.relative(z).reshape(-1, 24, 30)), -1)


def metrics(prediction, target):
    delta = prediction-target
    relative_delta = (prediction[:, :, 1:]-prediction[:, :, :1]) - (target[:, :, 1:]-target[:, :, :1])
    return dict(point_rmse_m=float(np.sqrt(np.mean(np.sum(delta**2, -1)))),
                h8_rmse_m=float(np.sqrt(np.mean(np.sum(delta[:, 7]**2, -1)))),
                h24_rmse_m=float(np.sqrt(np.mean(np.sum(delta[:, -1]**2, -1)))),
                wrist_rmse_m=float(np.sqrt(np.mean(np.sum(delta[:, :, 0]**2, -1)))),
                relative_rmse_m=float(np.sqrt(np.mean(np.sum(relative_delta**2, -1)))))
