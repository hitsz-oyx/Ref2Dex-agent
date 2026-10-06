"""Training-only eight-step GT targets; no simulator or PPO dependency."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

HORIZON = 8
TARGET_DIM = 19


def physical_state(task):
    """Object xyz/xyzw, hand contact-link centroid, five force-gated contacts."""
    links = task._contact_body_ids
    contact = ((task._contact_forces[:, links].norm(dim=-1) > .1) &
               (task._tar_contact_forces.norm(dim=-1, keepdim=True) > .1))
    return torch.cat((task._target_states[:, :7].clone(),
                      task._rigid_body_pos[:, links].mean(1), contact.float()), -1)


def interaction_target(before, after):
    if before.shape != after.shape or before.shape[-1] != 15:
        raise ValueError('GT interaction state must be matching [...,15]')
    q0 = F.normalize(before[..., 3:7], dim=-1)
    q1 = F.normalize(after[..., 3:7], dim=-1)
    # q1 * inverse(q0), sign-invariant small-angle rotation vector.
    v0, w0 = q0[..., :3], q0[..., 3:4]
    v1, w1 = q1[..., :3], q1[..., 3:4]
    vector = -w1*v0 + w0*v1 - torch.cross(v1, v0, dim=-1)
    scalar = w1*w0 + (v1*v0).sum(-1, keepdim=True)
    vector = torch.where(scalar < 0, -vector, vector)
    obj = (after[..., :3] - before[..., :3]) / .03
    relative0 = before[..., :3] - before[..., 7:10]
    relative1 = after[..., :3] - after[..., 7:10]
    target = torch.cat((obj, 2*vector/.5, (relative1-relative0)/.03,
                        after[..., 10:15], after[..., 10:15]-before[..., 10:15]), -1)
    if not torch.isfinite(target).all():
        raise FloatingPointError('nonfinite GT interaction target')
    return target


def rollout_targets(before, after, actions, dones):
    """Same-episode forward windows only, including the last transition done."""
    t, n, _ = before.shape
    if after.shape != before.shape or actions.shape != (t, n, 18) or dones.shape != (t, n):
        raise ValueError('rollout shape mismatch')
    target = before.new_zeros(t, n, TARGET_DIM)
    chunks = actions.new_zeros(t, n, HORIZON, 18)
    mask = before.new_zeros(t, n, 1)
    for start in range(t-HORIZON+1):
        end = start+HORIZON-1
        target[start] = interaction_target(before[start], after[end])
        chunks[start] = actions[start:start+HORIZON].transpose(0, 1).clamp(-1, 1)
        mask[start, :, 0] = ~dones[start:start+HORIZON].bool().any(0)
    return target, chunks, mask


class InteractionDecoder(nn.Module):
    def __init__(self, width=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width+HORIZON*18, 128), nn.ReLU(),
                                 nn.Linear(128, TARGET_DIM))

    def forward(self, latent, action_chunk, detach=False):
        latent = latent.detach() if detach else latent
        return self.net(torch.cat((latent, action_chunk.flatten(1)), -1))


def auxiliary_rows(prediction, target, mask):
    # Preserve the mean over valid samples after PPO averages all rows.
    rows = F.smooth_l1_loss(prediction, target, reduction='none').mean(-1)
    weights = mask.squeeze(-1)
    return rows * weights / weights.mean().clamp_min(1e-6)
