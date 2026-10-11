"""Known human/object reference anchors; distinct from measured expert futures."""
import numpy as np
import torch
from torch import nn

from .competent_actor import CompetentActor, actor_windows
from .history import measured_history

SCHEMA = 'ref2dex.reference-trajectory-actor.v1'
OFFSETS = (1, 16)
GOAL_DIM = 2 * (16 * 3 + 16 * 9 + 3 + 9)
INPUT_DIM = 328 + GOAL_DIM


def quaternion_matrix(quaternion):
    q = np.asarray(quaternion, np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if (norm < 1e-8).any() or not np.isfinite(q).all():
        raise ValueError('finite nonzero xyzw reference quaternion required')
    x, y, z, w = np.moveaxis(q / norm, -1, 0)
    return np.stack((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
                     2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
                     2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def reference_from_raw(raw):
    """Same source pose fields as DExplore's HOI loader; no robot q/contact fields."""
    raw = np.asarray(raw, np.float32)
    if raw.ndim != 2 or raw.shape[1] < 245 + 32 * 4:
        raise ValueError('official interaction_hand raw motion required')
    obj = np.broadcast_to(np.eye(4, dtype=np.float32), (len(raw), 4, 4)).copy()
    obj[:, :3, 3], obj[:, :3, :3] = raw[:, 198:201], quaternion_matrix(raw[:, 201:205])
    return dict(hand_pos=raw[:, 102 + 16 * 3:102 + 32 * 3].reshape(-1, 16, 3).copy(),
                hand_rot=quaternion_matrix(raw[:, 245 + 16 * 4:245 + 32 * 4].reshape(-1, 16, 4)), obj=obj)


def reference_from_processed(processed):
    """Read only the matching reference poses from native task.hoi_data."""
    processed = np.asarray(processed, np.float32)
    obj = np.broadcast_to(np.eye(4, dtype=np.float32), (len(processed), 4, 4)).copy()
    obj[:, :3, 3], obj[:, :3, :3] = processed[:, 106:109], quaternion_matrix(processed[:, 109:113])
    return dict(hand_pos=processed[:, 119:167].reshape(-1, 16, 3).copy(),
                hand_rot=quaternion_matrix(processed[:, 232:296].reshape(-1, 16, 4)), obj=obj)


def reference_goal(current_obj, reference, ticks):
    current_obj = np.asarray(current_obj, np.float32)
    ticks = np.broadcast_to(np.asarray(ticks, np.int64), (len(current_obj),))
    rotation, center = current_obj[:, :3, :3], current_obj[:, :3, 3]
    fields = []
    for offset in OFFSETS:
        index = np.minimum(ticks + offset, len(reference['obj']) - 1)
        hp = np.einsum('npj,njk->npk', reference['hand_pos'][index] - center[:, None], rotation)
        hr = np.matmul(rotation.swapaxes(-1, -2)[:, None], reference['hand_rot'][index])
        op = np.einsum('nj,njk->nk', reference['obj'][index, :3, 3] - center, rotation)
        orn = np.matmul(rotation.swapaxes(-1, -2), reference['obj'][index, :3, :3])
        fields.extend((hp.reshape(-1, 48), hr.reshape(-1, 144), op, orn.reshape(-1, 9)))
    return np.concatenate(fields, -1).astype(np.float32)


def reference_input(obj, hand, q, dq, velocity, reference, ticks):
    return np.concatenate((measured_history(obj, hand, q, dq, velocity),
                           reference_goal(obj[-1], reference, ticks)), -1)


def reference_windows(data, envs, reference):
    old, state, tau, target, ticks = actor_windows(data, envs)
    ids = np.tile(envs, len(ticks) // len(envs))
    x = np.concatenate((old[:, :328], reference_goal(data['obj'][ticks, ids], reference, ticks)), -1)
    return x, state, tau, target, ticks


class ReferenceActor(CompetentActor):
    def __init__(self, width=256):
        super().__init__(width)
        self.input_mean = torch.zeros(INPUT_DIM)
        self.input_scale = torch.ones(INPUT_DIM)
        self.net[0] = nn.Linear(INPUT_DIM, width)

    def forward(self, inputs):
        if inputs.shape != (len(inputs), INPUT_DIM):
            raise ValueError('H328 + reference human/object anchors408 required')
        return self.net((inputs - self.input_mean) / self.input_scale).reshape(-1, 24, 33) * self.tau_scale + self.tau_mean


def load_reference_actor(packet, device):
    if packet.get('schema') != SCHEMA:
        raise ValueError('reference-conditioned actor checkpoint required')
    model = ReferenceActor(packet['width']).to(device)
    model.load_state_dict(packet['model'], strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('nonfinite reference model')
    if (model.input_scale <= 0).any() or (model.tau_scale <= 0).any():
        raise ValueError('nonpositive reference normalization')
    return model.eval()
