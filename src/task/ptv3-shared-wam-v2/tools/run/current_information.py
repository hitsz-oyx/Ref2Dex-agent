"""Current-tick reference/geometry information for an isolated R-only Probe.

This module stays outside production src so the active WAM run is unchanged.
Reference information is privileged, explicitly permitted for this diagnostic.
"""
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch
from torch import nn

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from ptv3_shared_wam_v2 import paths as _paths
from ptv3_shared_wam.action_targets import GEOMETRY_LIFTED, encode_targets
from ptv3_shared_wam.target_retargeter import make_retargeter
from trajectory_policy.retargeter_data import features

REFERENCE_DIM, GEOMETRY_DIM = 24, 33
EXTRA_DIM = REFERENCE_DIM + GEOMETRY_DIM
ARMS = dict(baseline=(0, 0), current_reference=(1, 0), current_geometry=(0, 1), both=(1, 1))


def current_information(q, hand, obj, q_ref, object_ref, surface):
    """Read ONLY the supplied current-tick arrays and the fixed metric CAD.

    Returns relative reference-q15, reference-object pose9 and actual nearest
    surface displacement33 (hand - nearest CAD, not a contact-force label).
    """
    q, hand, obj, q_ref, object_ref = [np.asarray(v, dtype=np.float32)
                                     for v in (q, hand, obj, q_ref, object_ref)]
    n = len(q)
    if (q.shape != (n, 18) or hand.shape != (n, 11, 3) or obj.shape != (n, 4, 4)
            or q_ref.shape != q.shape or object_ref.shape != obj.shape):
        raise ValueError('matching current-frame q/hand/object/reference arrays required')
    if not all(np.isfinite(v).all() for v in (q, hand, obj, q_ref, object_ref, surface)):
        raise ValueError('finite current information required')
    repeated = np.repeat(q_ref[:, None], 8, axis=1)
    reference_q = encode_targets(repeated, q, GEOMETRY_LIFTED)[:, 0]
    inverse_rotation = obj[:, :3, :3].swapaxes(-1, -2)
    translation = np.einsum('nij,nj->ni', inverse_rotation, object_ref[:, :3, 3]-obj[:, :3, 3])
    relative_rotation = inverse_rotation @ object_ref[:, :3, :3]
    rotation6 = relative_rotation[:, :, :2].transpose(0, 2, 1).reshape(n, 6)
    reference = np.concatenate((reference_q, translation, rotation6), -1)
    local_hand = np.matmul(hand-obj[:, None, :3, 3], obj[:, :3, :3])
    surface = np.asarray(surface, dtype=np.float32)
    nearest = cKDTree(surface).query(local_hand.reshape(-1, 3), workers=1)[1]
    geometry = (local_hand-surface[nearest].reshape(n, 11, 3)).reshape(n, GEOMETRY_DIM)
    return np.concatenate((reference, geometry), -1).astype(np.float32)


def pack(dataset):
    """Same retained windows as the active scientific run; no WAM inference."""
    n = len(dataset)
    result = dict(state=np.empty((n, 87), np.float32), tau=np.empty((n, 24, 33), np.float32),
                  target=np.empty((n, 8, 15), np.float32), pd=np.empty((n, 18), np.float32),
                  extra=np.empty((n, EXTRA_DIM), np.float32), tick=np.empty(n, np.int64),
                  sequence=np.empty(n, np.int64))
    for seq, (record, ticks, rows) in enumerate(zip(dataset.records, dataset.ticks_by_sequence, dataset.pools)):
        if not len(rows):
            continue
        data, canonical = dataset.sequence(seq)
        result['state'][rows], result['tau'][rows] = features(
            *(data[k][ticks] for k in ('q', 'dq', 'hand', 'obj', 'velocity')),
            data['hand'][ticks[:, None]+np.arange(1, 25)])
        result['pd'][rows] = data['pd_targets'][ticks]
        result['target'][rows] = encode_targets(data['pd_targets'][ticks[:, None]+np.arange(8)],
                                               data['q'][ticks], GEOMETRY_LIFTED)
        # Both references are indexed at t. No t+1/t+16 reference is read here.
        result['extra'][rows] = current_information(data['q'][ticks], data['hand'][ticks], data['obj'][ticks],
            dataset.reference_sequence(seq)[ticks], data['reference_obj'][ticks], canonical['points'])
        result['tick'][rows], result['sequence'][rows] = ticks, seq
    return result


class InformationRetargeter(nn.Module):
    """Identical R and extra-encoder capacity for every masked-information arm."""
    def __init__(self, r_state, extra_mean, extra_scale):
        super().__init__()
        self.base = make_retargeter(GEOMETRY_LIFTED)
        self.base.load_state_dict(r_state, strict=True)
        width = self.base.action_queries.shape[-1]
        self.extra = nn.Sequential(nn.Linear(EXTRA_DIM, width), nn.GELU(), nn.Linear(width, width))
        nn.init.zeros_(self.extra[-1].weight)
        nn.init.zeros_(self.extra[-1].bias)
        self.register_buffer('extra_mean', torch.as_tensor(extra_mean, dtype=torch.float32))
        self.register_buffer('extra_scale', torch.as_tensor(extra_scale, dtype=torch.float32))

    def forward(self, state, tau, extra, arm):
        reference, geometry = ARMS[arm]
        mask = extra.new_tensor([reference]*REFERENCE_DIM+[geometry]*GEOMETRY_DIM)
        extra = ((extra-self.extra_mean)/self.extra_scale)*mask
        net = self.base
        state_token = net.state_encoder((state-net.state_mean)/net.state_scale)+self.extra(extra)
        memory = torch.cat((net.tau_encoder((tau-net.tau_mean)/net.tau_scale)+net.step_embedding,
                            state_token[:, None]), 1)
        queries = net.action_queries[None].expand(len(state), -1, -1)
        return net.head(net.decoder(queries, memory))*net.action_scale+net.action_mean
