"""GT trajectory-conditioned native action chunk; replan and execute only a0."""
import numpy as np
import torch
from torch import nn

from .learned_retargeter import LearnedRetargeter
from .retargeter_data import ACTIVE, features, build_windows

SCHEMA = 'ref2dex.wam-chunk-native-R.v1'


class ChunkRetargeter(LearnedRetargeter):
    def __init__(self, width=128):
        super().__init__(width, use_tau=True)
        self.tau_encoder = nn.Sequential(nn.Linear(33, width), nn.GELU(), nn.Linear(width, width))


def chunk_windows(data, envs):
    """543 measured states,542 actually applied controls; no missing tail padding."""
    controls = len(data['applied'])
    if (not np.array_equal(data['action'], data['applied'])
            or any(len(data[k]) != controls+1 for k in ('q', 'dq', 'hand', 'obj', 'velocity'))):
        raise ValueError('actual applied actions and terminal measured state required')
    subset = {k: data[k][:, envs] for k in ('q','dq','hand','obj','velocity','action','applied')}
    subset.update(episode_tick=np.arange(controls+1), episode_start=np.zeros(controls+1, np.int64))
    state, tau, action = build_windows(subset, np.arange(controls+1-24))
    return state.reshape(-1,87), tau.reshape(-1,24,33), action.reshape(-1,8,12)


def expert_chunk_rows(data, takeover):
    """Only own role3 future and actually applied expert8; pure rows excluded."""
    ticks = np.arange(takeover, len(data['applied'])-23)
    if (not data['expert_applied'][takeover:,1].all()
            or data['expert_clipped'][takeover:,1].any()
            or not np.array_equal(data['applied'][takeover:,3], data['expert_action'][takeover:,1])):
        raise ValueError('actual unclipped own expert control required')
    future = data['hand'][ticks[:,None]+np.arange(1,25),3]
    state, tau = features(*(data[k][ticks,3] for k in ('q','dq','hand','obj','velocity')), future)
    actions = data['applied'][ticks[:,None]+np.arange(8),3][...,list(ACTIVE)]
    return state, tau, actions.astype(np.float32)


def load_chunk(packet, device):
    if packet.get('schema') != SCHEMA or packet.get('width') != 128:
        raise ValueError('chunk action R schema/width differs')
    net = ChunkRetargeter(packet['width']).to(device)
    net.load_state_dict(packet['model'], strict=True)
    if (any(not torch.isfinite(v).all() for v in net.state_dict().values())
            or any((getattr(net,k)<=0).any() for k in ('state_scale','tau_scale','action_scale'))):
        raise ValueError('finite chunk R and positive normalization required')
    return net.eval()
