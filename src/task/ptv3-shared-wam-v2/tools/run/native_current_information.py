"""Isolated native-delta0 information adapter; production WAM code is untouched."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from ptv3_shared_wam_v2 import paths as _paths
from ptv3_shared_wam.action_targets import GEOMETRY_LIFTED, encode_targets
from ptv3_shared_wam.target_retargeter import make_retargeter
from trajectory_policy.retargeter_data import features

EXTRA_DIM = 721
ARMS = ('baseline', 'native_current')


def current_rows(data, ticks):
    """Read the captured delta0 branch at t, never a teacher future branch."""
    obs = np.asarray(data['native_current_obs'])
    if obs.shape != (len(data['q']), EXTRA_DIM) or not np.isfinite(obs).all():
        raise ValueError('finite same-execution delta0 721D observations required')
    ticks = np.asarray(ticks, dtype=np.int64)
    if not np.array_equal(data['frame_tick'], np.arange(len(obs))):
        raise ValueError('continuous native frame clock required')
    return obs[ticks].astype(np.float32, copy=True)


def pack(dataset):
    n = len(dataset)
    result = dict(state=np.empty((n,87),np.float32),tau=np.empty((n,24,33),np.float32),
                  target=np.empty((n,8,15),np.float32),pd=np.empty((n,18),np.float32),
                  extra=np.empty((n,EXTRA_DIM),np.float32),tick=np.empty(n,np.int64),
                  sequence=np.empty(n,np.int64))
    for seq,(record,ticks,rows) in enumerate(zip(dataset.records,dataset.ticks_by_sequence,dataset.pools)):
        if not len(rows): continue
        data,_ = dataset.sequence(seq)
        result['state'][rows],result['tau'][rows] = features(
            *(data[k][ticks] for k in ('q','dq','hand','obj','velocity')),
            data['hand'][ticks[:,None]+np.arange(1,25)])
        result['pd'][rows] = data['pd_targets'][ticks]
        result['target'][rows] = encode_targets(data['pd_targets'][ticks[:,None]+np.arange(8)],data['q'][ticks],GEOMETRY_LIFTED)
        result['extra'][rows] = current_rows(data,ticks)
        result['tick'][rows],result['sequence'][rows] = ticks,seq
    return result


class NativeInformationRetargeter(nn.Module):
    """Same parameter capacity in both arms; mask only the native input values."""
    def __init__(self,r_state,extra_mean,extra_scale):
        super().__init__()
        self.base=make_retargeter(GEOMETRY_LIFTED)
        self.base.load_state_dict(r_state,strict=True)
        width=self.base.action_queries.shape[-1]
        self.extra=nn.Sequential(nn.Linear(EXTRA_DIM,width),nn.GELU(),nn.Linear(width,width))
        nn.init.zeros_(self.extra[-1].weight);nn.init.zeros_(self.extra[-1].bias)
        self.register_buffer('extra_mean',torch.as_tensor(extra_mean,dtype=torch.float32))
        self.register_buffer('extra_scale',torch.as_tensor(extra_scale,dtype=torch.float32))
    def forward(self,state,tau,extra,arm):
        if arm not in ARMS: raise ValueError(arm)
        extra=(extra-self.extra_mean)/self.extra_scale
        if arm=='baseline': extra=torch.zeros_like(extra)
        net=self.base
        state_token=net.state_encoder((state-net.state_mean)/net.state_scale)+self.extra(extra)
        memory=torch.cat((net.tau_encoder((tau-net.tau_mean)/net.tau_scale)+net.step_embedding,state_token[:,None]),1)
        queries=net.action_queries[None].expand(len(state),-1,-1)
        return net.head(net.decoder(queries,memory))*net.action_scale+net.action_mean
