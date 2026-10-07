"""DDP scheduling and state helpers independent of the frozen model/entry.

The reference objective is the mean of normalized TWO-sample microbatch
losses, not the normalized loss over all samples in an optimizer update.
Distribute whole microbatches, then average equal numbers per rank. This
preserves masks/weight denominators and the reference spatial grid origins.
"""
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist


def local_accumulation(global_accumulation, world_size):
    if world_size < 1 or global_accumulation < 1 or global_accumulation % world_size:
        raise ValueError('world size must divide the global accumulation count')
    return global_accumulation // world_size


def rank_indices(indices, microbatch, global_accumulation, rank, world_size, step=0):
    local_accumulation(global_accumulation, world_size)
    if not 0 <= rank < world_size or step < 0 or microbatch < 1:
        raise ValueError('invalid rank, step or microbatch')
    indices = np.asarray(indices)
    effective = microbatch * global_accumulation
    if len(indices) % effective or step > len(indices)//effective:
        raise ValueError('draw must contain complete optimizer updates')
    updates = indices[step*effective:].reshape(-1, global_accumulation, microbatch)
    return updates[:, rank::world_size, :].reshape(-1).copy()


def sync_context(model, micro, accumulation):
    return model.no_sync() if micro < accumulation-1 else nullcontext()


def capture_rng():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu())
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def gather_rng(world_size):
    states = [None]*world_size
    dist.all_gather_object(states, capture_rng())
    return states


def any_rank(flag, device):
    value = torch.tensor(int(flag), dtype=torch.int32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())
