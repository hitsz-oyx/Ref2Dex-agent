"""Ordered stateless CPU prefetching, isolated from the model's RNG stream."""
from functools import lru_cache
import multiprocessing
import os
import random
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, get_worker_info

from .data import shared_collate
from .paths import ROOT, add_dependencies

add_dependencies()
from oakink_wm.multisource import mixed_indices


class StepBatchSampler:
    """Exactly the existing draw(seed, step, rank), regardless of worker count."""
    def __init__(self, dataset, batch_size, seed, rank, first_step, last_step):
        self.dataset = dataset if hasattr(dataset, "sample_indices") else dataset.base
        self.batch_size, self.seed, self.rank = batch_size, seed, rank
        self.first_step, self.last_step = first_step, last_step

    def __len__(self):
        return max(0, self.last_step - self.first_step + 1)

    def __iter__(self):
        for step in range(self.first_step, self.last_step + 1):
            seed = int(np.random.SeedSequence([self.seed, step, self.rank]).generate_state(1)[0])
            if hasattr(self.dataset, "sample_indices"):
                yield self.dataset.sample_indices(np.random.default_rng(seed), self.batch_size)
            else:
                yield mixed_indices(self.dataset, self.batch_size, seed).tolist()


def forecasting_collate(samples):
    batch = shared_collate(samples)
    # Legacy transport arrays have already produced all forecasting labels.
    # Keep observations, labels, masks, task goals and sample/source identities.
    for key in ("action", "action_valid", "effect", "points", "radius", "object_features"):
        batch.pop(key, None)
    return batch


def initialize_loader_worker(worker_id):
    """Cache only in spawned workers; no CUDA work and no parent-side mutation."""
    torch.set_num_threads(1)
    project_ipc_directory()
    info = get_worker_info()
    random.seed(info.seed)
    np.random.seed(info.seed % (2**32))
    size = int(getattr(info.dataset, "loader_sequence_cache", 0))
    if size:
        # Each wrapper retains at most size sequence dictionaries per source.
        # It is installed after spawn, so unpicklable caches are never shipped.
        if hasattr(info.dataset, "base"):
            for source in info.dataset.base.sources:
                source.sequence = lru_cache(maxsize=size)(source.sequence)
        elif hasattr(info.dataset, "sequence"):
            info.dataset.sequence = lru_cache(maxsize=size)(info.dataset.sequence)


def project_ipc_directory():
    # Resource-sharer AF_UNIX sockets must fit the Linux108-byte limit. Long
    # run/rank scratch names exceed it; each process gets a short project path.
    directory = ROOT / "tmp" / ("lw-%d" % os.getpid())
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.environ["TMPDIR"] = str(directory)
    tempfile.tempdir = str(directory)
    multiprocessing.current_process()._config["tempdir"] = str(directory)
    return directory


def pretrain_loader(dataset, batch_size, seed, rank, first_step, last_step,
                    workers=0, prefetch_factor=2, sequence_cache=0, pin_memory=False,
                    collate_fn=forecasting_collate):
    if not 0 <= workers <= 8 or not 1 <= prefetch_factor <= 4 or not 0 <= sequence_cache <= 1024:
        raise ValueError("bounded workers[0,8], prefetch[1,4], sequence cache[0,1024] required")
    dataset.loader_sequence_cache = sequence_cache
    if workers:
        project_ipc_directory()
    # DataLoader iterator creation normally consumes the global torch RNG.
    # A separate generator keeps dropout/PTv3 randomness unchanged on resume.
    generator = torch.Generator().manual_seed(seed * 1009 + rank * 101 + 17)
    options = dict(dataset=dataset,
                   batch_sampler=StepBatchSampler(dataset, batch_size, seed, rank, first_step, last_step),
                   num_workers=workers, collate_fn=collate_fn, pin_memory=pin_memory,
                   generator=generator)
    if workers:
        options.update(multiprocessing_context="spawn", persistent_workers=True,
                       prefetch_factor=prefetch_factor, worker_init_fn=initialize_loader_worker,
                       timeout=120)
    return DataLoader(**options)


def to_device(batch, device, non_blocking=False):
    return {key: value.to(device, non_blocking=non_blocking) for key, value in batch.items()}
