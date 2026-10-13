"""Prefetching preserves rank/step draws, order and the model's RNG state."""
from types import SimpleNamespace

import numpy as np
import torch

from ptv3_shared_wam.loading import StepBatchSampler, pretrain_loader, project_ipc_directory
from oakink_wm.multisource import mixed_indices


class ToyWindows(torch.utils.data.Dataset):
    def __init__(self):
        groups = [np.arange(12), np.arange(12, 16), np.arange(16, 20)]
        sources = [SimpleNamespace(groups=groups) for _ in range(3)]
        self.base = SimpleNamespace(sources=sources, source_weights=np.array([5., 2., 2.]) / 9,
                                    offsets=np.array([0, 20, 40, 60]))

    def __len__(self):
        return 60

    def __getitem__(self, index):
        # Identity-dependent values reveal row reordering, duplication or loss.
        return torch.tensor([index, index * 2, index % 7])


def toy_collate(rows):
    return torch.stack(rows)


def test_sampler_matches_existing_draw_and_resume_suffix_for_each_rank():
    dataset = ToyWindows()
    for rank in range(4):
        batches = list(StepBatchSampler(dataset, 16, 100, rank, 1, 8))
        for step, batch in enumerate(batches, 1):
            seed = int(np.random.SeedSequence([100, step, rank]).generate_state(1)[0])
            assert batch == mixed_indices(dataset.base, 16, seed).tolist()
        assert list(StepBatchSampler(dataset, 16, 100, rank, 5, 8)) == batches[4:]


def test_spawn_workers_preserve_batch_order_and_global_torch_rng():
    dataset = ToyWindows()
    synchronous = pretrain_loader(dataset, 16, 100, 2, 5, 8, collate_fn=toy_collate)
    expected = list(synchronous)
    before = torch.get_rng_state().clone()
    asynchronous = pretrain_loader(dataset, 16, 100, 2, 5, 8,
                                    workers=2, prefetch_factor=2, collate_fn=toy_collate)
    actual = list(asynchronous)
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert len(actual) == len(expected) == 4


def test_worker_ipc_uses_short_project_socket_path():
    from multiprocessing.connection import Listener
    directory = project_ipc_directory()
    address = directory / "listener-abcdefgh"
    assert len(str(address).encode()) < 108
    listener = Listener(str(address))
    listener.close()
