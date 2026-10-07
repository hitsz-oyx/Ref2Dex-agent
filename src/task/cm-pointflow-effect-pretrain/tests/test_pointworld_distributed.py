"""DDP regression: matched microbatch objective, sampling, RNG and stop collectives."""
import importlib.util
import os
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK/'src'))
from oakink_wm.distributed import (rank_indices, local_accumulation, sync_context,
                                  gather_rng, restore_rng, any_rank)


@pytest.mark.parametrize('world', [1, 2, 4])
def test_rank_schedule_preserves_reference_pairs_and_resume(world):
    draw = np.arange(5*16)
    shards = [rank_indices(draw, 2, 8, rank, world).reshape(5, 8//world, 2) for rank in range(world)]
    reference = draw.reshape(5, 8, 2)
    for step in range(5):
        reconstructed = np.empty((8, 2), dtype=np.int64)
        for rank in range(world):
            for micro in range(8//world):
                global_micro = micro*world+rank
                reconstructed[global_micro] = shards[rank][step, micro]
        np.testing.assert_array_equal(reconstructed, reference[step])
    for rank in range(world):
        np.testing.assert_array_equal(rank_indices(draw, 2, 8, rank, world, 3), shards[rank][3:].reshape(-1))


def test_invalid_world_size_and_incomplete_draw_are_rejected():
    with pytest.raises(ValueError, match='divide'):
        local_accumulation(8, 3)
    with pytest.raises(ValueError, match='complete'):
        rank_indices(np.arange(17), 2, 8, 0, 2)
    with pytest.raises(ValueError, match='invalid'):
        rank_indices(np.arange(16), 2, 8, 2, 2)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.3))
        self.unused = nn.Parameter(torch.tensor(.5))
    def forward(self, x):
        return self.weight*x


def _gloo_worker(rank, store, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+store, rank=rank, world_size=2)
    try:
        model = DDP(Toy(), find_unused_parameters=True)
        x = torch.arange(1., 9.)
        mask = torch.tensor([1., 1., 1., 0., 1., 1., 0., 1.])
        draw = np.arange(8)
        groups = rank_indices(draw, 2, 4, rank, 2).reshape(2, 2)
        for micro, indices in enumerate(groups):
            with sync_context(model, micro, 2):
                loss = (model(x[indices]).square()*mask[indices]).sum()/mask[indices].sum()
                (loss/2).backward()
        reference = Toy()
        objective = sum((reference(x[i:i+2]).square()*mask[i:i+2]).sum()/mask[i:i+2].sum() for i in range(0, 8, 2))/4
        objective.backward()
        torch.testing.assert_close(model.module.weight.grad, reference.weight.grad, rtol=0, atol=2e-6)
        assert model.module.unused.grad is None
        # A single normalization over all valid labels changes the reference
        # objective when pair masks differ; preserving pairs matters.
        naive = Toy()
        (naive(x).square()*mask).sum().div(mask.sum()).backward()
        assert abs(float(naive.weight.grad-reference.weight.grad)) > 1
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        optimizer.step()
        torch.manual_seed(100+rank); np.random.seed(100+rank); random.seed(100+rank)
        rngs = gather_rng(2)
        expected = (torch.rand(4), np.random.rand(4), random.random())
        restore_rng(rngs[rank])
        torch.testing.assert_close(torch.rand(4), expected[0], rtol=0, atol=0)
        np.testing.assert_array_equal(np.random.rand(4), expected[1])
        assert random.random() == expected[2]
        assert any_rank(rank == 1, torch.device('cpu'))
        assert not any_rank(False, torch.device('cpu'))
        torch.save(dict(model=model.module.state_dict(), grad=reference.weight.grad,
                        optimizer=optimizer.state_dict(), rngs=rngs), Path(output)/('rank%d.pt'%rank))
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_matches_masked_reference_objective_and_rng(tmp_path):
    mp.spawn(_gloo_worker, args=(str(tmp_path/'store'), str(tmp_path)), nprocs=2, join=True)
    states = [torch.load(tmp_path/('rank%d.pt'%rank), weights_only=False) for rank in range(2)]
    for key,value in states[0]['model'].items():
        torch.testing.assert_close(value, states[1]['model'][key], rtol=0, atol=0)
    assert not torch.equal(states[0]['rngs'][0]['torch'], states[0]['rngs'][1]['torch'])


def test_checkpoint_resume_rejects_different_world_or_implementation():
    spec = importlib.util.spec_from_file_location('ddp_trainer_contract', TASK/'tools/run/train_oakink2_pointworld_ddp.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    config = {'seed': 217}
    identity = dict(dataset_hash='data', arm='action', stats_sha256='stats', vendor_sources={}, implementation_sources={})
    state = dict(config=config, dataset_hash='data', identity=identity,
                 checkpoint_kind='pointworld-temporal.ddp.v1', world_size=4)
    model = Toy(); optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    with pytest.raises(ValueError, match='world size'):
        trainer.load_checkpoint(state, model, optimizer, config, identity, 0, 2)
    state['world_size'] = 2
    state['identity'] = dict(identity, implementation_sources={'old.py': 'wrong'})
    with pytest.raises(ValueError, match='implementation mismatch'):
        trainer.load_checkpoint(state, model, optimizer, config, identity, 0, 2)
