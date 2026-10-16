import numpy as np
import torch
from ptv3_shared_wam_v2.wrist_shared_probe import WristSharedMLP, pool_shared, condition_rows, phase_draw


def test_original_path_is_exact_and_shared_can_learn():
    torch.manual_seed(196)
    model=WristSharedMLP(shared_dim=12)
    x=torch.randn(4,811);z=torch.randn(4,12)
    # Nonzero output head exercises more than the trivial zero initial output.
    torch.nn.init.normal_(model.net[-1].weight,std=.02)
    torch.testing.assert_close(model(x,z),model.net(x),rtol=0,atol=0)
    model(x,z).square().mean().backward()
    assert model.shared_input.weight.grad.abs().sum()>0


def test_pool_excludes_invalid_payloads_and_preserves_groups():
    tokens=torch.ones(2,28,3);valid=torch.ones(2,28,dtype=torch.bool)
    tokens[:,:4]=2;tokens[:,4:26]=3;tokens[:,26:]=4
    valid[:,0]=False;tokens[:,0]=float('nan')
    expected=torch.tensor([[2.,2.,2.,3.,3.,3.,4.,4.,4.]]).repeat(2,1)
    torch.testing.assert_close(pool_shared(tokens,valid,4),expected)


def test_shuffle_and_draw_are_split_local_and_deterministic():
    z=np.arange(60).reshape(20,3)
    assert not condition_rows(z,'none',199).any()
    shuffled=condition_rows(z,'shuffled',199)
    assert not np.array_equal(z,shuffled)
    assert set(map(tuple,z))==set(map(tuple,shuffled))
    ticks=np.tile(np.arange(32),3);seq=np.repeat(np.arange(3),32)
    a=phase_draw(ticks,seq,512,np.random.default_rng(197))
    b=phase_draw(ticks,seq,512,np.random.default_rng(197))
    for _ in range(3):
        ix=a();assert np.array_equal(ix,b())
        assert (ticks[ix[:256]]<16).all() and (ticks[ix[256:]]>=16).all()
