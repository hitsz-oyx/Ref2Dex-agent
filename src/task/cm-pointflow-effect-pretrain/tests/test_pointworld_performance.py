"""Integer equality for the actual PTv3 spatial orders used by the model."""
import itertools
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / 'src'))
from oakink_wm.pointworld_temporal import VENDOR
from oakink_wm.pointworld_performance import install_fused_hilbert, restore_hilbert
from ptv3.serialization.default import encode


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for fused kernel parity')
def test_all_spatial_orders_integer_exact_with_boundaries_and_duplicate_points():
    generator = torch.Generator().manual_seed(217)
    for depth in range(1, 17):
        points = (torch.tensor(list(itertools.product(range(1 << depth), repeat=3)))
                  if depth <= 3 else torch.randint(1 << depth, (513, 3), generator=generator))
        boundaries = torch.tensor([[0, 0, 0], [(1 << depth)-1]*3, [0, (1 << depth)-1, 0]])
        points = torch.cat((points, boundaries, points[:5])).cuda()
        batch = (torch.arange(len(points), device='cuda') % 4).long()
        for order in ('z', 'z-trans', 'hilbert', 'hilbert-trans'):
            expected = encode(points, batch, depth=depth, order=order)
            original = install_fused_hilbert()
            try:
                actual = encode(points, batch, depth=depth, order=order)
            finally:
                restore_hilbert(original)
            assert torch.equal(actual, expected), (depth, order)
            assert torch.equal(actual.argsort(), expected.argsort()), (depth, order)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_singleton_noncontiguous_int32_and_large_grid():
    from ptv3.serialization.hilbert import encode as reference
    from oakink_wm.pointworld_performance import fused_hilbert_encode
    generator = torch.Generator().manual_seed(218)
    for count in (1, 4096):
        points = torch.randint(65536, (count, 6), generator=generator, dtype=torch.int32).cuda()[:, ::2]
        assert torch.equal(fused_hilbert_encode(points), reference(points, 3, 16))


def test_reference_checkpoint_import_extends_only_matching_implementation_identity():
    path = TASK / 'tools/run/train_oakink2_pointworld_fast.py'
    spec = importlib.util.spec_from_file_location('fast_entry', path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    reference = {'base.py': 'frozen'}
    fast = dict(reference, **{'fast.py': 'fused'})
    state = {'identity': {'implementation_sources': reference, 'arm': 'action'},
             'step': 37, 'model': {'weight': torch.ones(2)}, 'optimizer': {'step': 37}}
    imported = entry.extend_reference_checkpoint(state, reference, fast)
    assert state['identity']['implementation_sources'] == reference
    assert imported['identity']['implementation_sources'] == fast
    assert imported['model'] is state['model'] and imported['optimizer'] is state['optimizer']
    assert imported['step'] == 37 and imported['identity']['arm'] == 'action'
    with pytest.raises(ValueError, match='source mismatch'):
        entry.extend_reference_checkpoint(state, {'base.py': 'changed'}, fast)
    with pytest.raises(ValueError, match='native resume'):
        entry.extend_reference_checkpoint(imported, reference, fast)
