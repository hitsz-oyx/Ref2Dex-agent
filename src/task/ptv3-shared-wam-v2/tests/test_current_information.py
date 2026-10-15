from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/run'))
from current_information import (ARMS, EXTRA_DIM, REFERENCE_DIM, InformationRetargeter,
                                 current_information, pack)
from ptv3_shared_wam.target_retargeter import make_retargeter
from ptv3_shared_wam.action_targets import GEOMETRY_LIFTED


def test_current_features_known_reference_and_geometry():
    q = np.zeros((2, 18), np.float32)
    ref = q.copy(); ref[:, 0] = .1
    hand = np.zeros((2, 11, 3), np.float32); hand[:, :, 0] = .02
    obj = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    target = obj.copy(); target[:, 1, 3] = .3
    surface = np.zeros((512, 3), np.float32)
    extra = current_information(q, hand, obj, ref, target, surface)
    assert extra.shape == (2, EXTRA_DIM)
    np.testing.assert_allclose(extra[:, 0], .1)
    np.testing.assert_allclose(extra[:, 15:18], [[0, .3, 0]]*2)
    np.testing.assert_allclose(extra[:, REFERENCE_DIM:].reshape(2, 11, 3), hand)


def test_zero_adapter_preserves_same_parent_function_for_all_arms():
    torch.manual_seed(9)
    parent = make_retargeter(GEOMETRY_LIFTED).eval()
    torch.nn.init.normal_(parent.head.weight, std=.02)
    model = InformationRetargeter(parent.state_dict(), np.ones(EXTRA_DIM), np.ones(EXTRA_DIM)).eval()
    state, tau, extra = torch.randn(3, 87), torch.randn(3, 24, 33), torch.randn(3, EXTRA_DIM)
    expected = parent(state, tau)
    for arm in ARMS:
        torch.testing.assert_close(model(state, tau, extra, arm), expected, atol=1e-6, rtol=0)
    # Baseline masks AFTER normalization: it cannot depend on hidden extra values.
    torch.nn.init.normal_(model.extra[-1].weight, std=.02)
    torch.testing.assert_close(model(state, tau, extra, 'baseline'),
                               model(state, tau, extra+99, 'baseline'))


def test_packing_extra_uses_t_only_and_never_action_labels_or_future_reference():
    from types import SimpleNamespace
    n, tick = 40, 3
    data = dict(q=np.zeros((n, 18), np.float32), dq=np.zeros((n, 18), np.float32),
                hand=np.zeros((n, 11, 3), np.float32),
                obj=np.broadcast_to(np.eye(4), (n, 4, 4)).copy(), velocity=np.zeros((n, 6), np.float32),
                pd_targets=np.zeros((n-1, 18), np.float32),
                reference_obj=np.broadcast_to(np.eye(4), (n, 4, 4)).copy())
    refs = np.zeros((n, 18), np.float32)
    dataset = SimpleNamespace(records=[dict(sequence='test')], ticks_by_sequence=[np.array([tick])],
        pools=[np.array([0])], keys=[(tick, 0)], sequence=lambda _: (data, dict(points=np.zeros((512, 3)))),
        reference_sequence=lambda _: refs, __len__=lambda: 1)
    class Dataset:
        def __len__(self): return 1
    d = Dataset(); d.__dict__.update(dataset.__dict__)
    before = pack(d)['extra'].copy()
    data['q'][tick+1:] = 7
    data['hand'][tick+1:] = 9
    data['obj'][tick+1:, :3, 3] = 3
    data['reference_obj'][tick+1:, :3, 3] = 5
    data['pd_targets'][:] = .4
    refs[tick+1:] = 2
    np.testing.assert_array_equal(pack(d)['extra'], before)
    refs[tick, 0] = .2
    assert not np.array_equal(pack(d)['extra'], before)
