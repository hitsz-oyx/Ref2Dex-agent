"""Regression contracts for time identity and slow/rotational effect weighting."""
import json
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK/'src'))
from oakink_wm.pointworld_temporal import (
    TemporalPoint, TemporalGridPooling, TemporalPointWorldWM, model_from_config)
from oakink_wm.pointworld import PointWorldWM, capped_collate
from oakink_wm.data import Windows
from ptv3.ptv3 import GridPooling


def selector_model(mode='cumulative_effect'):
    # Selector/loss-only checks do not allocate a 50M-parameter backbone.
    m = TemporalPointWorldWM.__new__(TemporalPointWorldWM)
    nn.Module.__init__(m)
    m.motion_weighting, m.motion_tau_m, m.rotation_tau_rad = mode, .002, .02
    m.motion_temperature, m.motion_floor = 5., .1
    m.flow_mean = m.translation_mean = torch.zeros(24, 3)
    m.flow_std = m.translation_std = torch.ones(24, 3)*.02
    m.rotation_scale = torch.ones(24)
    return m


def labels():
    return dict(points=torch.zeros(1, 2, 8, 3),
                effect=torch.eye(4).repeat(1, 2, 24, 1, 1),
                object_valid=torch.tensor([[True, False]]))


def test_adapter_retains_24_action_times_in_one_spatial_voxel():
    stats = dict(flow_mean=torch.zeros(24, 3), flow_std=torch.ones(24, 3),
        translation_mean=torch.zeros(24, 3), translation_std=torch.ones(24, 3),
        rotation_scale=torch.ones(24), scene_mean=torch.zeros(18), scene_std=torch.ones(18),
        action_mean=torch.zeros(9), action_std=torch.ones(9))
    class Capture(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
        def forward(self, data):
            self.data = data
            return SimpleNamespace(feat=data['feat'])
    with patch('oakink_wm.pointworld.PointTransformerV3', Capture):
        model = TemporalPointWorldWM(stats).eval()
    capture = Capture()
    model.backbone = capture
    action = torch.zeros(1, 24, 22, 9)
    action[0, :, 0, 3] = torch.arange(24)*.001
    valid = torch.zeros(1, 24, 22, dtype=torch.bool)
    valid[:, :, 0] = True
    batch = dict(xyz=torch.zeros(1, 1, 3), features=torch.zeros(1, 1, 18),
        point_valid=torch.ones(1, 1, dtype=torch.bool), scene_object=torch.zeros(1, 1, dtype=torch.long),
        object_valid=torch.ones(1, 1, dtype=torch.bool), object_features=torch.zeros(1, 1, 15),
        action=action, action_valid=valid)
    with torch.no_grad():
        model(batch)
        original = capture.data
        model(dict(batch, action=action.flip(1)))
    assert len(original['feat']) == 25
    assert original['time_id'].tolist() == list(range(25))
    assert original['grid_coord'].count_nonzero() == 0
    assert not torch.allclose(original['feat'], capture.data['feat'])


def test_each_pool_level_preserves_time_and_physical_attention_batch():
    # Original spatial pooling collapses these 50 rows to 2; fixed pooling
    # preserves all25 times in each sample, even after four downsamplings.
    point = TemporalPoint(feat=torch.randn(50, 4), coord=torch.zeros(50, 3),
        grid_coord=torch.zeros(50, 3, dtype=torch.int32),
        batch=torch.arange(2).repeat_interleave(25), time_id=torch.arange(25).repeat(2))
    point.serialization(order=['z'], shuffle_orders=False)
    for _ in range(4):
        layer = TemporalGridPooling(GridPooling(4, 4, norm_layer=None, act_layer=None))
        child = layer(point, skip_postprocess=True)
        assert child.feat.shape[0] == 50
        assert child.offset.tolist() == [25, 50]
        assert torch.equal(child.time_id[child.pooling_inverse], point.time_id)
        assert torch.equal(child.batch[child.pooling_inverse], point.batch)
        child.serialization(order=['z'], shuffle_orders=False)
        point = child


def test_cumulative_slow_motion_rotation_and_invalid_padding():
    b = labels()
    b['effect'][0, 0, :, 0, 3] = torch.arange(1, 25)*(.04/24)
    m = selector_model()
    w = m.motion_weights(b)
    assert w[0, 0, -1].mean() > .99
    assert w[0, 0, 1].mean() > .95
    assert w[0, 1].count_nonzero() == 0
    old = selector_model('released_incremental').motion_weights(b)
    assert old[0, 0].mean() < .04
    b = labels()
    angle = torch.tensor(.04)
    c, s = angle.cos(), angle.sin()
    b['effect'][0, 0, :, :3, :3] = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    # Zero-radius points have no surface displacement: rotation independently
    # protects the supervision on small objects.
    assert m.motion_weights(b)[0, 0].mean() > .99
    b = labels()
    assert m.motion_weights(b)[0, 0].min() >= .1


def test_legacy_selector_matches_actual_v1_loss_and_padding_is_inert():
    b = labels()
    b['effect'][0, 0, :, 0, 3] = torch.arange(1, 25)*.001
    m = selector_model('released_incremental')
    pred = dict(rotation=torch.eye(3).repeat(1, 2, 24, 1, 1),
                translation=torch.full((1, 2, 24, 3), .003))
    loss, terms = m.loss(pred, b)
    old_loss, old_terms = PointWorldWM.loss(m, pred, b)
    torch.testing.assert_close(loss, old_loss, rtol=0, atol=0)
    for key in terms:
        torch.testing.assert_close(terms[key], old_terms[key], rtol=0, atol=0)
    b['effect'][0, 1, :, :3, 3] = 1000
    torch.testing.assert_close(m.loss(pred, b)[0], loss, rtol=0, atol=0)
    m.motion_weighting = 'cumulative_effect'
    perfect = dict(rotation=b['effect'][..., :3, :3], translation=b['effect'][..., :3, 3])
    assert m.loss(perfect, b)[0] == 0


def test_temporal_entry_rejects_legacy_config():
    with pytest.raises(ValueError, match='temporal configuration'):
        model_from_config({}, {'schema': 'pointworld-small-wm24.v1'})


def test_training_identity_accepts_relative_script_invocation(monkeypatch):
    path = TASK/'tools/run/train_oakink2_pointworld_temporal.py'
    spec = importlib.util.spec_from_file_location('temporal_trainer_contract', path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    expected = trainer.implementation_sources()
    monkeypatch.setattr(trainer, '__file__', os.path.relpath(path))
    assert trainer.implementation_sources() == expected
    assert len(expected) == 6


def test_real_cuda_time_identity_masks_gradients_and_checkpoint():
    data, stats = os.environ.get('POINTWORLD_DATA'), os.environ.get('POINTWORLD_STATS')
    if not data or not stats or not torch.cuda.is_available():
        pytest.skip('Set POINTWORLD_DATA/STATS for the real temporal CUDA contract')
    torch.set_num_threads(2)
    dataset = Windows(data, 'train')
    samples = [dataset[int(dataset.groups[0][i])] for i in (0, 20)]
    b = {k: v.cuda() for k, v in capped_collate(samples).items()}
    torch.manual_seed(9)
    model = TemporalPointWorldWM(json.loads(Path(stats).read_text())).cuda().eval()
    seen = []
    def verify_layer(module, inputs, output):
        p = output
        assert isinstance(p, TemporalPoint)
        if isinstance(module, TemporalGridPooling):
            assert torch.equal(p.time_id[p.pooling_inverse], inputs[0].time_id)
        keys = p.sparse_conv_feat.indices
        assert len(torch.unique(keys, dim=0)) == len(keys)
        for bid in range(2):
            assert set(p.time_id[p.batch == bid].tolist()) == set(range(25))
        assert len(p.offset) == 2  # cross-time attention stays in one real sample
        seen.append(len(keys))
    handles = [model.backbone.register_forward_hook(verify_layer)]
    handles += [p.register_forward_hook(verify_layer) for p in model.modules() if isinstance(p, TemporalGridPooling)]
    class NoLabels(dict):
        def __getitem__(self, key):
            if key == 'effect':
                raise AssertionError('future effect leaked into model input')
            return super().__getitem__(key)
    with torch.no_grad():
        first = model(NoLabels(b))
        repeat = model(NoLabels(b))
        torch.testing.assert_close(first['translation'], repeat['translation'], rtol=0, atol=1e-6)
    assert len(seen) == 10
    for h in handles:
        h.remove()
    with torch.no_grad():
        masked = dict(b, action_valid=b['action_valid'].clone())
        masked['action_valid'][:, :, 11:] = False
        base = model(masked)
        altered = dict(masked, action=b['action'].clone())
        altered['action'][:, :, 11:] += 1000
        torch.testing.assert_close(base['translation'], model(altered)['translation'], rtol=0, atol=1e-6)
        altered = dict(b, action=b['action'].flip(1))
        assert not torch.allclose(first['translation'], model(altered)['translation'], rtol=0, atol=1e-8)
        history = model(b, 'history')
        torch.testing.assert_close(history['translation'], model(dict(b, action=b['action']+1000), 'history')['translation'], rtol=0, atol=1e-6)
        R = first['rotation']
        torch.testing.assert_close(R.transpose(-1, -2)@R, torch.eye(3, device='cuda').expand_as(R), rtol=0, atol=2e-5)
    model.train()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred = model(b)
    loss, _ = model.loss(pred, b)
    loss.backward()
    for module in (model.action_proj[0], model.time):
        assert torch.isfinite(module.weight.grad).all() and module.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        expected = model(b)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        model.head[-1].weight.add_(1)
        model.load_state_dict(state)
        torch.testing.assert_close(expected['translation'], model(b)['translation'], rtol=0, atol=1e-6)
