"""Repository-level counterparts of the user-provided 28361da audit cases."""
import math

import pytest
import torch
torch.set_num_threads(2)

from test_v4_contract import batch, config, model, stats
from ptv3_shared_wam_v4 import objectives, training
from ptv3_shared_wam_v4.normalization import HorizonNormalizer


def objective_fixture(n=8):
    base = batch()
    b = {k: v[:1].repeat(n, *([1] * (v.ndim-1))) for k, v in base.items()}
    b['object_valid'].fill_(True)
    b['trajectory_target'].zero_()
    b['scene_target'].zero_()
    b['trajectory_valid'][:n//2, :, :, 1:] = False
    b['scene_target'][n//2:] = .01
    b['interaction_target'].zero_()
    b['interaction_valid'].fill_(True)
    for i in range(n):
        b['interaction_target'][i, :, :, :i+1, :2] = 1
    b['interaction_valid'][0, :, :, 8:] = False
    # A whole rank can have no supervision for a component in a mixed batch.
    b['trajectory_valid'][:2] = False
    b['scene_valid'][:2] = False
    b['interaction_valid'][:2] = False
    torch.manual_seed(19)
    p = dict(trajectory_normalized=torch.randn_like(b['trajectory_target'], requires_grad=True),
             scene_normalized=torch.randn_like(b['scene_target'], requires_grad=True),
             interaction_logits=torch.randn_like(b['interaction_target'], requires_grad=True))
    return b, p


@pytest.mark.parametrize('world', [2, 4])
@pytest.mark.parametrize('interleave', [False, True])
def test_complete_ddp_loss_and_gradients_equal_concatenated_batch(monkeypatch, world, interleave):
    b, p = objective_fixture()
    c = config(); normalizer = HorizonNormalizer(stats())
    reference, _ = objectives.forecast_loss(p, b, c, normalizer)
    expected = torch.autograd.grad(reference, list(p.values()))
    # Counts are independently derived from the fixture's public label masks.
    hand_valid = b['trajectory_valid'] & b['start_hand_valid'][:, None]
    scene_valid = b['scene_valid'] & b['object_valid'][:, :, None, None]
    motion = b['scene_target'].norm(dim=-1)
    weight = c['motion_floor'] + (1-c['motion_floor']) * torch.sigmoid(5*(motion/c['motion_tau_m']-1))
    totals = torch.stack((hand_valid.float().sum(), torch.where(scene_valid, weight, 0).sum(),
        b['interaction_valid'].float().sum(),
        (b['interaction_valid'] & (b['interaction_target'][..., 0] > 0)).float().sum()))
    monkeypatch.setattr(objectives.dist, 'is_available', lambda: True)
    monkeypatch.setattr(objectives.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(objectives.dist, 'get_world_size', lambda: world)
    def reduce_counts(counts):
        counts.copy_(totals)
    monkeypatch.setattr(objectives.dist, 'all_reduce', reduce_counts)
    groups = [torch.arange(rank, 8, world) if interleave else
              torch.arange(rank*8//world, (rank+1)*8//world) for rank in range(world)]
    actual = [torch.zeros_like(v) for v in p.values()]
    losses = []
    for ids in groups:
        local = {k: v[ids].detach().requires_grad_() for k, v in p.items()}
        loss, _ = objectives.forecast_loss(local, {k: v[ids] for k, v in b.items()}, c,
                                           normalizer, distributed=True)
        grads = torch.autograd.grad(loss, list(local.values()))
        for dest, grad in zip(actual, grads):
            dest[ids] = grad / world
        losses.append(loss.detach()/world)
    torch.testing.assert_close(sum(losses), reference.detach())
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target)


def test_observation_only_forward_ignores_all_future_supervision():
    m = model(); b = batch()
    original = m(b)
    future = ('trajectory_target', 'trajectory_valid', 'scene_target', 'scene_valid',
              'interaction_target', 'interaction_valid')
    observation = {k: v for k, v in b.items() if k not in future}
    independent = m(observation)
    b['interaction_valid'] = ~b['interaction_valid']
    changed = m(b)
    for key in ('trajectory', 'scene', 'interaction', 'interaction_logits'):
        torch.testing.assert_close(original[key], independent[key], rtol=0, atol=0)
        torch.testing.assert_close(original[key], changed[key], rtol=0, atol=0)


class FixedPredictions(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.normalizer = HorizonNormalizer(stats())

    def forward(self, b):
        logits = self.logits[b['sample_id']]
        return dict(trajectory=torch.zeros_like(b['trajectory_target']),
            trajectory_normalized=torch.zeros_like(b['trajectory_target']),
            scene=torch.zeros_like(b['scene_target']), scene_normalized=torch.zeros_like(b['scene_target']),
            interaction_logits=logits, interaction=torch.cat((logits[..., :1].sigmoid(),
                                                               logits[..., 1:].softmax(-1)), -1))


def test_dev_objective_and_model_ranking_ignore_microbatch_and_order(monkeypatch):
    b, _ = objective_fixture(32)
    b['trajectory_target'].zero_(); b['scene_target'].zero_()
    b['scene_points'].zero_(); b['start_object_points'].zero_(); b['goal_object_delta'].zero_()
    b['goal_object_slot'] = torch.zeros(32, dtype=torch.long)
    b['interaction_target'].zero_(); b['interaction_valid'].fill_(True)
    b['interaction_target'][:16, :, :, :1, :2] = 1.
    b['interaction_target'][16:, :, :, :, :2] = 1.
    dataset = [{**{k: v[i] for k, v in b.items()}, 'sample_id': torch.tensor(i)} for i in range(32)]
    def collate(rows):
        return dict(model_inputs={k: torch.stack([r[k] for r in rows]) for k in rows[0]})
    monkeypatch.setattr(training, 'collate', collate)
    a = torch.zeros_like(b['interaction_target']); other = torch.zeros_like(a)
    a[:16, ..., 1] = -2.3905620875658995
    a[16:, ..., 1] = 3.1172090394179337
    other[..., 1] = 1.176685782734542
    orders = [list(range(32)), [x for i in range(16) for x in (i, i+16)]]
    reference = None
    for microbatch in (1, 8, 16):
        for order in orders:
            scores = [training.evaluate(FixedPredictions(logits), dataset, {'0': order},
                       dict(config(), amp=False), torch.device('cpu'), microbatch_size=microbatch)
                      for logits in (a, other)]
            values = [r['objective'] for r in scores]
            assert values[0] < values[1]
            if reference is None: reference = values
            assert values == pytest.approx(reference, rel=1e-7, abs=1e-7)
            assert scores[0]['sources']['0']['metrics']['interaction/part_ce']['unit'] == 'nats'
    assert reference == pytest.approx([.9227453, 1.6258998], rel=1e-6)


def test_fixed_dev_cpu_cache_preserves_objective_rng_and_avoids_reader(monkeypatch):
    b, p = objective_fixture()
    b['sample_id'] = torch.arange(8)
    b['goal_object_slot'] = torch.zeros(8, dtype=torch.long)
    b['start_object_points'] = b['scene_points'][:, 0].clone()
    rows = [{k:v[i] for k,v in b.items()} for i in range(8)]
    def fake_collate(samples):
        return dict(model_inputs={k:torch.stack([r[k] for r in samples]) for k in samples[0]})
    monkeypatch.setattr(training, 'collate', fake_collate)
    from ptv3_shared_wam_v4 import data
    monkeypatch.setattr(data, 'collate', fake_collate)
    panel={'0':list(range(8))}; c=dict(config(),amp=False,validation_microbatch_size=2)
    m=FixedPredictions(p['interaction_logits'].detach())
    reference=training.evaluate(m,rows,panel,c,torch.device('cpu'))
    cache=data.DevInputCache(rows,panel,2)
    class FailReader:
        def __getitem__(self, index):
            raise AssertionError('cached DEV re-entered reader')
    before=torch.get_rng_state().clone()
    cached=training.evaluate(m,FailReader(),panel,c,torch.device('cpu'),input_cache=cache)
    assert cached == reference
    assert torch.equal(before,torch.get_rng_state())
    assert cache.bytes > 0 and cache.build_seconds >= 0
    with pytest.raises(ValueError,match='panel/order/microbatch'):
        training.evaluate(m,rows,{'0':list(reversed(range(8)))},c,torch.device('cpu'),input_cache=cache)
    with pytest.raises(ValueError,match='budget'):
        data.DevInputCache(rows,panel,2,max_bytes=1)
