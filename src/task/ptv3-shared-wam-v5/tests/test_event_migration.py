import copy
import pytest
import torch
from test_v5_contract import Geometry, config, stats
from ptv3_shared_wam_v5.model import SharedWAMV5
from ptv3_shared_wam_v5.events import EVENT_ENCODING, initialize_event_model
from ptv3_shared_wam_v5.event_migration import expand_event_optimizer


def test_named_expansion_preserves_original_next_update_and_initializes_only_new_states():
    c = dict(config(), geometry_lr=1e-4, shared_lr=1e-4)
    base = SharedWAMV5(stats(), c, Geometry(8))
    target = SharedWAMV5(stats(), dict(c, event_conditioning=True, event_encoding=EVENT_ENCODING), Geometry(8))
    old = torch.optim.AdamW(base.parameter_groups(c['geometry_lr'], c['shared_lr']))
    for p in base.parameters(): p.grad = torch.ones_like(p)*.03
    old.step(); saved = copy.deepcopy(old.state_dict())
    initialize_event_model(target, base.state_dict())
    expanded, report = expand_event_optimizer(base, target, saved, c)
    new = torch.optim.AdamW(target.parameter_groups(c['geometry_lr'], c['shared_lr']))
    new.load_state_dict(expanded)
    assert report['preserved_states'] == len(saved['state'])
    assert all(p not in new.state for n, p in target.named_parameters() if n.startswith('event_goal.'))
    for m in (base, target):
        for p in m.parameters(): p.grad = torch.ones_like(p)*.07
    old.step(); new.step()
    for name, p in base.named_parameters():
        torch.testing.assert_close(p, dict(target.named_parameters())[name], rtol=0, atol=0)
    assert new.state[target.event_goal.encoder[0].weight]['step'] == 1
    assert next(iter(old.state.values()))['step'] == 2


def test_expansion_rejects_wrong_optimizer_group_identity():
    c = dict(config(), geometry_lr=1e-4, shared_lr=1e-4)
    base = SharedWAMV5(stats(), c, Geometry(8))
    target = SharedWAMV5(stats(), dict(c, event_conditioning=True, event_encoding=EVENT_ENCODING), Geometry(8))
    saved = torch.optim.AdamW(base.parameter_groups(1e-4, 1e-4)).state_dict()
    saved['param_groups'][0]['name'] = 'wrong'
    with pytest.raises(ValueError, match='identity/order'):
        expand_event_optimizer(base, target, saved, c)
