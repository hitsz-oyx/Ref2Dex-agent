"""Expired/cumulative limits must not invalidate an authorized exact-config resume."""
import copy
import json
from pathlib import Path
import sys

import pytest

from ptv3_shared_wam_v6.stop_budget import (
    resolve_stop_budget, stop_budget_exhausted, validation_budget_view,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'ptv3-shared-wam-v5/src'))
from ptv3_shared_wam_v5.training_contract import execution_contract, validate_resume


def test_explicit_budget_preserves_strict_resume_identity_and_batch():
    config = dict(batch_size=96, updates=2500, max_seconds=10000, hard_stop_epoch=10)
    original = copy.deepcopy(config)
    budget = resolve_stop_budget(config, 3000, True, 7200, 20000)
    view = validation_budget_view(config, budget)
    assert view['updates'] == 3000 and view['max_seconds'] == 7200
    assert config == original
    saved = dict(config=original, data_identity='same', world_size=3,
                 execution_contract=execution_contract(original, 3))
    assert validate_resume(saved, config, 'same', 3)['kind'] == 'bound-effective-batch'
    assert not stop_budget_exhausted(budget, 1, 8241, 100)
    assert stop_budget_exhausted(budget, 7201, 8241, 100)
    assert stop_budget_exhausted(budget, 1, 8241, 20000)


def test_default_keeps_old_cumulative_and_absolute_stop_semantics():
    config = dict(max_seconds=10000, hard_stop_epoch=20)
    budget = resolve_stop_budget(config, 2500, True)
    assert validation_budget_view(config, budget) is config
    assert not stop_budget_exhausted(budget, 1, 8241, 19)
    assert stop_budget_exhausted(budget, 1760, 8241, 19)
    assert stop_budget_exhausted(budget, 0, 0, 20)


def test_legacy_without_absolute_deadline_remains_json_serializable():
    budget = resolve_stop_budget(dict(max_seconds=10000), 100, False)
    json.dumps(budget, allow_nan=False)
    assert not stop_budget_exhausted(budget, 1, 0, 100000)


@pytest.mark.parametrize('resume,seconds,epoch', [
    (False,7200,20000), (True,None,20000), (True,7200,None),
    (True,-1,20000), (True,float('inf'),20000), (True,7200,float('nan')),
])
def test_partial_or_invalid_override_is_rejected(resume, seconds, epoch):
    with pytest.raises(ValueError):
        resolve_stop_budget(dict(max_seconds=10000), 3000, resume, seconds, epoch)
