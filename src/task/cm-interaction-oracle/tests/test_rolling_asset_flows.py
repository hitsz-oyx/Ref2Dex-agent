import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/audit/reconstruct_rolling_asset_flows.py'
spec = importlib.util.spec_from_file_location('rolling_asset_flows', SCRIPT)
asset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asset)


def test_pruned_candidates_keep_missing_slots_and_join_identity():
    rd = dict(candidates=[dict(candidate=0)]+[None]*6,
              candidate_observed=torch.tensor([True]+[False]*6), actual={})
    assert asset.validate_slots(rd).tolist() == [True]+[False]*6
    assert rd['candidates'][1:] == [None]*6
    rd['candidate_observed'][2] = True
    with pytest.raises(ValueError, match='observed-mask/join'):
        asset.validate_slots(rd)
    rd['candidate_observed'][2] = False
    rd['candidates'][0]['candidate'] = 1
    with pytest.raises(ValueError, match='candidate join'):
        asset.validate_slots(rd)


def test_current_q0_drift_is_rejected_before_FK():
    record = dict(H=dict(native_q0=torch.ones(1,18), history=torch.zeros(1,10,139)))
    with pytest.raises(ValueError, match='q0/history'):
        asset.to_packet(record)
