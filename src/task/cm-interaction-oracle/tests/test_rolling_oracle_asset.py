"""Contracts for missing candidates and post-treatment versus actual observations."""
import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/audit/package_rolling_oracle_asset.py'
spec = importlib.util.spec_from_file_location('rolling_oracle_asset', SCRIPT)
asset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asset)


def test_certificate_leaves_unobserved_candidates_missing():
    y = torch.tensor([[1., 1., 0., 1., 1., 0., 0., 1.]])
    records = [dict(Y32=y)] + [None] * 6
    plan = dict(baseline_upper_bound_certificate=True, y=y[:, None].tolist(),
                utility=[[1.25]], choices=[0])
    assert asset.validate_plan(plan, records, torch.tensor([0])).tolist() == [0]
    assert records[1:] == [None] * 6
    # A lower score is not a valid reason to omit alternatives.
    records[0]['Y32'] = y * 0
    plan.update(y=(y * 0)[:, None].tolist(), utility=[[0.]])
    with pytest.raises(ValueError, match='invalid baseline certificate'):
        asset.validate_plan(plan, records, torch.tensor([0]))


def test_full_panel_cannot_drop_candidate_or_change_choice():
    y = torch.zeros(1, 8)
    records = [dict(Y32=y.clone()) for _ in range(7)]
    records[2]['Y32'][0, 7] = 1
    plan = dict(baseline_upper_bound_certificate=False,
                y=torch.stack([r['Y32'] for r in records], 1).tolist(),
                utility=[[0., 0., 1., 0., 0., 0., 0.]], choices=[2])
    assert asset.validate_plan(plan, records, torch.tensor([0])).tolist() == [2]
    plan['choices'] = [0]
    with pytest.raises(ValueError, match='selection replay'):
        asset.validate_plan(plan, records, torch.tensor([0]))
    records[5] = None
    with pytest.raises(ValueError, match='candidate missingness'):
        asset.validate_plan(plan, records, torch.tensor([0]))


def test_posttreatment_tip_flow_is_not_surface_flow_or_candidate_Z():
    before = torch.zeros(1, 72); before[:, 6] = 1; before[:, 2] = .04; before[:, 71] = 1
    after = before[:, None].repeat(1, 32, 1)
    tips = torch.ones(1, 32, 5, 3) * .01
    panel = dict(before=before, trajectory=after, valid_steps=torch.ones(1, 32, dtype=torch.bool),
                 full_world_prefix_errors=torch.zeros(1, 7), height=after[:, :, 2],
                 pair=torch.ones(1, 32, dtype=torch.bool), rest_height=torch.zeros(1),
                 fingertip_positions=tips, before_fingertip_positions=torch.zeros(1, 5, 3),
                 history=torch.zeros(1, 10, 139), actor_obs=torch.zeros(1, 4),
                 hand_root=torch.zeros(1, 13), before_hand_base_pose=torch.zeros(1, 7),
                 base_actions=torch.zeros(1, 8, 18), actions=torch.zeros(1, 8, 18),
                 pd_targets=torch.zeros(1, 8, 18), delta=torch.zeros(7, 18),
                 native_q=torch.zeros(1, 32, 18), hand_base_pose=torch.zeros(1, 32, 7),
                 initial_fingerprint='init', simulation_contract='test',
                 model_fingerprint='model', rms_fingerprint='rms',
                 clipped_steps=torch.zeros(1, dtype=torch.long), control_dt=1/30)
    r = asset.panel_record(panel, torch.tensor([0]), 0, 'speculative_same_current_state_fork32')
    assert r['F_actual_surface'] is None and r['Z_candidate'] is None
    assert torch.equal(r['F_actual_tip_displacement'], tips[:, :8])
    assert r['E8'].shape == (1, 12) and r['I8'].shape == (1, 14)
    panel['valid_steps'][0, 31] = False
    with pytest.raises(ValueError, match='incomplete'):
        asset.panel_record(panel, torch.tensor([0]), 0, 'fork')
