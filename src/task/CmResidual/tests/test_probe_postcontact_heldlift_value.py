import numpy as np
import pytest
import torch

from src.task.CmResidual.tools.probe_postcontact_heldlift_value import (
    SCHEMA, _design, _hajek, _load_rows,
)


def _payload(path, *, assignments=None):
    if assignments is None:
        assignments = tuple((-1, 1, 0) * 10)
    count = len(assignments)
    g = torch.Generator().manual_seed(7)
    q = torch.randn(count, 18, generator=g)
    dof_vel = torch.randn(count, 18, generator=g)
    object_state = torch.randn(count, 13, generator=g)
    base = torch.zeros(count, 18)
    executed = base.clone()
    executed[:, 2] += torch.tensor(assignments, dtype=torch.float32) * .1
    records = {
        "q": q, "dof_vel": dof_vel, "object_state": object_state,
        "base_action": base, "executed_action": executed,
        "assignment": torch.tensor(assignments, dtype=torch.int8),
        "pre_contact": torch.ones(count, dtype=torch.bool),
        "env_id": torch.arange(count), "progress": torch.full((count,), 50),
        "start_frame": torch.arange(count),
        "final_lift_success": torch.tensor(
            [bool(index % 2) for index in range(count)]),
        "final_max_contact_lift_m": torch.linspace(.05, .3, count),
        "final_contact_fraction": torch.linspace(.2, .6, count),
        "final_episode_steps": torch.full((count,), 500),
    }
    payload = {
        "schema": SCHEMA, "run_status": "COMPLETED",
        "record_final_outcome": True, "delta_z_action": .1,
        "records": records,
    }
    torch.save(payload, path)


def test_load_rows_filters_unassigned_and_builds_preaction_state(tmp_path):
    path = tmp_path / "transitions.pt"
    _payload(path)
    rows = _load_rows(path, 3)
    assert rows["state"].shape == (20, 69)
    assert rows["assignment"].tolist()[:3] == [-1.0, 1.0, -1.0]
    assert rows["source_id"] == 3
    assert rows["propensity_plus"] == pytest.approx(.5)


def test_design_interaction_changes_with_candidate_action():
    state = np.arange(6, dtype=np.float32).reshape(2, 3)
    plus = _design(state, np.ones(2), "cm_aware")
    minus = _design(state, -np.ones(2), "cm_aware")
    assert plus.shape == (2, 7)
    assert np.array_equal(plus[:, :3], minus[:, :3])
    assert np.array_equal(plus[:, 3], np.ones(2))
    assert np.array_equal(plus[:, 4:], -minus[:, 4:])


def test_hajek_uses_observed_propensity():
    y = np.array([1., 0., 1., 0.])
    assignment = np.array([1., -1., 1., -1.])
    policy = np.ones(4)
    value, matched = _hajek(y, assignment, policy, np.full(4, .5))
    assert matched == 2
    assert value == pytest.approx(1.0)


def test_load_rows_rejects_missing_final_label(tmp_path):
    path = tmp_path / "transitions.pt"
    _payload(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["records"]["final_lift_success"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="missing record fields"):
        _load_rows(path, 0)
