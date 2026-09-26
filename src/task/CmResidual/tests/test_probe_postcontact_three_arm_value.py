import numpy as np
import pytest
import torch

from src.task.CmResidual.tools.probe_postcontact_three_arm_value import (
    ARMS,
    SCHEMA,
    _design,
    _load,
    _multi_arm_value,
)


def _payload(path):
    assignments = torch.tensor((-1, 0, 1) * 10, dtype=torch.int8)
    count = len(assignments)
    generator = torch.Generator().manual_seed(17)
    q = torch.randn(count, 18, generator=generator)
    dof_vel = torch.randn(count, 18, generator=generator)
    object_state = torch.randn(count, 13, generator=generator)
    base_action = torch.zeros(count, 18)
    executed_action = base_action.clone()
    executed_action[:, 2] += assignments.float() * 0.1
    records = {
        "q": q,
        "dof_vel": dof_vel,
        "object_state": object_state,
        "base_action": base_action,
        "executed_action": executed_action,
        "assignment": assignments,
        "intervention_valid": torch.ones(count, dtype=torch.bool),
        "pre_contact": torch.ones(count, dtype=torch.bool),
        "env_id": torch.arange(count),
        "progress": torch.full((count,), 50),
        "start_frame": torch.arange(count),
        "final_lift_success": torch.tensor([index % 2 for index in range(count)]),
        "final_max_contact_lift_m": torch.linspace(0.05, 0.3, count),
        "final_contact_fraction": torch.linspace(0.2, 0.6, count),
    }
    torch.save({
        "schema": SCHEMA,
        "run_status": "COMPLETED",
        "record_final_outcome": True,
        "three_arm_randomized": True,
        "delta_z_action": 0.1,
        "records": records,
    }, path)


def test_design_has_one_hot_and_state_interactions():
    state = np.arange(6, dtype=np.float32).reshape(2, 3)
    design = _design(state, np.array([-1, 1]), "cm_aware")
    assert design.shape == (2, 15)
    assert np.array_equal(design[:, :3], state)
    assert np.array_equal(design[0, 3:6], [1, 0, 0])
    assert np.array_equal(design[1, 3:6], [0, 0, 1])


def test_load_checks_three_arm_schema_and_action_delta(tmp_path):
    path = tmp_path / "transitions.pt"
    _payload(path)
    rows = _load(path, 7)
    assert rows["state"].shape == (30, 69)
    assert rows["assignment"].tolist()[:3] == list(ARMS)
    assert rows["counts"] == {-1: 10, 0: 10, 1: 10}

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["records"]["executed_action"][1, 2] += 0.01
    torch.save(payload, path)
    with pytest.raises(ValueError, match="executed action mismatch"):
        _load(path, 7)


def test_multi_arm_value_uses_source_specific_propensity():
    y = np.array([1., 0., 1., 0., 0., 1.])
    assignment = np.array([-1, 0, 1, -1, 0, 1])
    source = np.array([10, 10, 10, 11, 11, 11])
    propensities = [{-1: 0.5, 0: 0.25, 1: 0.25},
                    {-1: 0.25, 0: 0.5, 1: 0.25}]
    policy = np.ones(6, dtype=np.int64)
    value, matched = _multi_arm_value(
        y, assignment, policy, source, propensities)
    assert matched == 2
    assert value == pytest.approx(1.0)
