"""CPU contract tests for the HF04 trajectory-credit screen."""
from __future__ import annotations

import torch

from src.task.CmResidual.tools.fit_trajectory_credit_audit import (
    MODEL_NAMES,
    concatenate,
    features,
    fit_ridge,
    load_rows,
    metric,
)


def test_three_arm_sources_have_complete_short_trajectory_contract():
    rows = [load_rows(seed) for seed in (250, 251, 252, 253)]
    assert [int(row["valid_rows"]) for row in rows] == [63, 58, 63, 63]
    for row in rows:
        assert row["split"] in ("fit", "holdout")
        assert row["motion_id"].unique().tolist() == [0]
        assert torch.isfinite(row["trajectory"]).all()
        assert sorted(row["assignment"].unique().tolist()) == [-1, 0, 1]
        assert torch.allclose(
            row["treatment"][:, 2].abs(),
            row["assignment"].abs().float() * 0.1,
            atol=1e-6,
        )


def test_trajectory_shuffle_changes_only_training_trajectory_block():
    rows = concatenate([load_rows(250), load_rows(251)])
    permutation = torch.arange(len(rows["held_lift"]) - 1, -1, -1)
    for name in MODEL_NAMES:
        assert features(rows, name, training=True, permutation=permutation).shape[0] == 121
    normal = features(rows, "trajectory_credit", training=True)
    shuffled = features(rows, "trajectory_shuffled", training=True, permutation=permutation)
    assert normal.shape == shuffled.shape
    assert not torch.equal(normal, shuffled)
    assert torch.equal(
        features(rows, "trajectory_credit", training=False),
        features(rows, "trajectory_shuffled", training=False),
    )


def test_ridge_and_metrics_are_finite():
    x = torch.arange(36, dtype=torch.float32).reshape(12, 3)
    y = torch.linspace(0.0, 1.0, 12)
    model = fit_ridge(x, y)
    assert all(torch.isfinite(value).all() for value in model.values())
    report = metric(torch.linspace(-1.0, 1.0, 12),
                    torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
                    binary=True)
    assert all(value == value for value in report.values())
    assert 0.0 <= report["brier"] <= 1.0
