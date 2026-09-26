"""CPU contract tests for the HF03 contact-supported credit audit."""
from __future__ import annotations

import torch

from src.task.CmResidual.tools.fit_contact_supported_credit_audit import (
    MODEL_NAMES,
    concatenate,
    feature_blocks,
    fit_ridge,
    load_rows,
    metrics,
)


def test_existing_four_runs_have_pinned_first_episode_and_balanced_arms():
    rows = [load_rows(seed) for seed in (246, 247, 248, 249)]
    assert [int(row["valid_rows"]) for row in rows] == [63, 60, 60, 62]
    for row in rows:
        assert row["split"] in ("fit", "holdout")
        assert torch.equal(row["motion_id"].unique(), torch.tensor([0]))
        assert torch.isfinite(row["state"]).all()
        assert torch.isfinite(row["post_handflow"]).all()
        assert sorted(row["assignment"].unique().tolist()) == [-1, 1]
        assert abs(float(row["treatment_delta"][:, 2].abs().mean()) - 0.1) < 1e-6


def test_feature_contract_has_post_handflow_and_fixed_placebo():
    rows = concatenate([load_rows(246), load_rows(247)])
    permutation = torch.arange(len(rows["held_lift"]) - 1, -1, -1)
    dimensions = {
        name: tuple(feature_blocks(
            rows, name, training=True, permutation=permutation).shape)
        for name in MODEL_NAMES
    }
    assert dimensions["state_only"][0] == 123
    assert dimensions["action_aware"] == dimensions["action_shuffled"]
    assert dimensions["post_handflow"] == dimensions["post_handflow_shuffled"]
    assert not torch.equal(
        feature_blocks(rows, "post_handflow", training=True),
        feature_blocks(rows, "post_handflow_shuffled", training=True,
                       permutation=permutation),
    )
    assert torch.equal(
        feature_blocks(rows, "post_handflow", training=False),
        feature_blocks(rows, "post_handflow_shuffled", training=False),
    )


def test_ridge_and_binary_metrics_are_finite():
    x = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    y = torch.linspace(0.0, 1.0, 10)
    fitted = fit_ridge(x, y)
    assert all(torch.isfinite(value).all() for value in fitted.values())
    report = metrics(torch.linspace(-1.0, 1.0, 10),
                     torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
                     binary=True)
    assert all(value == value for value in report.values())
    assert 0.0 <= report["brier"] <= 1.0
    assert 0.0 <= report["auroc"] <= 1.0
