from pathlib import Path

import torch

from src.task.CmResidual.temporal_cm import (
    HistoryValueCm,
    STEP_DIM,
    TemporalHistoryBuffer,
    FrozenTemporalHistoryCm,
    temporal_step_features,
)


def _checkpoint(path: Path):
    model = HistoryValueCm(history_len=3, width=16)
    torch.save({
        "schema": "ref2dex.history_value_cm.v1",
        "variant": "history_action",
        "model": model.state_dict(),
        "feature_mean": torch.zeros(STEP_DIM),
        "feature_std": torch.ones(STEP_DIM),
        "target_mean": torch.zeros(4),
        "target_std": torch.ones(4),
    }, path)


def test_temporal_step_features_and_first_padding():
    q = torch.zeros(2, 18)
    velocity = torch.ones(2, 18)
    object_state = torch.zeros(2, 13)
    action = torch.full((2, 18), 2.0)
    step = temporal_step_features(q, velocity, object_state, action)
    assert step.shape == (2, STEP_DIM)
    buffer = TemporalHistoryBuffer(2, 3, "cpu")
    history, mask = buffer.append(step)
    assert torch.allclose(history[:, -1], step)
    assert torch.equal(mask, torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]))
    second, second_mask = buffer.append(step + 1)
    assert torch.allclose(second[:, -2], step)
    assert torch.equal(second_mask, torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]))
    buffer.reset([1])
    assert not bool(buffer.initialized[1])
    assert bool(buffer.initialized[0])


def test_frozen_temporal_checkpoint_predicts(tmp_path):
    path = tmp_path / "history_action.pt"
    _checkpoint(path)
    cm = FrozenTemporalHistoryCm(path, "cpu")
    history = torch.zeros(4, 3, STEP_DIM)
    mask = torch.ones(4, 3)
    output = cm.predict(history, mask)
    assert output["outcomes"].shape == (4, 4)
    assert output["supported_lift_mm"].shape == (4,)
    assert torch.isfinite(output["final_contact_probability"]).all()
