import numpy as np
import pytest
import torch

from consequence_evaluator.history_to_tau import (
    CURRENT_HAND_DIM,
    HISTORY_DIM,
    HORIZON,
    HistoryToTau,
)


def test_history_to_tau_shape_and_finite_output():
    torch.manual_seed(416)
    model = HistoryToTau(width=16).eval()
    history = torch.randn(3, HISTORY_DIM)
    current = torch.randn(3, 11, 3)
    with torch.inference_mode():
        prediction = model(history, current)
    assert prediction.shape == (3, HORIZON, CURRENT_HAND_DIM // 3, 3)
    assert torch.isfinite(prediction).all()


def test_history_to_tau_rejects_mismatched_contract():
    model = HistoryToTau(width=8)
    with pytest.raises(ValueError, match="shape mismatch"):
        model(np.zeros((2, HISTORY_DIM), dtype="float32"),
              np.zeros((2, 10, 3), dtype="float32"))
