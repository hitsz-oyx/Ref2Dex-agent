import importlib.util
from pathlib import Path

import numpy as np


def load(name, relative):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_builder = load("history_candidate_panel_builder", "tools/audit/build_history_candidate_panel.py")
_fit = load("history_candidate_utility_fit", "tools/run/train_history_candidate_utility.py")


def test_farthest_candidate_selection_is_deterministic_and_label_free():
    tau = np.zeros((5, 2, 1, 1), dtype="float32")
    tau[:, 0, 0, 0] = np.arange(5.)
    first = _builder.farthest_candidates(0, [1, 2, 3, 4], tau)
    second = _builder.farthest_candidates(0, [1, 2, 3, 4], tau)
    assert first == second == [4, 1, 2, 3]


def test_fit_stats_has_train_only_feature_shape():
    value = np.arange(12., dtype="float32").reshape(3, 4)
    mean, scale = _fit.fit_stats(value)
    assert mean.shape == (4,) and scale.shape == (4,)
    np.testing.assert_allclose(mean, value.mean(0))
    assert np.all(scale > 0)
