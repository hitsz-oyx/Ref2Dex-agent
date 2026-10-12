"""Evaluation windows must not share observed/target frames with prior panels."""
import importlib.util
from pathlib import Path

import numpy as np


_path = Path(__file__).resolve().parents[1] / 'tools/audit/freeze_validation_panel.py'
_spec = importlib.util.spec_from_file_location('freeze_validation_panel', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def test_temporal_exclusion_includes_different_anchor_and_boundary_frames():
    previous = np.array([[3, 0, 100, 0]])
    candidates = np.array([[3, 1, 127, 1], [3, 1, 128, 1],
                           [3, 0, 73, 1], [3, 0, 72, 1], [4, 0, 100, 1]])
    # First/third windows share one frame despite differing ticks/anchors.
    assert _module.unseen_frames(candidates, previous).tolist() == [False, True, False, True, True]


def test_exclusion_checks_nearest_prior_window_on_both_sides():
    previous = np.array([[3, 0, 100, 0], [3, 1, 200, 1]])
    candidates = np.array([[3, 0, 150, 1], [3, 0, 175, 1], [3, 1, 120, 0]])
    assert _module.unseen_frames(candidates, previous).tolist() == [True, False, False]
