import importlib.util
from pathlib import Path

import numpy as np


_path = Path(__file__).resolve().parents[1] / "tools/audit/audit_finger_command_coverage.py"
_spec = importlib.util.spec_from_file_location("finger_command_coverage", _path)
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)


def test_near_pair_stats_detects_same_state_finger_command_difference():
    hand = np.zeros((542, 3, 11, 3), dtype="float32")
    q = np.zeros((542, 3, 18), dtype="float32")
    dq = np.zeros_like(q)
    action = np.zeros_like(q)
    action[:, 1, 6:18] = .1
    pair = np.zeros((542, 3), dtype=bool)
    mode = np.zeros((542, 3), dtype=np.int8)
    result = _audit.near_pair_stats(hand, q, dq, action, pair, mode,
                                    np.r_[np.zeros(120, np.int8),
                                          np.ones(120, np.int8),
                                          np.full(302, 2, np.int8)])
    assert result["approach"]["near"] == 3 * 120
    assert result["approach"]["nontrivial"] == 2 * 120
    assert result["contact"]["same_mode_nontrivial"] == 2 * 120


def test_near_pair_stats_rejects_nonmatching_state():
    hand = np.zeros((542, 2, 11, 3), dtype="float32")
    q = np.zeros((542, 2, 18), dtype="float32")
    q[:, 1, 0] = .2
    dq = np.zeros_like(q)
    action = np.zeros_like(q)
    pair = np.zeros((542, 2), dtype=bool)
    mode = np.zeros((542, 2), dtype=np.int8)
    phase = np.r_[np.zeros(120, np.int8), np.ones(120, np.int8),
                  np.full(302, 2, np.int8)]
    result = _audit.near_pair_stats(hand, q, dq, action, pair, mode, phase)
    assert all(row["near"] == 0 for row in result.values())
