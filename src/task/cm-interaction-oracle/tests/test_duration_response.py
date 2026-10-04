import importlib.util
from pathlib import Path

import numpy as np
import torch

PATH = Path(__file__).resolve().parents[1] / "tools/audit/probe_duration_response.py"
SPEC = importlib.util.spec_from_file_location("duration_response", PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_continuation_crosses_step16_and_keeps_early_height_failure():
    trajectory = torch.zeros(3, 32, 72)
    trajectory[:, :, 2] = .04
    trajectory[:, :, 71] = 1
    trajectory[0, :, 2] = .01
    trajectory[1, 12:18, 71] = 0  # sixth lost observation is step18.
    trajectory[2, :8, 71] = 0  # recovers before late region; still saved early.
    y, details = probe.response_targets(trajectory, torch.zeros(3))
    assert y[:, 2].tolist() == [1., 0., 0.]
    assert y[:, 3].tolist() == [1., 1., 0.]
    assert details["early_failure"].tolist() == [True, False, True]
    assert details["all32_failure"].tolist() == [True, True, True]
    assert y[2, 1] == 1


def test_gate_requires_growth_monotonicity_late_alignment_and_both_halves():
    beta = np.zeros((18, 19))
    beta[[0, 6, 12], 0] = [.02, .08, .15]
    beta[12, 1] = .09
    tails = np.ones(19)
    tails[:2] = .05
    assert probe.retention_candidates(beta, tails, [beta, beta])[0]["direction"] == "beneficial"
    assert probe.retention_candidates(-beta, tails, [-beta, -beta])[0]["direction"] == "harmful"
    weak_half = beta.copy()
    weak_half[12, 0] = .01
    assert not probe.retention_candidates(beta, tails, [beta, weak_half])
    nonmonotonic = beta.copy()
    nonmonotonic[6, 0] = .20
    assert not probe.retention_candidates(nonmonotonic, tails, [beta, beta])
    no_growth = beta.copy()
    no_growth[0, 0] = .14
    assert not probe.retention_candidates(no_growth, tails, [beta, beta])
    no_late = beta.copy()
    no_late[12, 1] = 0
    assert not probe.retention_candidates(no_late, tails, [beta, beta])
