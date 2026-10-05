"""Semantic boundaries for the diagnostic, independent of simulator execution."""
import importlib.util
from pathlib import Path
import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'tools/audit/audit_y_temporal_channels.py'
SPEC = importlib.util.spec_from_file_location('temporal_channels', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pre_event_joint_risk_and_missing_event_censoring():
    # A larger earlier signal outside risk and a signal at failure are ineligible.
    out = MODULE.first_signal([1, .1, 1], [0, 8, 16], [False, True, True], 16)
    assert out['correct'] == {'offset': 8, 'lead_steps': 8}
    assert out['eligible_queries'] == 1
    assert MODULE.first_signal([1], [0], [True], None)['correct'] is None


def test_inverse_does_not_count_as_correct_and_epsilon_is_strict():
    out = MODULE.first_signal([-.1, .02, .03], [0, 8, 16], [True]*3, 32)
    assert out['inverse']['offset'] == 0
    assert out['absolute']['offset'] == 0
    assert out['correct']['offset'] == 16


def test_height_margin_uses_local_steps_9_to_32_and_velocity_units():
    h = np.full((1, 1, 40), .05)
    h[..., :8] = .01
    h[..., 31] = .04
    out = MODULE.continuous(h, np.ones_like(h, bool), np.zeros((1, 1)),
                            np.full((1, 1), .05), [0])
    np.testing.assert_allclose(out[0, 0, 0], [.02, -.01, -.01, 1])


def test_unknown_event_distribution_not_zero_filled():
    assert MODULE.distribution([])['median'] is None
    assert MODULE.distribution([29, 34, 47])['median'] == 34
