import numpy as np
import pytest
from scripts.analyze_hold_plateau_substrate import score_phase


def trajectory():
    height = np.full(120, .04, dtype=np.float32)
    contact = np.ones((120, 2), dtype=np.float32)
    progress = np.arange(1, 121)
    return height, contact, progress


@pytest.mark.parametrize('held_steps,expected', [(74, False), (75, True)])
def test_followup_requires_75_steps_inside_actual_plateau(held_steps, expected):
    height, contact, progress = trajectory()
    # Continuous holds outside the predetermined phase cannot extend its run.
    contact[10 + held_steps:100] = 0
    result = score_phase(height, contact, progress, 0., 11, 100)
    assert result['max_phase_hold_steps'] == held_steps
    assert result['stable45']
    assert result['retained75'] == expected


def test_one_missing_force_proxy_breaks_continuous_hold():
    height, contact, progress = trajectory()
    contact[55, 1] = 0
    result = score_phase(height, contact, progress, 0., 11, 100)
    assert result['max_phase_hold_steps'] == 45
    assert result['stable45'] and not result['retained75']


def test_lift_is_measured_from_own_initial_native_height():
    height, contact, progress = trajectory()
    result = score_phase(height, contact, progress, .02, 11, 100)
    assert result['max_phase_hold_steps'] == 0
    assert not result['stable45'] and not result['retained75']


def test_progress_discontinuity_cannot_silently_shift_phase():
    height, contact, progress = trajectory()
    progress[55] += 1
    with pytest.raises(ValueError, match='discontinuous'):
        score_phase(height, contact, progress, 0., 11, 100)


def test_incomplete_native_phase_is_invalid():
    height, contact, progress = trajectory()
    with pytest.raises(ValueError, match='incomplete'):
        score_phase(height[:99], contact[:99], progress[:99], 0., 11, 100)
