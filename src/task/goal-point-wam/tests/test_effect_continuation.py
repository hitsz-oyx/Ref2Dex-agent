import copy
import importlib.util
from pathlib import Path


def test_continuation_retains_control_gate_and_exposes_strict_forecast_failure():
    path = Path(__file__).resolve().parents[1] / 'tools/run/train_effect_consistency.py'
    spec = importlib.util.spec_from_file_location('effect_trainer_guard_test', path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    baseline = {name: dict(h1_mm=7., first8_mm=10., PD_xyz_p95_mm=20.,
                          PD_rotation_p95_rad=.1, PD_finger_p95_rad=.02,
                          clipping_rate=0.) for name in trainer.NAMES}
    current = copy.deepcopy(baseline)
    current['startup']['first8_mm'] = 11.4
    assert not trainer.nonregression(current, baseline)['startup/first8_mm']
    assert all(trainer.continuation_checks(current, baseline).values())
    current['startup']['PD_xyz_p95_mm'] = 22.2
    assert not trainer.continuation_checks(current, baseline)['startup/PD_xyz_p95_mm']
    current['startup']['h1_mm'] = 14.1
    assert not trainer.continuation_checks(current, baseline)['startup/h1_mm']
