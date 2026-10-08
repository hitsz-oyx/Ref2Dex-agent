"""Replay acceptance distinguishes metres from angular roundoff."""
import importlib.util
from pathlib import Path
import sys

import pytest

run = Path(__file__).parents[1]/'tools/run'
sys.path.insert(0, str(run))
spec = importlib.util.spec_from_file_location('hocap_comparison', run/'compare_hocap_checkpoints.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
sys.path.pop(0)


def test_replay_tolerates_small_rotation_roundoff_without_relaxing_point_error():
    point = 'model/anchor/cat0/h24/point_epe'
    angle = 'model/anchor/cat0/h12/rotation'
    original = {point: .017804544, angle: .105076998}
    replay = {point: .017804544, angle: .105076721}
    assert comparison.check_replay(original, replay)['point_epe'] == 0
    with pytest.raises(ValueError, match='replay mismatch'):
        comparison.check_replay(original, dict(replay, **{point: original[point]+2e-7}))
    with pytest.raises(ValueError, match='replay mismatch'):
        comparison.check_replay(original, dict(replay, **{angle: original[angle]+2e-6}))


def test_replay_rejects_missing_stratum_or_horizon():
    with pytest.raises(ValueError, match='coverage changed'):
        comparison.check_replay({'model/anchor/cat1/h24/point_epe': .022}, {})
