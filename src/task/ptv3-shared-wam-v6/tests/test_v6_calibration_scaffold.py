"""Reject mixed episodes, clock/mask mistakes and over-budget fitting before CUDA."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'tools/audit/v6_single_trajectory_calibration.py'
spec = importlib.util.spec_from_file_location('v6_calibration_scaffold', SCRIPT)
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


def dataset():
    x = np.zeros((24, 686), np.float32)
    x[:, 418] = 1
    x[3:-8, -1] = 1
    meta = dict(schema=calibration.DATA_SCHEMA, native_hz=30, tau_frame='world',
        tau_representation='displacement_from_current_hand', hand_order='native_inspire11',
        tau_kind='gt', tau_times_s=(np.arange(1, 9)/30).tolist(),
        source_sequence='single', task_semantics='declared-complete-task-v2', source_recording='source-pd/trajectory.npz')
    return x, np.zeros((24, 12), np.float32), np.arange(24), meta


def test_single_trajectory_contract():
    calibration.validate_data(*dataset())


@pytest.mark.parametrize('fault', ['mixed_episode', 'clock', 'tau_frame', 'startup', 'future', 'mask', 'commands', 'provenance'])
def test_reject_broken_recording(fault):
    x, y, tick, meta = dataset()
    if fault == 'mixed_episode': tick[12:] -= 12
    elif fault == 'clock': meta['native_hz'] = 60
    elif fault == 'tau_frame': meta['tau_frame'] = 'object_anchor'
    elif fault == 'startup': x[0, -1] = 1
    elif fault == 'future': x[-1, -1] = 1
    elif fault == 'mask': x[0, 421] = 1
    elif fault == 'commands': y[0, 0] = 1.1
    elif fault == 'provenance': del meta['source_recording']
    with pytest.raises(ValueError): calibration.validate_data(x, y, tick, meta)


@pytest.mark.parametrize('args', [['--updates', '2001'], ['--seconds', '1801'], ['--updates', '0']])
def test_budget_rejected_before_checkpoint_or_gpu(args):
    with pytest.raises(ValueError, match='hard limit'):
        calibration.main(['--dry-run', '--checkpoint', '/nonexistent', *args])


def test_execution_requires_real_data_and_output():
    with pytest.raises(ValueError, match='real fitting requires'):
        calibration.main(['--execute-training'])
