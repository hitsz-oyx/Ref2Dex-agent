"""Recovery budget regressions; no simulation or real output mutation."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

RUN = Path(__file__).resolve().parents[1] / 'tools/run'
sys.path.insert(0, str(RUN))
spec = importlib.util.spec_from_file_location('rolling_recovery_test_module', RUN/'resume_rolling_gt_y.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def put(path, value, timestamp):
    path.write_text(json.dumps(value))
    os.utime(path, (timestamp, timestamp))


def fixture_run(root):
    saved = dict(status='ROUND_EXECUTED', pid=999999999, inputs={},
                 physical_gpus=[6, 7], git_commit='frozen-science',
                 elapsed_seconds=10., events=[])
    put(root/'progress.json', saved, 1000.)
    put(root/'original-worker.log', {}, 1002.)
    # The shell creates this before the recovery constructor. Its timestamp
    # must not charge the 98 seconds of idle time as scientific execution.
    put(root/'resume-coordinator.log', {}, 1100.)
    return SimpleNamespace(run_dir=root, gpus=[6, 7], workers=4,
                           wall_seconds=7200, storage_gib=4)


def test_opening_recovery_log_does_not_charge_downtime(tmp_path):
    campaign = module.Resume(fixture_run(tmp_path))
    assert campaign.charged_seconds == 12.


def previous_recovery(root, error):
    put(root/'resume_manifest.json', {'charged_seconds': 110.}, 1101.)
    put(root/'resume_progress.json', dict(status='FAILED', pid=999999999,
        elapsed_seconds=160., events=[{'stage':'FAILED', 'error':error}]), 1150.)


def test_correction_preserves_all_prior_active_work(tmp_path):
    args = fixture_run(tmp_path)
    previous_recovery(tmp_path, 'KeyboardInterrupt()')
    campaign = module.Resume(args)
    assert campaign.charged_seconds == 62.  # 12 original + 50 recovery
    ledger = json.loads((tmp_path/'resume_manifest-002.json').read_text())
    assert ledger['accounting']['excluded_downtime_seconds'] == 98.
    assert (tmp_path/'resume_progress-001.json').exists()


def test_recovery_cannot_bypass_failed_scientific_checks(tmp_path):
    args = fixture_run(tmp_path)
    previous_recovery(tmp_path, "ValueError('input drift')")
    with pytest.raises(ValueError, match='cannot be bypassed'):
        module.Resume(args)
    assert not (tmp_path/'resume_manifest-002.json').exists()
