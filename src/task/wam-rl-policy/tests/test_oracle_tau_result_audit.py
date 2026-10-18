import importlib.util
import json
from pathlib import Path

import numpy as np


def test_archive_audit_recomputes_latched_success_and_marks_missing_terminal(tmp_path):
    tool = Path(__file__).resolve().parents[1] / 'tools/audit/audit_oracle_tau_results.py'
    spec = importlib.util.spec_from_file_location('oracle_result_audit', tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(
        seed=453, variant='p2', policy='fixture.pt', success_frames=3)))
    obs = np.zeros((4, 300), np.float32)
    obs[:, 204] = [.10, .01, .15, .20]
    action = np.zeros((4, 24), np.float32)
    np.savez(tmp_path / 'trajectory.npz', obs=obs, action=action,
             action_prefix=action, replay_error=np.zeros(4),
             object_pose=np.zeros((4, 13)))
    result = module.audit_run(tmp_path)
    assert result['upstream_reported_success_frames'] == 3
    assert result['last_record_hold']['success_frames'] == 1
    assert result['last_record_hold']['longest_hold_frames'] == 1
    assert not result['last_record_hold']['final_success']
    assert result['terminal_after_last_action'] is None
