import importlib.util
from pathlib import Path

import pytest

TASK = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('certify_reset_goal_channel', TASK / 'tools/audit/certify_reset_goal_channel.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_missing_evidence_stays_unknown_and_order_cannot_change():
    admitted = [dict(source_id=1, sequence_id=2, slot=0, tick=t) for t in (3, 8)]
    rows = [dict(r, missing='explicit_instruction_source_role;target_execution_and_reset_binding',
                 source_role='UNKNOWN', available_before_reset='UNPROVEN',
                 waypoint_status='UNCERTAIN_MISSING_RECORDS') for r in admitted]
    result = audit.coverage(rows, admitted)
    assert result['certified_legal'] == 0 and result['uncertain'] == 2
    assert result['missing_window_counts']['target_execution_and_reset_binding'] == 2
    assert result['available_before_reset_proven_hits'] == 0
    with pytest.raises(ValueError, match='exactly'):
        audit.coverage(rows[::-1], admitted)
    rows[0]['waypoint_status'] = 'CERTIFIED_LEGAL'
    with pytest.raises(ValueError, match='new evidence'):
        audit.coverage(rows, admitted)


def test_new_synthetic_artifacts_never_certify_old_windows(tmp_path):
    result = audit.synthetic_collector(tmp_path)
    assert result['historical_windows_certified'] == 0
    assert result['persisted_instruction_and_receipts_replayed_bit_exact']
    assert result['condition_quantity'] == 'task_level_goal'
    assert result['target_stage'] == 'task_completion'
    assert result['task_goal_distinct_from_planner_waypoint']
    assert result['exact_retries_no_duplicate_writes']
    assert result['decisions'] == 4
