"""Authentication is stricter than file age, names or a TaskSpec default."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

TASK=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('source_certification',TASK/'tools/audit/certify_waypoint_sources.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def legal_proof():
    return dict(target_tuple_bound=True,source_role='explicit_task_instruction',
        available_before_reset=True,independent_of_target_rollout=True,
        artifact_digest_verified=True,reset_binding_verified=True,derivation_verified=True,
        goal_semantics='spatial_path_waypoint',source_execution_id='old-source-A',target_execution_id='new-run-B')


def test_independent_old_source_may_legally_include_its_own_future():
    proof=legal_proof()
    proof['source_contains_future_of_source_recording']=True
    assert audit.classify(proof)==audit.LEGAL


@pytest.mark.parametrize('field',['target_tuple_bound','source_role','available_before_reset',
    'independent_of_target_rollout','artifact_digest_verified','reset_binding_verified',
    'derivation_verified','goal_semantics','source_execution_id','target_execution_id'])
def test_each_missing_fact_is_unknown_not_illegal(field):
    proof=legal_proof();proof.pop(field)
    assert audit.classify(proof)==audit.UNKNOWN


def test_mtime_and_task_spec_default_never_upgrade_unknown():
    assert audit.classify(dict(mtime_ns=1,reset_time_ns=100,artifact_exists=True))==audit.UNKNOWN
    assert not audit.hits(dict(schema='ref2dex.object-task.v1',object_path=[[0,0,0]]))
    assert audit.classify(dict(source_role='explicit_task_instruction'))==audit.UNKNOWN


def test_proven_same_target_future_requires_target_binding_and_derivation():
    proof=dict(target_tuple_bound=True,same_target_execution=True,source_reads_target_future=True,
        derivation_verified=True,source_execution_id='A',target_execution_id='A')
    assert audit.classify(proof)==audit.LEAK
    proof['target_tuple_bound']=False
    assert audit.classify(proof)==audit.UNKNOWN


def test_same_name_other_execution_is_not_same_target_proof():
    proof=legal_proof();proof.update(same_target_execution=True,source_reads_target_future=True)
    assert audit.classify(proof)==audit.LEGAL
    proof['source_execution_id']=proof['target_execution_id']
    assert audit.classify(proof)==audit.LEAK


def test_raw_declaration_is_discovered_without_defaults_or_guessing():
    found=audit.hits(dict(records=[dict(task_spec=dict(source_role='explicit_task_instruction'))]))
    assert any(h['field']=='source_role' and h['value']=='explicit_task_instruction' for h in found)
    assert audit.explicit_candidates(found)


def test_inventory_keeps_clock_scope_and_does_not_unpickle_opaque_npz(tmp_path):
    path=tmp_path/'raw.npz'
    np.savez(path,source_role=np.array(dict(role='secret'),dtype=object),
        available_before_reset=np.bool_(True))
    result=audit.Inventory().inspect(path)
    assert result['birthtime'] is None or isinstance(result['birthtime'],float)
    assert 'does not certify' in result['timestamp_scope']
    assert any(h['value']=='<opaque object; not decoded>' for h in result['metadata_hits'])
    assert result['read_error'] is None


def test_large_reference_array_is_not_silently_ignored(tmp_path):
    path=tmp_path/'candidate.npz'
    np.savez(path,scheduler_reference_poses=np.zeros((100,4,4)))
    result=audit.Inventory().inspect(path)
    assert audit.explicit_candidates(result['metadata_hits'])
