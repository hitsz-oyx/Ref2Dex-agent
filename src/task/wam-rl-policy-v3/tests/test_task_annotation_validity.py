from dataclasses import replace
import numpy as np
import pytest
from wam_rl_policy_v3.task_spec import TaskSpec
from wam_rl_policy_v3.task_annotations import teacher_rollout_task_spec,task_with_safe_prefix
from wam_rl_policy_v3.task_scheduler import TaskScheduler


def test_missing_raw_evidence_is_explicit_and_never_becomes_a_hold_task():
    pose=np.repeat(np.eye(4)[None],20,0);force=np.zeros((20,5,3));valid=np.zeros(20,bool)
    spec=teacher_rollout_task_spec(pose,force,valid)
    assert spec.terminal_mode.value=='UNKNOWN' and spec.annotation_invalid
    assert 'missing_raw_force_observations' in spec.annotation_invalid_reasons
    for cut in [0,3,18]:
        same,_=task_with_safe_prefix(spec,cut)
        assert same.packet()==spec.packet()
    with pytest.raises(ValueError,match='UNKNOWN'):TaskScheduler(spec,np.eye(3))


def test_defined_but_invalid_annotation_cannot_enter_full_task_scoring():
    pose=np.repeat(np.eye(4)[None],20,0);force=np.zeros((20,5,3));force[3:12,:2,2]=1
    valid=np.ones(20,bool);valid[8]=False
    spec=teacher_rollout_task_spec(pose,force,valid)
    assert spec.annotation_invalid and spec.terminal_mode.value=='RELEASE'
    with pytest.raises(ValueError,match='annotation_invalid'):TaskScheduler(spec,np.eye(3))
    restored=TaskSpec.from_packet(spec.packet())
    assert restored.annotation_invalid_reasons==spec.annotation_invalid_reasons


def test_missing_archive_validity_cannot_be_certified_by_assuming_tick_valid():
    from wam_rl_policy_v3.task_annotations import task_from_teacher_archive
    pose=np.repeat(np.eye(4)[None],20,0);force=np.zeros((20,5,3));force[3:12,:2,2]=1
    spec=task_from_teacher_archive(dict(obj=pose,frame_tick=np.arange(20),body_contact_forces=force),np.arange(5))
    assert spec.terminal_mode.value=='RELEASE'
    assert spec.annotation_invalid and 'missing_raw_force_validity' in spec.annotation_invalid_reasons
    with pytest.raises(ValueError,match='annotation_invalid'):TaskScheduler(spec,np.eye(3))
