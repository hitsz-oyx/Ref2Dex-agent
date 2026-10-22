"""Audit-only guards: no human-label/velocity imputation or future reads."""
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest

TOOL=Path(__file__).resolve().parents[1]/'tools/audit/audit_event_raw_signals.py'
spec=importlib.util.spec_from_file_location('raw_signal_audit',TOOL)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_npy_headers_are_read_without_payload_or_pickle():
    stream=io.BytesIO();np.save(stream,np.zeros((5,3),np.float32));stream.seek(0)
    result=audit.header(stream)
    assert result==dict(shape=[5,3],dtype='<f4')
    assert stream.tell()<len(stream.getvalue())


def test_bad_headers_are_rejected():
    with pytest.raises(ValueError,match='NPY header'):audit.header(io.BytesIO(b'badmagic'))


def test_human_contact_and_program_are_not_robot_signals_or_taskspec():
    layer=dict(fields={k:dict(shape=[30,1],dtype='|b1') for k in ('contact','valid','near','program')})
    flags=audit.signal_flags([layer])
    assert not flags['robot_contact_forces'] and not flags['robot_force_valid']
    assert not flags['task_event_kind'] and not flags['task_object_path']


def test_robot_dq_and_pose7_do_not_become_object_velocity():
    fields=dict(object=dict(shape=[100,7]),q=dict(shape=[100,22]),dq=dict(shape=[100,22]),goal=dict(shape=[100,3]))
    flags=audit.signal_flags([dict(fields=fields)])
    assert flags['native_q_dq'] and flags['recorded_goal_position']
    assert not flags['native_object_velocity']


def test_missing_inputs_stop_before_state_machine_construction():
    with pytest.raises(ValueError,match='missing exact deployment inputs'):
        audit.replay_complete_record(dict(signals={}),3)


def complete_synthetic_record():
    # Independent task instruction and explicitly measured synthetic fixture;
    # these are never presented as curated data or prototype admissions.
    import sys
    sys.path.insert(0,str(audit.ROOT/'src/task/wam-rl-policy-v3/src'))
    from wam_rl_policy_v3.task_spec import TaskSpec,TaskEvent,EventType,TerminalMode
    poses=np.broadcast_to(np.eye(4),(8,4,4)).copy()
    query=np.zeros((32,3));query[:,0]=np.linspace(-.01,.01,32);query[:,1]=np.tile([-.01,.01],16)
    task=TaskSpec(poses,np.arange(8),30.,8,TerminalMode.HOLD,(TaskEvent(1,EventType.CONTACT),))
    observations=[dict(pose=pose,hand=np.full((11,3),.1),velocity=np.zeros(6),time=t/30,
        contact_forces=np.zeros((5,3)),force_valid=True,supported=True,workspace_ok=True,observation_valid=True)
        for t,pose in enumerate(poses)]
    return dict(signals={k:True for k in ('observed_pose','observed_hand','observed_clock','native_object_velocity',
        'robot_contact_forces','robot_force_valid','measured_support','measured_workspace')},
        task_spec=task.packet(),available_before_reset=True,reset_frame=0,query=query,surface=query,observations=observations)


def test_complete_independent_fixture_replays_without_reading_future():
    record=complete_synthetic_record()
    expected=audit.replay_complete_record(record,3)
    class Guard:
        def __getitem__(self,frame):
            assert frame<=3,'future observation read'
            return record['observations'][frame]
    guarded=dict(record,observations=Guard())
    assert audit.replay_complete_record(guarded,3)==expected
    assert all(set(('event_type','event_delta_steps','event_valid','known_NONE'))<=set(r) for r in expected)


def test_invalid_prefix_is_not_silently_bridged():
    record=complete_synthetic_record();record['observations'][1]['observation_valid']=False
    with pytest.raises(ValueError,match='invalid measured prefix'):
        audit.replay_complete_record(record,3)


def test_force_free_mode_cannot_confirm_the_same_release_semantics():
    record=complete_synthetic_record()
    from wam_rl_policy_v3.task_feedback import FeedbackObserver
    without=FeedbackObserver(record['surface']);with_force=FeedbackObserver(record['surface'])
    last=None
    for t in range(8):
        row=record['observations'][t]
        missing=without.update(row['pose'],row['hand'],row['velocity'],row['time'],force_valid=False)
        last=with_force.update(row['pose'],row['hand'],row['velocity'],row['time'],
            contact_forces=row['contact_forces'],force_valid=True)
        assert not missing.released
    assert last.released
