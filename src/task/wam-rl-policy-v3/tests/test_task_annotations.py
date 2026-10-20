import numpy as np
import pytest
from wam_rl_policy_v3.task_annotations import planned_contact_events,reference_task_spec,task_from_teacher_archive,REFERENCE_CONTACT_COLUMNS
from wam_rl_policy_v3.task_spec import TerminalMode,Phase
from wam_rl_policy_v3.task_scheduler import TaskScheduler
from wam_rl_policy_v3.task_feedback import InteractionFeedback


def test_multiple_planned_intervals_keep_original_frame_numbers():
    ref=np.zeros((40,428));ref[5:12,REFERENCE_CONTACT_COLUMNS]=1
    ref[14:17,REFERENCE_CONTACT_COLUMNS]=1 # two-frame gap merges
    ref[25:,REFERENCE_CONTACT_COLUMNS]=1
    events,terminal=planned_contact_events(ref)
    assert [(e.source_frame,e.kind.value) for e in events]==[(5,'CONTACT'),(17,'RELEASE'),(25,'CONTACT')]
    assert terminal==TerminalMode.HOLD
    pose=np.repeat(np.eye(4)[None],40,0)
    spec=reference_task_spec(pose,ref)
    assert spec.source_length==40 and spec.source_frames.tolist()==[5,17,25,39]
    ref[35:]=0
    assert planned_contact_events(ref)[1]==TerminalMode.RELEASE
    ref[:]=0
    assert planned_contact_events(ref)==((),TerminalMode.UNKNOWN)
    ref[3,REFERENCE_CONTACT_COLUMNS[0]]=-1
    with pytest.raises(ValueError,match='binary'):planned_contact_events(ref)


def test_event_acknowledgement_requires_new_measured_edges_not_elapsed_time():
    ref=np.zeros((40,428));ref[3:12,REFERENCE_CONTACT_COLUMNS]=1;ref[25:,REFERENCE_CONTACT_COLUMNS]=1
    pose=np.repeat(np.eye(4)[None],40,0);pose[:,0,3]=np.arange(40)*.003
    spec=reference_task_spec(pose,ref)
    s=TaskScheduler(spec,np.random.default_rng(246).normal(size=(32,3))*.02)
    held=InteractionFeedback(grasp_established=True)
    d=s.update(pose[3],0,held)
    assert d.next_event.kind.value=='RELEASE'
    # A persistent grasp cannot acknowledge either a release or another contact.
    for i in range(1,6):d=s.update(pose[3],i/30,held)
    assert d.next_event.kind.value=='RELEASE'
    d=s.update(pose[12],6/30,InteractionFeedback(grasp_lost=True,released=True))
    assert d.phase==Phase.ACQUIRE and d.recovering and d.next_event.kind.value=='CONTACT'
    d=s.update(pose[12],7/30,InteractionFeedback(released=True))
    assert d.next_event.kind.value=='CONTACT'
    d=s.update(pose[25],8/30,held)
    # This release had no confirmed path context; regrasp must not consume
    # the still-pending planned release/contact pair.
    assert d.phase==Phase.TRACK and d.next_event.kind.value=='RELEASE' and d.next_event.valid
    assert s.event_index==1
    assert 'grasp_acquired' not in d.transitions


def test_intermediate_planned_release_can_acknowledge_at_confirmed_source_pose():
    ref=np.zeros((40,428));ref[3:12,REFERENCE_CONTACT_COLUMNS]=1;ref[25:,REFERENCE_CONTACT_COLUMNS]=1
    pose=np.repeat(np.eye(4)[None],40,0);pose[:,0,3]=np.arange(40)*.003
    s=TaskScheduler(reference_task_spec(pose,ref),np.random.default_rng(246).normal(size=(32,3))*.02)
    held=InteractionFeedback(grasp_established=True)
    for i in range(10):d=s.update(pose[3+i],i/30,held)
    assert s.plan_cursor_frame>=12 and d.next_event.kind.value=='RELEASE'
    d=s.update(pose[12],10/30,InteractionFeedback(grasp_lost=True,released=True))
    assert d.recovering and s.event_index==2 and d.next_event.kind.value=='CONTACT'
    d=s.update(pose[12],11/30,InteractionFeedback(released=True))
    assert s.event_index==2
    d=s.update(pose[12],12/30,held)
    assert d.phase==Phase.TRACK and s.event_index==3 and d.next_event.kind is None


def test_teacher_task_ignores_training_reference_and_preserves_actual_initial_anchor():
    n=40;pose=np.repeat(np.eye(4)[None],n,0);pose[:,0,3]=np.linspace(0,.1,n)
    forces=np.zeros((n,5,3));forces[5:30,:2,2]=1
    data=dict(obj=pose,body_contact_forces=forces,contact_force_valid=np.arange(n)>0,frame_tick=np.arange(n),
        reference_obj=np.full_like(pose,np.nan),reference_current=np.full((n,428),np.nan))
    a=task_from_teacher_archive(data,list(range(5)))
    data['reference_obj'][:]=1e6;data['reference_current'][:]=1
    b=task_from_teacher_archive(data,list(range(5)))
    assert a.packet()==b.packet()
    assert a.source_role=='same_execution_teacher_object_rollout' and a.source_frames[0]==0 and a.source_frames[-1]==39
    assert [(e.source_frame,e.kind.value) for e in a.events]==[(5,'CONTACT'),(30,'RELEASE')]
    np.testing.assert_array_equal(a.object_path[0],pose[0]);np.testing.assert_array_equal(a.object_path[-1],pose[-1])
