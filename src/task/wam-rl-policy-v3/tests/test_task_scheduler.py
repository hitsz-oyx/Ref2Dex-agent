from dataclasses import replace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from wam_rl_policy_v3.task_spec import TaskSpec,TaskEvent,EventType,TerminalMode,Phase
from wam_rl_policy_v3.task_scheduler import TaskScheduler
from wam_rl_policy_v3.task_feedback import InteractionFeedback,FeedbackObserver
from wam_rl_policy_v3.executor_contract import task_condition


def points():return np.random.default_rng(246).normal(size=(32,3))*.02


def reference(n=31):
    poses=np.repeat(np.eye(4)[None],n,0);poses[:,0,3]=np.linspace(0,.1,n)
    return poses


def test_simplification_preserves_return_and_source_event_frames():
    poses=reference(12);poses[:4]=poses[0];poses[8:]=poses[0]
    spec=TaskSpec.from_reference(poses,source_fps=30,terminal_mode='RELEASE',
        events=(TaskEvent(3,'CONTACT'),TaskEvent(10,'RELEASE')))
    assert spec.source_frames[0]==3 and spec.source_frames[-1]==10
    assert len(spec.object_path)>2 and np.array_equal(spec.object_path[0],spec.object_path[-1])
    restored=TaskSpec.from_packet(spec.packet())
    np.testing.assert_array_equal(restored.source_frames,spec.source_frames)
    assert not restored.object_path.flags.writeable
    repeated=TaskSpec.from_reference(poses,source_fps=30,terminal_mode='HOLD',
        events=(TaskEvent(1,'CONTACT'),TaskEvent(4,'RELEASE'),TaskEvent(8,'CONTACT')))
    assert len(repeated.events)==3 and 4 in repeated.source_frames
    with pytest.raises(ValueError,match='must alternate'):
        TaskSpec.from_reference(poses,source_fps=30,terminal_mode='HOLD',events=(TaskEvent(1,'CONTACT'),TaskEvent(4,'CONTACT')))


def test_static_contact_requests_probe_and_planned_contact_does_not_advance_phase():
    poses=np.repeat(np.eye(4)[None],31,0)
    spec=TaskSpec.from_reference(poses,source_fps=30,terminal_mode='HOLD',events=(TaskEvent(3,'CONTACT'),))
    scheduler=TaskScheduler(spec,points());candidate=InteractionFeedback(grasp_candidate=True,force_valid=True)
    for t in range(24):
        d=scheduler.update(poses[0],t/30,candidate)
        assert d.phase==Phase.ACQUIRE and d.path_progress==0 and not d.task_complete and d.probe_active
        assert np.linalg.norm(d.local_goal_pose[:3,3])<=.00300001
    assert d.next_event.delta_s==0 and d.next_event.valid
    assert d.next_event.wam_condition()['event_delta_steps']==0
    assert task_condition(poses[0],d)[-3:].tolist()==[1,0,0]
    d=scheduler.update(poses[0],.8,InteractionFeedback(grasp_established=True,stable=True))
    assert d.phase==Phase.TRACK and 'grasp_acquired' in d.transitions and d.next_event.kind is None
    assert d.next_event.wam_condition()==dict(event_type=0,event_delta_steps=0.,event_valid=True,event_hz=30)


def test_force_candidate_needs_observed_motion_coupling():
    observer=FeedbackObserver(points());pose=np.eye(4);hand=np.zeros((11,3));force=np.zeros((5,3));force[:2,2]=1
    for t in range(15):f=observer.update(pose,hand,np.zeros(6),t/30,contact_forces=force,force_valid=True)
    assert f.grasp_candidate and not f.grasp_established
    for t in range(15,25):
        p=pose.copy();p[2,3]=(t-14)*.001
        f=observer.update(p,hand+np.array([0,0,p[2,3]]),np.array([0,0,.03,0,0,0]),t/30,contact_forces=force,force_valid=True)
    assert f.grasp_established


def test_table_contact_force_does_not_block_geometric_release_from_object():
    observer=FeedbackObserver(points());pose=np.eye(4);force=np.ones((5,3))
    f=observer.update(pose,np.zeros((11,3)),np.zeros(6),0.,contact_forces=force,force_valid=True)
    assert f.contact_observed and not f.grasp_candidate
    for t in range(1,9):
        f=observer.update(pose,np.full((11,3),.5),np.zeros(6),t/30,contact_forces=force,force_valid=True)
    assert f.released and not f.grasp_candidate and not f.contact_observed


def test_closed_path_cannot_complete_at_its_initial_endpoint():
    poses=np.repeat(np.eye(4)[None],121,0);theta=np.linspace(0,2*np.pi,121)
    poses[:,0,3]=.05*(np.cos(theta)-1);poses[:,1,3]=.05*np.sin(theta)
    spec=TaskSpec.from_reference(poses,source_fps=30,terminal_mode='HOLD')
    scheduler=TaskScheduler(spec,points())
    for t in range(170):
        d=scheduler.update(poses[min(t,120)],t/30,InteractionFeedback(grasp_established=True,stable=t>=120))
        if t<120:assert not d.path_complete and not d.task_complete
        if d.terminated:break
    assert d.task_complete and d.path_progress==1
    with pytest.raises(ValueError,match='requires reset'):scheduler.update(poses[-1],6.,InteractionFeedback())


def test_release_requires_goal_support_and_measured_clearance_then_dwell():
    poses=reference();spec=TaskSpec.from_reference(poses,source_fps=30,terminal_mode='RELEASE',
        events=(TaskEvent(0,'CONTACT'),TaskEvent(30,'RELEASE')))
    scheduler=TaskScheduler(spec,points());held=InteractionFeedback(grasp_established=True,stable=True)
    t=0
    for t in range(60):
        d=scheduler.update(poses[min(t,30)],t/30,held)
        if d.phase==Phase.RELEASE:break
    assert d.phase==Phase.RELEASE and not d.task_complete
    released=InteractionFeedback(released=True,stable=True,supported=None)
    for _ in range(35):t+=1;d=scheduler.update(poses[-1],t/30,released)
    assert not d.task_complete
    released=replace(released,supported=True)
    for _ in range(40):
        t+=1;d=scheduler.update(poses[-1],t/30,released)
        if d.terminated:break
    assert d.task_complete


def test_recovery_preserves_confirmed_progress_and_does_not_repay_grasp_bonus():
    poses=reference();scheduler=TaskScheduler(TaskSpec.from_reference(poses,source_fps=30,terminal_mode='HOLD'),points())
    held=InteractionFeedback(grasp_established=True)
    for t in range(16):d=scheduler.update(poses[t],t/30,held)
    progress=d.path_progress;assert progress>0
    d=scheduler.update(poses[15],16/30,InteractionFeedback(grasp_lost=True))
    assert d.phase==Phase.ACQUIRE and d.recovering and d.path_progress==progress
    d=scheduler.update(poses[15],17/30,held)
    assert d.phase==Phase.TRACK and d.path_progress>=progress and 'grasp_acquired' not in d.transitions


def test_shadow_position_goal_does_not_add_an_initial_orientation_constraint():
    spec=TaskSpec.shadow_goal(np.eye(4),[.1,0,0]);scheduler=TaskScheduler(spec,points())
    actual=np.eye(4);actual[:3,:3]=Rotation.from_euler('z',.8).as_matrix()
    d=scheduler.update(actual,0.,InteractionFeedback(grasp_established=True))
    np.testing.assert_allclose(task_condition(actual,d)[3:6],0,atol=1e-7)
    assert d.goal_type=='position_only'


def test_geometry_is_observed_before_delayed_grasp_confirmation():
    poses=reference();s=TaskScheduler(TaskSpec.from_reference(poses,source_fps=30,terminal_mode='HOLD'),points())
    for t in range(11):
        d=s.update(poses[t],t/30,InteractionFeedback(grasp_candidate=True,grasp_established=t==10))
        if t<10:assert d.phase==Phase.ACQUIRE and not d.task_complete and d.path_progress==0 and d.source_cursor_frame==0
    assert d.phase==Phase.TRACK and d.path_progress>0
    assert s.path.initial_acquired and d.source_cursor_frame>0


def test_path_timeout_budget_does_not_include_acquisition_wait():
    poses=reference();s=TaskScheduler(TaskSpec.from_reference(poses,source_fps=30,terminal_mode='HOLD'),points())
    for t in range(13):
        d=s.update(poses[0],float(t),InteractionFeedback())
        assert not d.terminated
    d=s.update(poses[0],13.,InteractionFeedback(grasp_established=True))
    assert d.phase==Phase.TRACK and not d.terminated


def test_uncontrolled_object_path_cannot_be_committed_as_pregrasp_progress():
    poses=reference();s=TaskScheduler(TaskSpec.from_reference(poses,source_fps=30,terminal_mode='HOLD'),points())
    for t in range(30):
        d=s.update(poses[t],t/30,InteractionFeedback())
        assert d.path_progress==0 and not d.path_complete
    d=s.update(poses[-1],1.,InteractionFeedback(grasp_established=True))
    assert d.path_progress==0 and not d.path_complete


def test_failed_candidate_discards_its_unconfirmed_progress():
    poses=reference();s=TaskScheduler(TaskSpec.from_reference(poses,source_fps=30,terminal_mode='HOLD'),points())
    for t in range(16):d=s.update(poses[t],t/30,InteractionFeedback(grasp_candidate=True))
    assert s.path.pointer>0 and d.path_progress==0
    d=s.update(poses[15],16/30,InteractionFeedback())
    assert s.path.pointer==0 and d.path_progress==0
    d=s.update(poses[15],17/30,InteractionFeedback(grasp_established=True))
    assert d.path_progress==0


def test_unplanned_drop_keeps_planned_release_and_requests_regrasp_contact():
    poses=reference();spec=TaskSpec.from_reference(poses,source_fps=30,terminal_mode='RELEASE',
        events=(TaskEvent(0,'CONTACT'),TaskEvent(30,'RELEASE')))
    scheduler=TaskScheduler(spec,points());held=InteractionFeedback(grasp_established=True,stable=True)
    for t in range(16):d=scheduler.update(poses[t],t/30,held)
    prior=d.path_progress
    d=scheduler.update(poses[15],16/30,InteractionFeedback(grasp_lost=True,released=True))
    assert d.phase==Phase.ACQUIRE and d.recovering
    assert scheduler.event_index==1  # The intended terminal release was not completed by a drop.
    assert d.next_event.kind==EventType.CONTACT and d.next_event.valid
    d=scheduler.update(poses[15],17/30,InteractionFeedback(released=True))
    assert scheduler.event_index==1 and d.next_event.kind==EventType.CONTACT
    d=scheduler.update(poses[15],18/30,held)
    assert d.phase==Phase.TRACK and not d.recovering and d.path_progress>=prior
    assert d.next_event.kind==EventType.RELEASE and scheduler.event_index==1
    for t in range(19,80):
        pose=poses[min(30,t-3)]
        d=scheduler.update(pose,t/30,held)
        if d.phase==Phase.RELEASE:break
    assert d.phase==Phase.RELEASE
    t+=1;d=scheduler.update(poses[-1],t/30,InteractionFeedback(released=True,stable=True,supported=True))
    assert scheduler.event_index==2 and d.next_event.kind is None


def test_drop_before_declared_release_pose_keeps_terminal_release_pending():
    poses=reference();spec=TaskSpec.from_reference(poses,source_fps=30,terminal_mode='RELEASE',
        events=(TaskEvent(0,'CONTACT'),TaskEvent(30,'RELEASE')))
    scheduler=TaskScheduler(spec,points());held=InteractionFeedback(grasp_established=True,stable=True)
    for t in range(16):d=scheduler.update(poses[t],t/30,held)
    assert d.phase==Phase.TRACK and d.source_cursor_frame<30
    d=scheduler.update(poses[15],16/30,InteractionFeedback(grasp_lost=True,released=True))
    assert d.phase==Phase.ACQUIRE and d.next_event.kind==EventType.CONTACT
    assert scheduler.event_index==1
