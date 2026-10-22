"""Declared release occurs before the recording ends; keep post-release scoring."""
import numpy as np
from wam_rl_policy_v3.task_spec import TaskSpec,TaskEvent,Phase
from wam_rl_policy_v3.task_scheduler import TaskScheduler
from wam_rl_policy_v3.task_feedback import InteractionFeedback


def fixture():
    poses=np.repeat(np.eye(4)[None],91,0)
    poses[:31,0,3]=np.linspace(0,.10,31)
    # An explicitly released object can settle another 2 mm before record end.
    poses[31:,0,3]=.102
    spec=TaskSpec.from_reference(poses,source_fps=30,terminal_mode='RELEASE',
        events=(TaskEvent(0,'CONTACT'),TaskEvent(30,'RELEASE')),trim_interaction=False)
    return poses,spec,np.random.default_rng(43).normal(size=(32,3))*.02


def test_release_at_declared_frame_does_not_require_extra_held_dwell():
    poses,spec,points=fixture();scheduler=TaskScheduler(spec,points)
    held=InteractionFeedback(grasp_established=True,stable=False)
    for t in range(31):d=scheduler.update(poses[t],t/30,held)
    assert d.phase==Phase.RELEASE and not d.task_complete
    assert not d.path_complete  # .2 s geometric settling still required.
    released=InteractionFeedback(released=True,grasp_lost=True,stable=True,supported=True)
    for t in range(31,91):
        d=scheduler.update(poses[t],t/30,released)
        if d.terminated:break
    assert d.task_complete and d.path_complete and scheduler.event_index==2
    assert t/30>=1.+scheduler.config.terminal_hold_s


def test_release_request_cannot_skip_ordered_transport():
    poses,spec,points=fixture();scheduler=TaskScheduler(spec,points)
    held=InteractionFeedback(grasp_established=True,stable=True)
    scheduler.update(poses[0],0.,held)
    # This jump exceeds the .08 m crossing bound and skips required milestones.
    d=scheduler.update(poses[30],1/30,held)
    assert d.phase==Phase.TRACK and not d.task_complete


def test_correct_release_does_not_waive_post_release_endpoint_or_support():
    poses,spec,points=fixture();scheduler=TaskScheduler(spec,points)
    held=InteractionFeedback(grasp_established=True,stable=False)
    for t in range(31):d=scheduler.update(poses[t],t/30,held)
    assert d.phase==Phase.RELEASE
    released=InteractionFeedback(released=True,grasp_lost=True,stable=True,supported=False)
    for t in range(31,91):d=scheduler.update(poses[t],t/30,released)
    assert not d.task_complete
    off=poses[-1].copy();off[0,3]+=.05
    released=InteractionFeedback(released=True,stable=True,supported=True)
    for t in range(91,121):d=scheduler.update(off,t/30,released)
    assert not d.task_complete
