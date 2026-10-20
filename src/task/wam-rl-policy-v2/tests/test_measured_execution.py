from types import SimpleNamespace

import numpy as np
import pytest

from wam_rl_policy_v2.measured_execution import (
    environment_geometry, online_support, run_single_episode_batch,
)
from wam_rl_policy_v2.task_runtime import TaskController
from wam_rl_policy_v2.task_scheduler import TaskScheduler, TaskSchedulerConfig
from wam_rl_policy_v2.task_spec import TaskSpec, Phase
from wam_rl_policy_v2.task_feedback import InteractionFeedback


def fixture():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = .1
    points = np.array([[x,y,z] for x in (-.01,.01) for y in (-.01,.01) for z in (-.02,.02)])
    spec = TaskSpec.from_reference(poses, source_fps=30, terminal_mode='RELEASE', trim_interaction=False)
    return poses, points, spec


def test_geometry_binding_uses_batch_order_with_nonleading_and_repeated_objects():
    objects = ['dexplore/banana', 'dexplore/apple', 'dexplore/banana']
    runtime = SimpleNamespace(env_objects=[dict(object_code=o) for o in objects], geometry=[object() for _ in objects])
    for i, code in enumerate(objects):
        assert environment_geometry(runtime, i, code) is runtime.geometry[i]
    with pytest.raises(ValueError, match='environment object'):
        environment_geometry(runtime, 0, 'dexplore/apple')


def test_live_support_enables_release_and_airborne_or_invalid_force_does_not():
    poses, points, spec = fixture()
    poses[-1,2,3] = .82
    spec = TaskSpec.from_reference(poses, source_fps=30, terminal_mode='RELEASE', trim_interaction=False)
    bounds = np.array([[-.5,-.5,-.05],[.5,.5,0]])
    snapshot = dict(contact_force_valid=[True], object_contact_force=[[0,0,1]], table_pose=[[0,0,.8,0,0,0,1]])
    scheduler = TaskScheduler(spec, points)
    scheduler.phase = Phase.RELEASE
    for tick in range(35):
        supported = online_support(snapshot, 0, poses[-1], points, bounds)
        decision = scheduler.update(poses[-1], tick/30, InteractionFeedback(released=True, stable=True, supported=supported))
        if decision.terminated:
            break
    assert decision.task_complete
    lifted = poses[-1].copy(); lifted[2,3] += .1
    assert not online_support(snapshot, 0, lifted, points, bounds)
    snapshot['contact_force_valid'] = [False]
    assert not online_support(snapshot, 0, poses[-1], points, bounds)
    snapshot.pop('table_pose')
    with pytest.raises(ValueError, match='live object contact'):
        online_support(snapshot, 0, poses[-1], points, bounds)


def test_batch_stops_at_first_real_task_terminal_without_postterminal_action():
    poses, points, spec = fixture()
    contexts = [TaskController(spec, points), TaskController(spec, points,
                scheduler_config=TaskSchedulerConfig(acquire_timeout_s=30))]
    decisions = [None, None]
    hand = np.full((11,3), .5)
    sent, observed, progress = [], [], []

    def observe(state):
        for i, ctx in enumerate(contexts):
            decisions[i] = ctx.observe(poses[0], hand, np.zeros(6), state['tick'][i]/30,
                                       contact_forces=np.zeros((5,3)), force_valid=True)['decision']
        observed.append(list(state['tick']))
        return np.zeros((2,12)), state

    def send(_, command):
        sent.append(command)

    def receive(_):
        t = len(sent)
        return dict(state=dict(tick=[t,t], episodes=[0,0]))

    action, state = observe(dict(tick=[0,0], episodes=[0,0]))
    completed = run_single_episode_batch(None, action, state, 1000, observe=observe,
        on_step=progress.append, is_terminal=lambda: any(d.terminated for d in decisions),
        send=send, receive=receive, expand_action=lambda a:a)
    assert completed == len(sent) == 450
    assert len(observed) == 451 and progress[-1] == 450
    assert decisions[0].failure_reason == 'acquire_timeout' and not decisions[1].terminated


def test_collector_rejects_hidden_episode_reset_before_observing_it():
    calls = []
    with pytest.raises(ValueError, match='crossed an episode'):
        run_single_episode_batch(None, np.zeros((2,12)), dict(episodes=[0,0],tick=[9,9]), 1,
            observe=lambda state:calls.append(state), on_step=lambda _:None, is_terminal=lambda:False,
            send=lambda *_:None, receive=lambda _:dict(state=dict(episodes=[1,0],tick=[0,10])), expand_action=lambda a:a)
    assert not calls
