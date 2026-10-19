from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from wam_rl_policy_v2.bc_mlp import CurrentMLP,NormalizedCommandMLP
from wam_rl_policy_v2.executor_contract import ExecutorLayout,expand_named_model
from wam_rl_policy_v2.robot_adapter import InspireAdapter
from wam_rl_policy_v2.task_spec import Phase
from wam_rl_policy_v2.task_reward import TaskReward,RewardFeatures,potential


def test_shadow_expansion_preserves_all_old_tau_columns_and_normalization():
    torch.set_num_threads(2);parent=CurrentMLP(565)
    with torch.no_grad():
        parent.net[-1].weight.normal_(std=.01);parent.x_mean.normal_();parent.x_scale.uniform_(.1,1.)
    layout=ExecutorLayout('shadow',300,24,60)
    child=expand_named_model(parent,dict(robot=300,tau=264,tau_valid=1),layout.blocks,clip_dim=300)
    old=torch.randn(4,565);task=torch.randn(4,10)
    new=torch.cat((old[:,:300],task,old[:,300:]),-1)
    torch.testing.assert_close(child(new),parent(old),atol=1e-6,rtol=1e-5)
    assert child.input_dim==575 and not child.net[0].weight[:,300:310].count_nonzero()


def test_inspire_adds_force_validity_without_clipping_physical_force_to_five():
    snap=dict(q=np.zeros(18),dq=np.zeros(18),physical=np.zeros(248),objects=np.zeros(13),
        dof_force=np.full(18,12.),contact_forces=np.full((5,3),20.),dof_force_valid=False,contact_force_valid=True)
    state=InspireAdapter.state(snap,np.zeros(3),np.zeros(12),shape_feature=np.ones(64))
    assert state.values.shape==(411,) and not state.values[312:330].any()
    assert np.all(state.values[330:345]==20) and state.values[-2:].tolist()==[0,1]
    layout=ExecutorLayout('inspire',411,12,30);task=np.r_[np.zeros(7),1,0,0]
    assert layout.pack(state.values,task).shape==(686,)
    assert layout.packet()['tau_times_s'][-1]==8/30
    with pytest.raises(ValueError,match='valid tau requires'):layout.pack(state.values,task,tau_valid=True)


def test_current_to_scheduled_expansion_preserves_task_and_force_information():
    parent=NormalizedCommandMLP(421,312,12)
    with torch.no_grad():parent.net[-1].weight.normal_(std=.01)
    child=expand_named_model(parent,dict(robot=411,task=10),dict(robot=411,task=10,tau=264,tau_valid=1),clip_dim=312)
    value=torch.randn(3,421)
    torch.testing.assert_close(child(torch.cat((value,torch.randn(3,265)),-1)),parent(value),atol=1e-6,rtol=1e-5)


def test_phase_potential_telescopes_and_terminal_is_zero():
    features=RewardFeatures(palm_gap_m=.2,mean_tip_gap_m=.1,tracking_error_m=.02)
    states=[SimpleNamespace(phase=p,path_progress=u,terminated=False,task_complete=False,failure_reason=None,transitions=())
        for p,u in [(Phase.ACQUIRE,0),(Phase.TRACK,.2),(Phase.ACQUIRE,0),(Phase.TRACK,.4)]]
    states[-1].terminated=True;states[-1].failure_reason='timeout'
    reward=TaskReward(event_bonus=0,success_bonus=0,failure_cost=0,time_cost_per_s=0);total=0.;gamma=.96
    for i in range(3):
        value,parts=reward.step(states[i],states[i+1],features,features,gamma=gamma,dt_s=1/30)
        total+=gamma**i*value
    assert total==pytest.approx(-potential(states[0],features))
    with pytest.raises(ValueError,match='reset reward'):reward.step(states[0],states[1],features,features,gamma=gamma,dt_s=1/30)


def test_eventless_wam_checkpoint_cannot_silently_consume_new_event_fields():
    from wam_rl_policy_v2.task_spec import TaskSpec,TaskEvent
    from wam_rl_policy_v2.task_scheduler import TaskScheduler
    from wam_rl_policy_v2.task_feedback import InteractionFeedback
    from wam_rl_policy_v2.task_runtime import wam_goal,EVENT_SCHEMA
    poses=np.repeat(np.eye(4)[None],31,0)
    spec=TaskSpec.from_reference(poses,source_fps=30,terminal_mode='HOLD',events=(TaskEvent(10,'CONTACT'),))
    points=np.random.default_rng(246).normal(size=(32,3))*.02
    decision=TaskScheduler(spec,points).update(poses[0],0.,InteractionFeedback())
    old=SimpleNamespace()
    with pytest.raises(ValueError,match='no trained planned-event'):wam_goal(old,poses[0],decision,points,require_events=True)
    fields,metadata=wam_goal(old,poses[0],decision,points)
    assert set(fields)=={'goal_delta'} and not metadata['event_consumed']
    fields,metadata=wam_goal(SimpleNamespace(event_conditioning_schema=EVENT_SCHEMA),poses[0],decision,points,require_events=True)
    assert metadata['event_consumed'] and fields['event_type']=='CONTACT' and fields['event_delta_steps']==10
