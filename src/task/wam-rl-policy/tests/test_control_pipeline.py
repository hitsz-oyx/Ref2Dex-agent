import numpy as np
import pytest
import torch
from wam_rl_policy.control_contracts import MeasuredShadowState,TaskIntent,ExecutionFeedback,goal_envelope
from wam_rl_policy.control_pipeline import ScheduledShadowControl
from wam_rl_policy.goal_scheduler import OrderedGoalScheduler


class Runtime:
    device=torch.device('cpu')
    def __init__(self,points):self.geometry=[(None,None,points)];self.calls=0
    def batch(self,objects,hands,*,goal_delta):
        self.calls+=1;self.last_goal=goal_delta.copy();return goal_delta
    def predict(self,batch):return batch
    def tau_for_student(self,prediction,pose):return np.full((8,11,3),.003,np.float32)
    def interaction_for_executor(self,prediction):return np.zeros((1,24,128,7)),np.ones((1,128),bool)


class Actor(torch.nn.Module):
    current_model=True
    def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.full((24,),2.))
    def forward(self,x):assert x.shape[-1]==565;return self.bias.expand(len(x),-1)
    def distribution(self,x):return torch.distributions.Normal(self(x),torch.ones(24)*.01)


def pipeline():
    points=np.random.RandomState(4).randn(32,3)*.01
    poses=np.repeat(np.eye(4)[None],2,0);poses[1,2,3]=.1
    scheduler=OrderedGoalScheduler(poses,[0,2],points)
    runtime=Runtime(points);actor=Actor()
    control=ScheduledShadowControl(runtime,scheduler,actor,TaskIntent('lift'))
    control.reset('episode-a')
    return control,runtime


def state(tick,**kwargs):
    return MeasuredShadowState('episode-a',tick,tick/60,np.zeros(300),np.eye(4),
        np.zeros((11,3)),kwargs.pop('hand_valid',np.ones(11,bool)),**kwargs)


def test_full_boundary_clock_goal_actions_and_ppo_replay():
    control,runtime=pipeline()
    feedback=ExecutionFeedback(execution_ready=False,source='native_contact_monitor')
    for tick in range(7):
        command=control.compute(state(tick),feedback=feedback,phase='grasp',sample=True)
        assert command['goal']['semantics']=='spatial_path_waypoint'
        assert command['goal']['reference_endpoint_time_s'] is None
        assert command['goal']['reachability_certified'] is False
        assert command['task_success'] is None and not command['phase_consumed_by_wam']
        assert command['scheduler']['confirmed_index']==0
        assert command['tau_contract']['valid'].all()==(tick==6)
        assert np.all(command['raw_action']>1) and np.all(command['canonical_action']==1)
        x=torch.as_tensor(command['executor_observation'])[None]
        raw=torch.as_tensor(command['raw_action'])[None]
        replay=control.actor.distribution(x).log_prob(raw).sum(-1).item()
        assert replay==command['action_log_prob']
        def sim_step(action):action[:,:6].mul_(7)
        control.execute(command,sim_step)
        np.testing.assert_array_equal(command['canonical_action'],1)
        with pytest.raises(ValueError,match='already executed'):control.execute(command,sim_step)
    assert runtime.calls==1
    np.testing.assert_allclose(runtime.last_goal[0],command['goal']['goal_delta'])
    assert command['tau_contract']['target_offsets_s'][-1]==8/60


def test_boundary_rejects_missing_robot_mismatch_stale_and_goal_relabel():
    with pytest.raises(ValueError,match='missing'):state(0,hand_valid=np.zeros(11,bool))
    with pytest.raises(ValueError,match='mismatch'):state(0,robot_id='inspire')
    with pytest.raises(ValueError,match='evidence'):ExecutionFeedback(execution_ready=True)
    control,_=pipeline();old=control.compute(state(0));control.compute(state(1))
    with pytest.raises(ValueError,match='stale'):control.execute(old,lambda _:None)
    with pytest.raises(ValueError,match='consecutive'):control.compute(state(3))
    decision=dict(old['scheduler']);decision['goal_semantics']='timed_reference_endpoint'
    with pytest.raises(ValueError,match='spatial'):goal_envelope(decision,state(0),control.bridge.scheduler.points)
    decision=dict(old['scheduler']);decision['goal_delta']=decision['goal_delta']+.1
    with pytest.raises(ValueError,match='disagrees'):goal_envelope(decision,state(0),control.bridge.scheduler.points)
    control.reset('episode-b')
    with pytest.raises(ValueError,match='reset'):control.compute(state(0))
