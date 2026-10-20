import numpy as np
from wam_rl_policy_v3.inspire_ppo_bridge import shadow_path_reward,path_outcome,FAILURE_COST


def test_stationary_near_reference_is_not_paid_as_task_progress():
    objects=np.zeros((1,13));hands=np.zeros((1,11,3));hands[:,0,0]=.05;hands[:,[2,4,6,8,10],0]=.03
    decision=dict(reference_error_m=0.,goal_point_m=.01,confirmed_progress=0.,complete=False)
    reward,_=shadow_path_reward(objects,hands,np.zeros((1,12)),np.zeros(1),[decision],np.zeros(1))
    assert reward[0]<=0, 'stationary proximity must not receive a persistent positive task reward'


def test_progress_deviation_and_completion_reward_ordering():
    objects=np.zeros((1,13));hands=np.zeros((1,11,3));hands[:,0,0]=.05;hands[:,[2,4,6,8,10],0]=.03
    def score(error,progress,complete=False,reached=False):
        d=dict(reference_error_m=0.,goal_point_m=error,confirmed_progress=progress,complete=complete)
        return shadow_path_reward(objects,hands,np.zeros((1,12)),np.zeros(1),[d],np.zeros(1),[reached])[0][0]
    static=score(.01,0);advanced=score(.005,.01);deviated=score(.04,0);finished=score(0,1,True,True)
    assert deviated<static<advanced<finished
    assert score(.01,0,True,False)==static # sticky completion cannot keep paying away from endpoint
    assert FAILURE_COST>=2.1/(1-.96)


def test_sticky_path_event_cannot_be_final_success_after_departure_or_workspace_failure():
    departed=path_outcome(True,False,65,False)
    failed=path_outcome(True,True,65,True)
    for outcome in (departed,failed):
        assert outcome['ever_ordered_path_complete']
        assert not outcome['final_path_complete'] and not outcome['task_success']
    assert not path_outcome(False,True,65,False)['task_success']
    assert not path_outcome(True,True,29,False)['task_success']
    assert path_outcome(True,True,30,False)['task_success']
