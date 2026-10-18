import numpy as np
from wam_rl_policy.local_goal_metrics import local_goal_outcomes


def test_fixed_horizon_alignment_and_censored_tail():
    poses=np.repeat(np.eye(4)[None],66,0);poses[:,0,3]=np.arange(66)/60
    goals=poses[:-1].copy();goals[:,0,3]+=.8
    result=local_goal_outcomes(poses,goals)
    assert result['ticks'].tolist()==list(range(18))
    assert result['censored_tail_goals']==47
    np.testing.assert_allclose(result['endpoint_position_error_m'],0,atol=1e-15)
    np.testing.assert_allclose(result['position_error_reduction_m'],.8)
    short=local_goal_outcomes(poses[:10],goals[:9])
    assert len(short['ticks'])==0 and short['censored_tail_goals']==9
