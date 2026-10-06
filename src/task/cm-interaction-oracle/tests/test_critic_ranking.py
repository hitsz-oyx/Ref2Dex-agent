import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from critic_ranking import bootstrapped_score,selection_summary


def test_discounted_reward_and_endpoint_bootstrap_use_correct_clock():
    rewards=np.zeros((2,7,8));rewards[...,0]=2;rewards[...,7]=3
    values=np.full((2,7),5.);done=np.zeros_like(rewards,dtype=bool)
    s,r,v=bootstrapped_score(rewards,values,done,gamma=.5)
    assert np.all(r==2+3*.5**7)
    assert np.all(v==5*.5**8)
    assert np.all(s==r+v)
    done[0,0,3]=True
    with pytest.raises(ValueError,match='terminal'):
        bootstrapped_score(rewards,values,done)


def test_critic_selection_reports_rescue_and_harm_without_imitating_y():
    z=np.zeros((2,7),dtype=bool);z[0,2]=True;z[1,0]=True
    y=np.zeros((2,7));y[0,2]=1
    scores=np.zeros((2,7));scores[:,2]=1
    out=selection_summary(scores,z,y)
    assert out['rescued']==1 and out['harmed']==1
    assert out['successes']==1 and out['gt_y_successes']==2
    assert out['selection']==[2,2]
    tied=selection_summary(np.zeros_like(scores),z,y)
    assert tied['selection']==[0,0]
    assert tied['ranking_vs_Z']['pairwise_accuracy']==.5
