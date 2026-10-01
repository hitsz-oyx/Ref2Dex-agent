import numpy as np
import torch
from scripts.analyze_selector_pool_contact import cluster_interval,contrast_support


def test_common_actions_cancel_and_aliases_do_not_inflate_different_command_support():
    commands=torch.zeros(100,5,12);matches=torch.ones(100,5,dtype=torch.bool)
    # Recommender IDs may differ on every row; identical physical commands
    # still contribute zero policy difference and no intervention support.
    episodes=[f'e{i}' for i in range(100)]
    support=contrast_support(matches,commands,episodes,1)
    assert support['different_command_windows']==0
    assert support['cm']['windows']==0 and not support['sufficient']
    commands[:80,1,0]=1;matches[:40,1]=False;matches[40:80,0]=False
    support=contrast_support(matches,commands,episodes,1)
    assert support['cm']['windows']==40 and support['control']['windows']==40
    assert support['sufficient']


def test_merged_action_inverse_probability_recovers_balanced_known_local_effect():
    # Three of five recommendations execute actionA, two actionB. Complete
    # balanced randomized blocks must recover A=10,B=4 and effect6.
    assigned=np.tile([0,0,0,1,1],20);probability=np.where(assigned==0,.6,.4)
    outcome=np.where(assigned==0,10.,4.)
    contribution=((assigned==0).astype(float)-(assigned==1).astype(float))*outcome/probability
    report=cluster_interval(contribution,[f'frame{i//5}' for i in range(100)])
    assert abs(report['value']-6.)<1e-10
    assert np.allclose(report['interval90'],[6.,6.])
    assert cluster_interval([],[])['interval90'] is None
