"""Separate bounded physical Z from short continuation utility and ties."""
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'src/task/cm-interaction-oracle/src')]
from oracle_y_utility import stable_grasp_z,utility,select,candidate_deltas,noise_curve


def test_stability_requires_qualification_and_subsequent_no_drop():
    height=torch.full((4,90),.04);pair=torch.ones_like(height,dtype=torch.bool)
    height[1,70:]=0 # held long enough but then dropped
    height[2,:20]=0 # cannot qualify bystep60
    pair[3,80:86]=False # force proxy loses for six after stable
    result,detail=stable_grasp_z(height,pair,torch.zeros(4))
    assert result.tolist()==[True,False,False,False]
    assert detail['first_stable_step'].tolist()==[45,45,-1,45]


def test_y_tie_prefers_baseline_and_zero_noise_preserves_oracle_selection():
    y=np.zeros((2,7,8));y[0,2,7]=.5;y[1,:,7]=.4
    z=np.zeros((2,7));z[0,2]=1;z[1,0]=1
    assert select(utility(y)).tolist()==[2,0]
    curve=noise_curve(y,z,sigmas=(0,),repeats=4)['rows'][0]
    assert curve['selected_z']==1 and curve['top1_regret']==0 and curve['pairwise_accuracy']==1


def test_candidates_are_small_native_interventions_not_claimed_desired_flows():
    d=candidate_deltas();assert d.shape==(7,18) and not d[0].any()
    assert d[1,14]==.2 and d[2,14]==-.2 and d[3,8]==.2 and d[4,8]==-.2
    assert d[-1,2]==.01 and (d[-1,6:]==0).all()
