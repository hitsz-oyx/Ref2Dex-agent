from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'src/task/cm-interaction-oracle/src'))
from conditional_consequence import environment_folds, permute_within, task_slots, contrast_scores, oracle_retention


def test_oof_keeps_all_waves_of_an_environment_together_and_test_unassigned():
    clusters=np.tile(np.arange(12),4); motion=clusters%3
    train=np.flatnonzero(clusters<9)
    folds=environment_folds(train,motion,clusters,11)
    for env in range(9): assert len(set(folds[clusters==env]))==1
    assert (folds[clusters>=9]==-1).all() and set(folds[train])=={0,1,2}


def test_shuffle_never_crosses_fit_hold_or_context_blocks():
    groups=np.tile(np.arange(3),20); ids=np.arange(30)
    permutation=permute_within(groups,ids,11)
    assert set(permutation[ids])==set(ids)
    assert np.array_equal(groups[permutation],groups)
    assert np.array_equal(permutation[30:],np.arange(30,60))


def test_downstream_slots_preserve_ref8_width_and_intended_action_only():
    h=torch.ones(5,104); a=torch.ones(5,14); z=torch.ones(5,26)
    x=task_slots(h,a,z)
    assert x.shape==(5,162) and x[:,104:118].count_nonzero()==0
    assert x[:,-18:].count_nonzero()==0
    assert torch.equal(task_slots(h,a,z,True)[:,104:118],a)


def test_contrast_metrics_detect_mean_predictor_and_sign_reversal():
    gt=np.arange(1,365,dtype=float).reshape(14,26)/50
    axes=np.arange(12,26); scale=np.ones(26)
    exact=contrast_scores(gt,gt,scale,axes)
    assert exact['gain_vs_zero']==1 and exact['signed_agreement']==1
    zero=contrast_scores(gt,np.zeros_like(gt),scale,axes)
    assert zero['gain_vs_zero']==0 and zero['signed_agreement']==0
    reversed=contrast_scores(gt,-gt,scale,axes)
    assert reversed['gain_vs_zero']==-3 and reversed['correlation']<-.99


def test_oracle_ratio_uses_current_matched_denominator_and_reports_invalid_draws():
    groups=np.repeat(np.arange(4),3); h=np.ones(12); gt=h*.5; pred=h*.75
    result=oracle_retention(h,gt,pred,groups,11)
    assert result['R']==.5 and result['valid_bootstrap_fraction']==1
    result=oracle_retention(h,h*2,pred,groups,11)
    assert result['R'] is None and result['valid_bootstrap_fraction']==0
