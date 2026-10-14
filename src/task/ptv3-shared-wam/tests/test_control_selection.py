import pytest
import torch
from types import SimpleNamespace

from ptv3_shared_wam.robot import robot_loss
from ptv3_shared_wam.selection import SelectionTracker, selection_keys
from ptv3_shared_wam.checkpoint import checkpoint_alias, prune_recovery_checkpoints
from test_shared_contract import batch


def report(hand, scene, xyz):
    return dict(metrics={'source3/hand_all24':dict(mean_mm=hand,count=100),
        'source3/scene_all24':dict(mean_mm=scene,count=200),
        **{k:dict(p95=xyz) for k in ['GT/PD_xyz_mm','pred/PD_xyz_mm','GT/PD_rotation_rad','GT/PD_finger_rad']}})


def test_control_objective_emphasizes_first_and_keeps_out_of_range_gradient():
    b=batch();b['native_action']=torch.zeros(2,8,12)
    action=torch.full((2,8,12),2.,requires_grad=True)
    prediction=dict(trajectory=b['trajectory_target'],scene=b['scene_target'],action=action,action_anchor=action)
    config=dict(action_objective='control-aware-first.v1',native_PD_scale=[1.]*12)
    _,terms=robot_loss(prediction,b,config,SimpleNamespace(action_scale=torch.ones(8,12)))
    gradient,=torch.autograd.grad(terms['action'],action)
    assert gradient[0,0,0]>28*gradient[0,1,0]>0
    assert 'action_first_PD' in terms
    _,legacy=robot_loss(prediction,b,{},SimpleNamespace(action_scale=torch.ones(8,12)))
    assert legacy['action']==2
    with pytest.raises(ValueError,match='unknown robot'):
        robot_loss(prediction,b,dict(action_objective='typo'),SimpleNamespace(action_scale=torch.ones(8,12)))


def test_control_and_forecast_selection_save_distinct_steps_and_ignore_missing_history(tmp_path):
    (tmp_path/'step1.pt').write_text('old')
    old=dict(step=1,**report(10,20,60));missing=dict(step=2,**report(1,1,1))
    tracker=SelectionTracker(tmp_path,'robot',[old,missing])
    assert tracker.missing_steps==[2]
    assert tracker.update(report(11,21,50),3)==['control_best']
    assert tracker.best['forecast_best']['step']==1
    assert tracker.best['control_best']['step']==3
    assert tracker.update(report(9,19,55),4)==['forecast_best']
    assert selection_keys({},'pretrain')=={}


def test_selected_weights_survive_retention(tmp_path):
    names=['step1.pt','step2.pt','step3.pt','step4.pt','step5.pt']
    for name in names:(tmp_path/name).write_text('owned')
    for alias,step in [('best.pt',1),('forecast_best.pt',2),('control_best.pt',3),('latest.pt',5)]:
        checkpoint_alias(tmp_path,alias,'step%d.pt'%step)
    kept=prune_recovery_checkpoints(tmp_path,names,dict(updates=10,checkpoint_keep_recent=1,checkpoint_milestone_interval=10))
    assert set(kept)=={'step1.pt','step2.pt','step3.pt','step5.pt'}
    assert all((tmp_path/alias).is_file() for alias in ['best.pt','forecast_best.pt','control_best.pt','latest.pt'])
