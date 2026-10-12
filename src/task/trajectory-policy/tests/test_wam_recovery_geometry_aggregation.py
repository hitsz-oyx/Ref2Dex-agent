"""CPU contracts for explicitly retired recovery geometry training."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


PATH=Path(__file__).resolve().parents[1]/'tools/run/fit_wam_recovery_geometry_aggregation.py'
spec=importlib.util.spec_from_file_location('geometry_admission_test',PATH)
geometry=importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


def source(split):
    controls,takeover=(542,16) if split=='train' else (192,32)
    roles=['teacher','gt_tau_R','wam_R','wam_command_replay_then_expert_tick%d'%takeover]
    manifest=dict(status='COMPLETED',controls=controls,takeover_tick=takeover,roles=roles,
        dt=1/30,
        training_allowed=True,expert_PD_training_allowed=True,mixed_policy_future_hand_training_allowed=False,
        trajectory_sha256=geometry.SOURCE_SHA[split],native_controller=dict(offset=[0.]*18,scale=[1.]*18))
    audit=dict(pass_=True,all_arrays_finite=True,controls=controls,measured_states=controls+1,
        checks_max=0.,check_count=9768 if split=='train' else 3468,
        native_data_sha256=geometry.SOURCE_SHA[split]);audit['pass']=audit.pop('pass_')
    result=dict(status='COMPLETED',calibration_pass=split=='train')
    if split=='cross':result.update(partial_episode=True,calibration_pass=None,recovery_pass=None,local_grasp_calibration=True)
    d={k:np.zeros((controls+1,4,18),np.float32) for k in ('q','dq')}
    d['hand']=np.zeros((controls+1,4,11,3),np.float32)
    d['obj']=np.broadcast_to(np.eye(4,dtype=np.float32),(controls+1,4,4,4)).copy()
    d['velocity']=np.zeros((controls+1,4,6),np.float32)
    for k in ('applied','pd_targets'):d[k]=np.zeros((controls,4,18),np.float32)
    for k in ('expert_action','expert_raw','expert_PD'):d[k]=np.zeros((controls,2,18),np.float32)
    d['expert_PD'][:,:,6:]=.5
    for distal,parent,ratio in ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)):
        d['expert_PD'][:,:,distal]=d['expert_PD'][:,:,parent]*ratio
    d['pd_targets'][:,3]=d['expert_PD'][:,1]
    d['expert_clipped']=np.zeros((controls,2),bool)
    d['expert_applied']=np.zeros((controls,2),bool);d['expert_applied'][takeover:,1]=True
    d['learner_hand_history']=np.zeros((controls,2,4,11,3),np.float32)
    d['learner_object_history']=np.broadcast_to(np.eye(4,dtype=np.float32),(controls,2,4,4,4)).copy()
    d['learner_goal']=np.broadcast_to(np.eye(4,dtype=np.float32),(controls,2,4,4)).copy()
    return manifest,audit,result,d


@pytest.mark.parametrize('split,windows,last_control',[('train',80,118),('cross',137,191)])
def test_scoped_admission_does_not_change_source_blanket_deny(split,windows,last_control):
    m,a,r,d=source(split);m['actor_future_hand_training_allowed']=False
    admission=geometry.own_expert_admission(m,a,r,d,geometry.SOURCE_SHA[split],split)
    assert m['actor_future_hand_training_allowed'] is False
    assert admission['actor_expert_future_training_allowed'] is True
    assert admission['diagnostic_only'] is False
    assert admission['prior_cross_diagnostic_retired_to_training']==(split=='cross')
    assert admission['windows']==windows and admission['future_causing_control_last']==last_control
    assert admission['mixed_policy_future_hand_training_allowed'] is False


@pytest.mark.parametrize('defect',['last_control_nonexpert','last_control_clipped','action','q_PD','H4','nan','audit_tick','audit_checks','dt','source_actor_allowed'])
def test_invalid_future_or_input_rejected(defect):
    m,a,r,d=source('train')
    if defect=='last_control_nonexpert':d['expert_applied'][118,1]=False
    elif defect=='last_control_clipped':d['expert_clipped'][118,1]=True
    elif defect=='action':d['applied'][16,3,0]=.1
    elif defect=='q_PD':d['q'][16,3,0]=.1
    elif defect=='H4':d['learner_hand_history'][16,1,0,0,0]=.1
    elif defect=='nan':d['hand'][119,3,0,0]=np.nan
    elif defect=='audit_tick':a['takeover_tick']=32
    elif defect=='audit_checks':a['check_count']=0
    elif defect=='dt':m['dt']=1/60
    else:m['actor_future_hand_training_allowed']=True
    with pytest.raises(ValueError):geometry.own_expert_admission(m,a,r,d,geometry.SOURCE_SHA['train'],'train')


def test_current_object_frame_native_wam_roundtrip_for_future_displacement():
    from trajectory_policy.wam_shadow import NATIVE_TO_WAM,WAM_TO_NATIVE,robot_goal_sample,active_hand_tau
    from trajectory_policy.retargeter_data import features
    rotation=np.asarray([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],np.float32)
    obj=np.eye(4,dtype=np.float32);obj[:3,:3]=rotation;obj[:3,3]=[.2,.1,.7]
    hand=np.arange(33,dtype=np.float32).reshape(11,3)/100
    future=hand[None]+np.arange(1,25,dtype=np.float32)[:,None,None]*np.asarray([.001,.002,.003],np.float32)
    _,tau=features(np.zeros(18),np.zeros(18),hand,obj,np.zeros(6),future)
    truth=(future-hand)@rotation
    assert np.allclose(tau.reshape(24,11,3),truth)
    full=np.zeros((24,2,11,3),np.float32);full[:,0]=truth[:,NATIVE_TO_WAM]
    mask=np.zeros((2,11),bool);mask[0]=True
    assert np.allclose(active_hand_tau(full,mask),tau)
    canonical=dict(points=np.zeros((512,3),np.float32),normals=np.zeros((512,3),np.float32),radius=.1,center=np.zeros(3))
    sample=robot_goal_sample(np.repeat(obj[None],4,0),np.repeat(hand[None],4,0),obj,canonical)
    assert sample['trajectory_valid'].sum()==0
    assert not sample['start_hand_valid'][1].any()
    assert np.allclose(sample['start_hand'][0],((hand-obj[:3,3])@rotation)[NATIVE_TO_WAM])


def test_balanced_sampler_uses_exactly32_per_source_and_no_pure_or_nominal():
    old=np.arange(100,180);retired=np.arange(180,317)
    rng=np.random.default_rng(296)
    for _ in range(500):
        draws=geometry.balanced_recovery_draws(rng,old,retired)
        assert len(draws)==64
        assert np.isin(draws,old).sum()==32
        assert np.isin(draws,retired).sum()==32
        assert np.isin(draws,np.r_[old,retired]).all()


@pytest.mark.parametrize('old,retired',[([], [1]),([1],[]),([1,2],[2,3])])
def test_sampler_rejects_missing_or_overlapping_source_pools(old,retired):
    with pytest.raises(ValueError):geometry.balanced_recovery_draws(np.random.default_rng(296),old,retired)
