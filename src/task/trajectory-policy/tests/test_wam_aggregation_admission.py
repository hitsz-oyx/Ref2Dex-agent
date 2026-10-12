"""Reject old episodes or detached audits before aggregation can use labels."""
import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'tools/run/fit_wam_robot_aggregation.py'
spec=importlib.util.spec_from_file_location('aggregation_admission',path)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

@pytest.fixture
def source():
    digest='a'*64
    m=dict(status='COMPLETED',controls=192,takeover_tick=32,
        roles=['teacher','gt_tau_R','wam_R','wam_command_replay_then_expert_tick32'],
        training_allowed=True,expert_PD_training_allowed=True,
        mixed_policy_future_hand_training_allowed=False,trajectory_sha256=digest)
    a=dict(pass_=True,controls=192,takeover_tick=32,measured_states=193,native_data_sha256=digest)
    a['pass']=a.pop('pass_')
    r=dict(status='COMPLETED',partial_episode=True,calibration_pass=None,recovery_pass=None,
        local_grasp_calibration=True,outcomes=[{}, {}, {},
        dict(maximum_held_frames=60,first_stable_tick=120,clipping_rate=0.)])
    return m,a,r,digest

def test_new_later_takeover_source_admitted(source):
    assert module.admit_aggregation_source(*source)

@pytest.mark.parametrize('defect',['old_episode','audit_hash','missing_state','teacher_failed','full_verdict','clipped','replay_role'])
def test_wrong_recovery_or_detached_evidence_rejected(source,defect):
    m,a,r,d=source
    if defect=='old_episode': m.update(controls=542,takeover_tick=16)
    elif defect=='audit_hash': a['native_data_sha256']='b'*64
    elif defect=='missing_state': a['measured_states']=192
    elif defect=='teacher_failed': r['local_grasp_calibration']=False
    elif defect=='full_verdict': r['recovery_pass']=True
    elif defect=='clipped': r['outcomes'][3]['clipping_rate']=.1
    else: m['roles'][3]='wam_command_replay_then_expert_tick16'
    with pytest.raises(ValueError): module.admit_aggregation_source(m,a,r,d)
