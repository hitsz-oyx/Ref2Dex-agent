"""A model representation boundary must not manufacture physical task failures."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'src/task/ptv3-shared-wam-v6/tools/audit'))


def test_later_voxel_violation_masks_tau_without_terminating_task(monkeypatch):
    from audit_action_reset_cpu import reset_replay
    from wam_rl_policy_v3.measured_official_env import MeasuredOfficialEnv
    from wam_rl_policy_v3.wam_v4_runtime import V4Runtime
    original_refresh=MeasuredOfficialEnv._refresh;original_gate=V4Runtime.workspace_valid;original_step=MeasuredOfficialEnv.step
    active=[];injected=[];observed=[]
    def refresh(env,*args,**kwargs):
        active[:]=[(env,kwargs.get('reset',False))];return original_refresh(env,*args,**kwargs)
    def gate(runtime,samples):
        if active and active[0][0].steps==2 and not active[0][1] and not injected:
            samples[0]['start_hand'][0,0,0]=-3.;injected.append(True)
        return original_gate(runtime,samples)
    def step(env,*args,**kwargs):
        result=original_step(env,*args,**kwargs)
        if env.steps==3:
            observed.append(dict(done=result[2].tolist(),valid=result[0][:,685].tolist(),env=env))
        return result
    monkeypatch.setattr(MeasuredOfficialEnv,'_refresh',refresh);monkeypatch.setattr(V4Runtime,'workspace_valid',gate);monkeypatch.setattr(MeasuredOfficialEnv,'step',step)
    report=reset_replay()
    assert injected and report['status']=='PASS'
    assert observed[0]['done']==[0,0] and observed[0]['valid'][0]==0
    assert observed[0]['env'].workspace_failures==0
    assert observed[0]['env'].model_domain_invalid_steps==1
    assert abs(observed[0]['env'].reward_components[0,-1])<1e-6  # float32 rounding only, no domain-failure cost


def test_initial_model_domain_failure_is_explicit_admission_error(monkeypatch):
    from audit_action_reset_cpu import reset_replay
    from wam_rl_policy_v3.wam_v4_runtime import V4Runtime
    monkeypatch.setattr(V4Runtime,'workspace_valid',lambda self,samples:np.zeros(len(samples),bool))
    with pytest.raises(ValueError,match='model-domain admission'):
        reset_replay()
