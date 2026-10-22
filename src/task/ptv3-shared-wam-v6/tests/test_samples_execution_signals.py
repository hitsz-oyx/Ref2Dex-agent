"""Exercise the actual three-arm step/reset boundary with saved physical state."""
import copy,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'src/task/wam-rl-policy-v3/src'),str(ROOT/'src/task/trajectory-policy/src')]
spec=importlib.util.spec_from_file_location('samples_execution_env',ROOT/'src/task/ptv3-shared-wam-v6/tools/run/samples_efficiency_env.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def make_env(tmp_path,arm='C',task_mode='synthetic_hold'):
    path=ROOT/'outputs/ptv3-shared-wam-v6/samples-efficiency-frozen-smoke-20261022-r2/A-seed191-eval0/policy/initial-state.npz'
    with np.load(path) as z:state={k:z[k][:2].copy() for k in z.files}
    state['episodes']=state['episodes'].tolist();state['sequences']=state['sequences'].tolist()
    source=ROOT/'outputs/wam-rl-policy-v3/inspire-task-prefix-semantics-20261022-r1/manifest.json'
    audit=json.loads(source.read_text());audit['records']=[dict(r,sidecar=str(source.parent/r['sidecar'])) for r in audit['records'] if r['sequence'] in state['sequences']]
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(audit))
    class Forecast:
        def predict(self,objects,hands,goals,events,codes):
            self.calls.append(dict(batch=len(objects),events=events,goals=goals.copy()))
            return np.full((len(objects),24,11,3),.012,np.float32)
        calls=[]
    forecast=Forecast();forecast.calls=[]
    model=SimpleNamespace(shape=lambda points:torch.zeros(len(points),64))
    env=module.SampleEfficiencyEnv(None,dict(state=state),manifest,SimpleNamespace(device=torch.device('cpu')),model,
        arm=arm,forecast=forecast,geometry_cache=ROOT/'outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/geometry',task_mode=task_mode)
    initial=copy.deepcopy(state)
    def rpc(op,**kwargs):
        state=copy.deepcopy(env.native)
        if op=='step':state['tick']=np.asarray(state['tick'])+1
        elif op=='reset':
            for i in kwargs['ids']:
                for key in state:
                    if key=='episodes':state[key][i]+=1
                    else:state[key][i]=copy.deepcopy(initial[key][i])
        else:raise ValueError(op)
        return dict(state=state)
    env.rpc=rpc
    return env,forecast


@pytest.mark.parametrize('arm',['A','B','C'])
def test_encoder_failure_masks_tau_and_preserves_physical_episode(tmp_path,monkeypatch,arm):
    env,forecast=make_env(tmp_path,arm)
    # Mutate only model-domain availability, leaving all physical evidence intact.
    monkeypatch.setattr(module,'encoder_domain_available',lambda *args,**kwargs:False)
    for _ in range(4):obs,reward,done,info=env.step(torch.zeros(2,12))
    assert done.tolist()==[0,0]
    assert not obs[:,421:686].any() and not forecast.calls
    assert env.encoder_unavailable_steps==8 and env.physical_safety_aborts==0
    assert info['encoder_available'].tolist()==[False,False]
    assert not info['physical_safety_abort'].any()


def test_success_survives_same_tick_encoder_failure(tmp_path,monkeypatch):
    from dataclasses import replace
    env,forecast=make_env(tmp_path)
    monkeypatch.setattr(module,'encoder_domain_available',lambda *args,**kwargs:False)
    refresh=env._refresh
    def inject(*args,**kwargs):
        refresh(*args,**kwargs)
        if not kwargs.get('reset',False):
            decision,features=env.latest[0];env.latest[0]=(replace(decision,task_complete=True,terminated=True,path_complete=True),features)
    monkeypatch.setattr(env,'_refresh',inject)
    _,_,done,info=env.step(torch.zeros(2,12))
    result=info['episode_results'][0]
    assert done[0] and result['ever_success'] and result['final_success'] and result['task_success']
    assert info['task_success'][0] and info['ever_success'][0] and info['final_success'][0]
    assert result['task_contract'].startswith('synthetic-grasp-lift') and result['source_full_task_success'] is None
    assert env.records[env.native['sequences'][0]][0].source_role=='same_execution_teacher_object_rollout'
    assert env.targets[0].source_role=='explicit_task_instruction'


@pytest.mark.parametrize('encoder_ok',[True,False])
def test_real_table_penetration_aborts_even_with_encodable_input(tmp_path,monkeypatch,encoder_ok):
    env,_=make_env(tmp_path)
    monkeypatch.setattr(module,'encoder_domain_available',lambda *args,**kwargs:encoder_ok)
    rpc=env.rpc
    def lower(op,**kwargs):
        reply=rpc(op,**kwargs)
        if op=='step':reply['state']['objects'][0,2]-=.10
        return reply
    monkeypatch.setattr(env,'rpc',lower)
    _,_,done,info=env.step(torch.zeros(2,12))
    assert done[0] and info['physical_safety_abort'][0]
    assert info['episode_results'][0]['failure_reason']=='physical_table_penetration'
    assert not info['episode_results'][0]['final_success']


def test_narrow_witness_uses_elapsed_seconds_and_keeps_ever_after_loss():
    from dataclasses import replace
    from wam_rl_policy_v3.task_feedback import InteractionFeedback
    from ptv3_shared_wam_v6.execution_contract import LiftHoldWitness
    witness=LiftHoldWitness(0.)
    feedback=InteractionFeedback(grasp_established=True,force_valid=True,supported=False)
    force=np.ones((5,3))
    for tick in range(30):assert not witness.update(.06,tick/30,feedback,force)
    assert witness.update(.06,1.,feedback,force)
    assert not witness.update(.06,31/30,replace(feedback,grasp_lost=True),force)
    assert witness.ever_success and not witness.final_success


def test_recoverable_encoder_error_continues_with_masked_state_only_mode(tmp_path):
    from ptv3_shared_wam_v6.online_runtime import EncodingUnavailable
    env,forecast=make_env(tmp_path)
    def fail(*args,**kwargs):raise EncodingUnavailable('deliberately injected nonfinite network output')
    forecast.predict=fail
    for _ in range(4):obs,_,done,info=env.step(torch.zeros(2,12))
    assert not done.any() and not obs[:,421:686].any()
    assert env.encoder_error_count>0 and env.physical_safety_aborts==0


def test_stage_and_reward_outputs_belong_to_terminal_episode_before_reset(tmp_path):
    env,_=make_env(tmp_path)
    env.reset_probe=True
    env.step(torch.zeros(2,12))
    _,reward,done,info=env.step(torch.zeros(2,12))
    assert done[0]
    result=info['episode_results'][0]
    assert result['stages']==info['stages'][0]
    assert info['reward_parts'][0]['sparse']==-1.
    assert float(reward[0])==pytest.approx(sum(info['reward_parts'][0].values()))
    assert result['reward_parts_sum']['sparse']==-1.
    assert env.stages[0].last_time==0. and info['stages'][0]['contract'].startswith('v6.approach')


def test_source_task_mode_preserves_declaration_and_cannot_label_reset_success(tmp_path):
    env,_=make_env(tmp_path,task_mode='source_declared')
    for i,name in enumerate(env.native['sequences']):
        assert env.targets[i].packet()==env.records[name][0].packet()
    env.reset_probe=True
    env.step(torch.zeros(2,12))
    _,_,_,info=env.step(torch.zeros(2,12))
    result=info['episode_results'][0]
    assert result['source_full_task_success'] is False
    assert result['task_contract'].startswith('source-declared-full-task')
