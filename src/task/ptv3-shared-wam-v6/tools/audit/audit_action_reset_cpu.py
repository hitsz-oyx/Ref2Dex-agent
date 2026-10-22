"""CPU replay of real native PD source and real Env reset logic, with explicit limits."""
import ast,copy,json,pickle,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/wam-rl-policy-v3/src'),str(ROOT/'src/task/trajectory-policy/src')]
from wam_rl_policy_v3.inspire_contract import ACTIVE,expand_action
from trajectory_policy.pd_retargeter import pd_from_action
from ptv3_shared_wam_v6.negative_result_invariants import verified


def native_pd_function():
    source=ROOT/'third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py'
    tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Dexplore_Inspire')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_action_to_pd_targets')
    namespace={};exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
    return namespace[method.name],dict(path=str(source),line=method.lineno,extraction='exact AST function, no edits; no Isaac import or simulator')


def action_audit():
    panel=json.loads((ROOT/'outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/panel.json').read_text())
    first=panel['records'][0];receipt=json.loads(Path(first['collection_batch_manifest']).read_text());ctrl=receipt['native_controller']
    off=torch.tensor(ctrl['offset']);scale=torch.tensor(ctrl['scale']);fn,evidence=native_pd_function()
    with np.load(Path(first['source'])/'trajectory.npz') as z:q=torch.tensor(z['q'][0:1])
    instance=SimpleNamespace(_pd_action_offset=off,_pd_action_scale=scale,_dof_pos=q)
    base=np.zeros((1,12),np.float32);base_pd=fn(instance,torch.tensor(expand_action(base))).numpy();axes=[];chains=[]
    for j,dim in enumerate(ACTIVE):
        for sign in [-1,1]:
            raw=base.copy();raw[0,j]=sign*.001;clipped=np.clip(raw,-1,1);expanded=expand_action(clipped)
            got=fn(instance,torch.tensor(expanded)).numpy();expected=pd_from_action(torch.tensor(expanded),q,off,scale).numpy()
            assert np.max(np.abs(got-expected))<1e-6
            delta=float(got[0,dim]-base_pd[0,dim]);wanted=sign*.001*float(scale[dim])*(1 if dim<6 else .5)
            assert np.isclose(delta,wanted,atol=2e-7)
            chains.append(dict(active_axis=j,native_dof=dim,raw=raw.tolist(),clipped=clipped.tolist(),canonical=clipped.tolist(),expanded=expanded.tolist(),pd_target=got.tolist(),target_delta=delta))
        axes.append(dict(active_axis=j,native_dof=dim,positive_negative_target_effect=True))
    maxq=np.zeros(12);maxdq=np.zeros(12);errors=[];transitions=0
    for row in panel['records']:
        receipt=json.loads(Path(row['collection_batch_manifest']).read_text());c=receipt['native_controller'];o=torch.tensor(c['offset']);s=torch.tensor(c['scale'])
        with np.load(Path(row['source'])/'trajectory.npz') as z:
            n=min(row['reference_controls'],len(z['action']));pred=pd_from_action(torch.tensor(z['applied'][:n]),torch.tensor(z['q'][:n]),o,s).numpy()
            errors.append(float(np.max(np.abs(pred-z['pd_targets'][:n]))));maxq=np.maximum(maxq,np.max(np.abs(np.diff(z['q'][:n+1,list(ACTIVE)],axis=0)),axis=0));maxdq=np.maximum(maxdq,np.max(np.abs(z['dq'][1:n+1,list(ACTIVE)]),axis=0));transitions+=n
    report=dict(native_source=evidence,axes=axes,perturbation_chains=chains,archive_sequences=16,archive_transitions=transitions,
                recorded_pd_reconstruction_max_error=max(errors),measured_max_abs_q_change=maxq.tolist(),measured_max_abs_dq=maxdq.tolist(),
                controlled_perturbation_measured_q_dq='NOT_RUN: no new physics. Archived q/dq are observational, not same-state counterfactual responses.',
                conclusion='12/12 action axes reach native PD target; archived actual controls have measured motion. Closed-loop efficacy and controlled perturbation physics are not certified.')
    check=verified('raw clipped expanded native PD target axis effect',np.array([a['positive_negative_target_effect'] for a in axes]),np.ones(12,bool),np.zeros(12,bool),'disconnect all active coordinates before native PD conversion')
    return report,check


def reset_replay(mutate=False):
    from wam_rl_policy_v3.measured_official_env import MeasuredOfficialEnv
    from wam_rl_policy_v3.wam_v4_runtime import load_geometry,V4Runtime
    base=ROOT/'outputs/ptv3-shared-wam-v6/samples-efficiency-frozen-smoke-20261022-r2/A-seed191-eval0/policy/initial-state.npz'
    with np.load(base) as z:native={k:z[k][:2].copy() for k in z.files}
    native['episodes']=native['episodes'].tolist();native['sequences']=native['sequences'].tolist()
    initial=copy.deepcopy(native);records=[];pending={}
    class Model:
        def shape(self,p):return torch.zeros(len(p),64)
    class Runtime:
        device=torch.device('cpu');config=dict(voxel_origin_m=[-2,-2,-2],voxel_m=.005,hand_in_ptv3=True)
        def predict_hand(self,batch,**kwargs):return dict(trajectory=torch.zeros(len(batch['start_hand']),24,2,11,3))
        workspace_valid=V4Runtime.workspace_valid
    rt=Runtime();rt.env_objects=[dict(object_code='dexplore/'+s.split('_')[1],scale=1.) for s in native['sequences']]
    rt.geometry=load_geometry(ROOT/'outputs/wam-rl-policy-v2/inspire-shadow-parity-20261019-r1/cache',rt.env_objects)
    env=MeasuredOfficialEnv(None,dict(state=native),ROOT/'outputs/wam-rl-policy-v3/inspire-task-prefix-semantics-20261022-r1/manifest.json',rt,Model(),reset_probe=True)
    def fingerprint(i):
        return pickle.dumps((env.controllers[i],env.trackers[i],list(env.histories[i]),list(env.hands[i]),env.previous[i].copy(),env.observations[i].numpy().copy()),protocol=4)
    original_refresh=env._refresh
    def refresh(indices,reset=False):
        original_refresh(indices,reset=reset)
        if reset and pending:
            ids=list(indices);untouched=[i for i in range(2) if i not in ids]
            if mutate:
                for i in untouched:env.histories[i].clear()
            records.append(dict(control_step=env.steps,reset_ids=ids,
                untouched_logical_state_equal=all(fingerprint(i)==pending['logical'][i] for i in untouched),
                untouched_native_fields_equal=all(np.array_equal(np.asarray(env.native[k])[untouched],np.asarray(pending['before'][k])[untouched]) for k in initial),
                reset_history_one=all(len(env.histories[i])==1 and len(env.hands[i])==1 for i in ids),
                reset_previous_zero=all(not env.previous[i].any() for i in ids),
                reset_task_cursor=all(env.controllers[i].scheduler.event_index==0 and env.controllers[i].scheduler.confirmed_progress==0 for i in ids),
                reset_reward_unpaid=all(not env.trackers[i].reward.paid and env.trackers[i].steps==0 for i in ids),
                reset_force_invalid=all(not env.native['contact_force_valid'][i] and not env.native['dof_force_valid'][i] for i in ids),
                reset_tau_invalid=all(env.observations[i,685].item()==0 for i in ids),episodes=list(env.native['episodes']),ticks=np.asarray(env.native['tick']).tolist()))
            pending.clear()
    env._refresh=refresh
    def rpc(op,**kw):
        current=copy.deepcopy(env.native)
        if op=='step':
            assert list(kw['episodes'])==list(current['episodes'])
            current['tick']=np.asarray(current['tick'])+1
            current['q']=np.asarray(current['q'])+.00001;current['dq']=np.full_like(current['dq'],.0003)
            current['contact_force_valid']=np.ones(2,bool);current['dof_force_valid']=np.ones(2,bool)
        elif op=='reset':
            ids=list(kw['ids']);pending.update(before=copy.deepcopy(current),logical={i:fingerprint(i) for i in range(2) if i not in ids})
            for k in initial:
                if k=='episodes':
                    for i in ids:current[k][i]+=1
                elif k=='sequences':pass
                else:current[k]=np.asarray(current[k]).copy();current[k][ids]=np.asarray(initial[k])[ids]
            for k in ['contact_force_valid','dof_force_valid']:current[k][ids]=False
            for k in ['contact_forces','dof_force','object_contact_force']:current[k][ids]=0
            current['tick'][ids]=0
        else:raise ValueError(op)
        return dict(state=current)
    env.rpc=rpc
    for tick in range(1,6):
        for tracker in env.trackers:tracker.reward.paid.add('grasp_acquired')
        env.step(torch.full((2,12),.1))
    passed=bool([r['control_step'] for r in records]==[2,4] and all(all(v for k,v in r.items() if k.startswith(('untouched_','reset_')) and k!='reset_ids') for r in records))
    return dict(status='PASS' if passed else 'FAIL',records=records,scope='real MeasuredOfficialEnv/TaskController/reward/reset code with CPU replay transport; no new physics and no hidden PhysX state certificate')
