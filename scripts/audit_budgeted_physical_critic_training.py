"""Independent raw SDK, data-budget, final critic/physical/actor forward replay."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from src.task.CmResidual.option_model_policy import DYNAMIC,state_features,numpy_forward,initialized_network
from src.task.CmResidual.budgeted_physical_q import read_panel,numpy_head,initialized_critic


def independent_function():
    p=ROOT/'scripts/audit_option_feature_contract.py';source=p.read_text().replace('env=np.arange(768)','env=np.arange(len(initial[\'motion\']))').replace('np.ones((768,1),np.float32)','np.ones((len(initial[\'motion\']),1),np.float32)');ns={'__name__':'budget_sdk_independent','__file__':str(p)};exec(compile(source,str(p),'exec'),ns);return ns['independent_points']


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory.resolve();begin=time.monotonic();torch.set_num_threads(2)
    m=json.loads((root/'run_manifest.json').read_text());r=json.loads((root/'fit/results.json').read_text());assert r['run_status']=='COMPLETED' and sha(root/'fit/models.pt')==r['models_sha256'];model=torch.load(root/'fit/models.pt',map_location='cpu',weights_only=False);packet=torch.load(root/'fit/fit_rows.pt',map_location='cpu',weights_only=False);base=torch.load(m['base_checkpoint'],map_location='cpu',weights_only=False);independent=independent_function();data={};errors={}
    for seed in (651,652,653,654):
        s=str(seed);d=root/f's{seed}';data[s]=read_panel(d,base);initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(d/'trace.pt',map_location='cpu',weights_only=False);meta=json.loads((d/'physical_metadata.json').read_text());ticks=initial['decision_steps'].numpy()
        for k in data[s]:assert np.array_equal(data[s][k],packet['data'][s][k],equal_nan=True)
        for name,t in (('current',ticks),('future',ticks+8)):
            compact,extra=independent(initial,trace,meta,base,t);errors[s+'_'+name+'_compact']=float(np.abs(compact-data[s][name]).max());errors[s+'_'+name+'_sdk']=float(np.abs(extra-data[s]['extra' if name=='current' else 'future_extra']).max());assert errors[s+'_'+name+'_compact']<=2e-6 and errors[s+'_'+name+'_sdk']<=2e-5
    assert np.isnan(data['651']['reward']).all() and not data['651']['label_available'].any();assert not data['652']['label_available'].any() and np.isnan(data['652']['reward']).all() and data['653']['label_available'].all() and data['654']['label_available'].all()
    extra=np.concatenate([data['653'][k] for k in ('extra','future_extra')]);mean=extra.mean(0);std=extra.std(0).clip(.001);assert np.array_equal(mean,model['extra_mean']) and np.array_equal(std,model['extra_std'])
    current={s:state_features(d['current'],d['extra'],mean,std) for s,d in data.items()};future={s:state_features(d['future'],d['future_extra'],mean,std) for s,d in data.items()};px=np.concatenate([current[s] for s in ('651','652','653')]);pa=np.concatenate([data[s]['raw'] for s in ('651','652','653')]);delta=np.concatenate([future[s][:,DYNAMIC]-current[s][:,DYNAMIC] for s in ('651','652','653')]);assert np.array_equal(delta,packet['physical_delta']) and np.array_equal(px,packet['physical_current']) and np.array_equal(pa,packet['physical_raw']);assert np.array_equal(delta.mean(0),model['delta_mean']) and np.array_equal(delta.std(0).clip(.001),model['delta_std'])
    initial=initialized_critic().state_dict();assert all(torch.equal(initial[k],model['initial_parameters'][k]) for k in initial);assert model['critic_updates_each']==1500
    actor_files=[root/'fit/actors.pt',root/'fit/actors_budget.pt'];bundles={}
    for key,p in zip(('common','budget'),actor_files):assert sha(p)==r['actor_sha256'][key];bundles[key]=torch.load(p,map_location='cpu',weights_only=False)
    actor_initial=initialized_network('actor').state_dict()
    for bundle in bundles.values():
        assert all(torch.equal(actor_initial[k],bundle['actor_initial_parameters'][k]) for k in actor_initial)
        assert np.array_equal(bundle['extra_mean'],mean) and np.array_equal(bundle['extra_std'],std) and bundle['actor_updates_each']==1000 and bundle['source_policy_sha256']==m['policy_sha256']
    assert bundles['common']['control_name']=='cold_q' and bundles['budget']['control_name']=='budget_q'
    for name in ('cm','dynamics_off'):assert all(torch.equal(bundles['common']['actors'][name][k],bundles['budget']['actors'][name][k]) for k in bundles['common']['actors'][name])
    for name,state in model['critics'].items():
        sources=('653','654') if name=='budget_q' else ('653',);x=np.concatenate([current[s] for s in sources]);raw=np.concatenate([data[s]['raw'] for s in sources]);labels=np.concatenate([data[s]['reward'] for s in sources]);assert np.array_equal(x,packet['task_current'][name]) and np.array_equal(raw,packet['task_raw'][name]) and np.array_equal(labels,packet['task_labels'][name])
        pred=numpy_head(state,np.concatenate((x,np.tanh(raw.astype(np.float64))),-1),'task').flatten();errors[name+'_task']=float(np.abs(pred-packet['task_gpu_outputs'][name]).max());phyact=np.tanh(pa.astype(np.float64)) if name=='cm' else np.zeros_like(pa,np.float64);pred=numpy_head(state,np.concatenate((px,phyact),-1),'physical');errors[name+'_physical']=float(np.abs(pred-packet['physical_gpu_outputs'][name]).max())
        key='budget' if name=='budget_q' else 'common';actor_name='direct_q' if name in ('cold_q','budget_q') else name;pred=numpy_forward(bundles[key]['actors'][actor_name],x,'tanh');errors[name+'_actor']=float(np.abs(pred-packet['actor_gpu_outputs'][name]).max());assert max(errors[name+'_task'],errors[name+'_physical'],errors[name+'_actor'])<=2e-5
    assert 1536*101+768*202==1536*202==310272 and r['actual_joint_critic_optimizer_steps']==6000 and r['physical_auxiliary_updates_subset']==3000 and r['actual_actor_optimizer_steps']==4000
    result=dict(run_status='COMPLETED',all_raw_current_and_future_sdk_reconstructed=True,short_terminal_labels_absent=True,extra_data_excluded_from_common_pools_and_statistics=True,exact_measured_env_control_tick_budget_equal=True,common_initial_critic_parameters=True,both_evaluation_bundles_share_same_cm_off=True,all_final_task_physical_and_actor_forwards_replayed=True,independent_optimizer_replay=False,maximum_errors=errors,wall_seconds=time.monotonic()-begin)
    (root/'training_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
