#!/usr/bin/env python3
"""Matched contact geometry versus nuisance, with bounded closed-form fits."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/cm-interaction-oracle'
sys.path[:0]=[str(ROOT),str(TASK/'src'),str(TASK/'tools/run'),str(TASK/'tools/audit')]
from contact_innovation import contact_inputs,action_normalize_fit,bounded_design
from probe_relative_finger_innovation import prepare_inputs,nuisance_replay
from probe_spatial_action_fidelity import load_inputs
from geometric_consequence import physics_baseline,standardize_fit,normalize,ridge_fit,ridge_predict
from probe_geometric_consequence import cpu_tree
from probe_interventions import sha
from probe_duration_response import current_design
from conditional_consequence import permute_within,contrast_scores
from consequence_sufficiency import cluster_gain
from intervention import physical_targets

KINDS=('State','Joint','Finger','Contact','ContactShuffled')
MODES=('Physics','OOFState')


def build_inputs(p,bridge,g,h,source,forecast,device):
    common,old_action,extra_norm=prepare_inputs(p,bridge,g,h,source,forecast,device)
    contact_state,contact_action=contact_inputs(p,bridge,g,forecast['oof_q']['Ha'].to(device))
    train=torch.as_tensor(source['train'],device=device)
    cn=standardize_fit(contact_state,train)
    common=torch.cat((common,normalize(contact_state,cn)),-1).clamp(-8,8)
    actions=dict(Joint=old_action['PredJoint'],Finger=old_action['PredFinger'],Contact=contact_action,
                 State=torch.zeros_like(contact_action))
    norms={name:action_normalize_fit(value,train) for name,value in actions.items()}
    return common,actions,dict(extra=extra_norm,contact=cn,actions=norms),contact_state


def evaluate_candidates(common,action,norm,state):
    n,k,_=action.shape
    x=bounded_design(common[:,None].expand(-1,k,-1).reshape(n*k,-1),action.flatten(0,1),norm)
    return ridge_predict(x,state).reshape(n,k,26)


def summarize(source,predictions,candidates,scale,test):
    errors={name:((pred[test]-source['z'].numpy()[test])/scale)**2 for name,pred in predictions.items()}
    metrics={name:dict(E_mse=float(err[:,:12].mean()),I_mse=float(err[:,12:].mean())) for name,err in errors.items()}
    pairs=[]
    for mode in MODES:
        pairs += [(mode+'_'+kind,mode+'_State') for kind in ('Joint','Finger','Contact')]
        pairs += [(mode+'_Contact',mode+'_'+kind) for kind in ('Joint','Finger','ContactShuffled')]
        pairs += [(mode+'_Contact_test_shuffled',mode+'_Contact'),(mode+'_Contact','FrozenState')]
    pairs += [('Physics_'+kind,'OOFState_'+kind) for kind in KINDS]
    comparisons={new+'_vs_'+control:{key:cluster_gain(errors[control][:,axes].mean(1),errors[new][:,axes].mean(1),source['clusters'][test],253)
        for key,axes in (('E',np.arange(12)),('I',np.arange(12,26)))} for new,control in pairs}
    contrasts={}
    for name,value in candidates.items():
        effect=(value[:,1:]-value[:,:1]).mean(0);beta=source['GT_contrasts']
        contrasts[name]={key:dict(raw=contrast_scores(beta,effect,scale,axes),
            centered=contrast_scores(beta-beta.mean(0),effect-effect.mean(0),scale,axes),
            plus_minus=contrast_scores(beta[::2]-beta[1::2],effect[::2]-effect[1::2],scale,axes))
            for key,axes in (('E',np.arange(12)),('I',np.arange(12,26)))}
    name='Physics_Contact';gain=comparisons[name+'_vs_FrozenState']['I']
    A=gain['gain']>=.05 and gain['lower95']>0
    specific=all(comparisons[name+'_vs_Physics_'+kind]['I']['gain']>=.03 and comparisons[name+'_vs_Physics_'+kind]['I']['lower95']>0 for kind in ('State','Joint','Finger','ContactShuffled'))
    shuffle=comparisons['Physics_Contact_test_shuffled_vs_'+name]['I']
    sensitivity=-shuffle['gain']>=.03 and shuffle['upper95']<0
    c=contrasts[name]['I'];B=c['raw']['correlation'] is not None and c['raw']['correlation']>=.5 and c['raw']['signed_agreement']>=.65 and c['raw']['gain_vs_zero']>=.1 and c['centered']['gain_vs_zero']>0
    gates=dict(A=A,B=B,contact_specific=specific,test_shuffle=sensitivity)
    return dict(status='PROMISING' if all(gates.values()) else 'UNPROMISING',gates=gates,metrics=metrics,comparisons=comparisons,contrasts=contrasts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--forecast-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();args.run_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(2)
    def save(name,value):(args.run_dir/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    manifest=dict(run_status='STARTED',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        created_at=datetime.now(timezone.utc).isoformat(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),seeds=[253],smoke=args.smoke,
        command=sys.argv,dataset_sha256=sha(args.dataset),forecast_manifest_sha256=sha(args.forecast_run/'manifest.json'),
        forecast_diagnostic_sha256=sha(args.forecast_run/'diagnostic.pt'),alpha=32.,new_ridge_fits=10,new_neural_fits=0,
        simulation_executed=False,task_or_policy_executed=False)
    paths=[Path(__file__).resolve(),TASK/'src/contact_innovation.py',TASK/'tools/run/probe_relative_finger_innovation.py',
           TASK/'tools/run/probe_spatial_action_fidelity.py',TASK/'src/geometric_consequence.py',TASK/'src/conditional_consequence.py',
           TASK/'src/consequence_sufficiency.py',TASK/'tools/audit/probe_duration_response.py',TASK/'src/intervention.py']
    manifest['code_sha256']={str(path.relative_to(ROOT)):sha(path) for path in paths};save('manifest.json',manifest)
    try:
        dev=torch.device('cuda:0');torch.cuda.set_device(dev);torch.cuda.reset_peak_memory_stats(dev)
        p,bridge,g,h,source,forecast,source_run,original_run=load_inputs(args,dev)
        manifest.update(source_run=str(source_run),original_run=str(original_run),source_diagnostic_sha256=sha(source_run/'diagnostic.pt'),
                        original_diagnostic_sha256=sha(original_run/'diagnostic.pt'))
        assert torch.equal(physical_targets(p['before'],p['trajectory'][:,7]),source['z'])
        replay=nuisance_replay(p,g,source,dev)
        common,actions,norms,contact_state=build_inputs(p,bridge,g,h,source,forecast,dev)
        train,test=source['train'],source['test'];tr,te=(torch.as_tensor(v,device=dev) for v in (train,test))
        arms=torch.as_tensor(source['arms'],device=dev);rows=torch.arange(len(arms),device=dev)
        _,groups=current_design(p,p['episode_id'].numpy()//168)
        permutation=np.arange(len(arms))
        for ids in (train,test):permutation[ids]=permute_within(groups,ids,253)[ids]
        shuffled=arms[torch.as_tensor(permutation,device=dev)]
        mu=dict(Physics=physics_baseline(p['before'].to(dev),8/30),OOFState=source['oof_predictions']['State'].to(dev))
        z=source['z'].to(dev);scale=source['target_scale'].to(dev)
        states,predictions,candidates,records={},{},{},{}
        for mode in MODES:
            for kind in KINDS:
                name=mode+'_'+kind;key='Contact' if kind=='ContactShuffled' else kind
                arm=shuffled if kind=='ContactShuffled' else arms
                x=bounded_design(common,actions[key][rows,arm],norms['actions'][key])
                target=(z-mu[mode])/scale;state=ridge_fit(x,target,tr)
                value=evaluate_candidates(common,actions[key],norms['actions'][key],state)
                prediction=mu[mode]+value[rows,arm]*scale
                assert torch.isfinite(value).all()
                states[name]=cpu_tree(state);predictions[name]=prediction.cpu().numpy()
                candidates[name]=(mu[mode][te,None]+value[te]*scale).cpu().numpy()
                centered=x[tr].double()-state['x_mean'];residual=target[tr].double()-state['y_mean']
                normal_error=(centered.T@(centered@state['weight']-residual)+32*state['weight']).abs().max()/(centered.T@residual).abs().max().clamp_min(1)
                records[name]=dict(source_normalized_mse=float(((prediction[tr]-z[tr])/scale).square().mean()),
                                   relative_normal_equation_error=float(normal_error),feature_width=x.shape[1])
                if kind=='State':assert float((value-value[:,:1]).abs().max())==0
                if kind=='Contact':predictions[name+'_test_shuffled']=(mu[mode]+value[rows,shuffled]*scale).cpu().numpy()
        predictions['FrozenState']=source['full_predictions']['State']
        result=summarize(source,predictions,candidates,scale.cpu().numpy(),test)
        if args.smoke:result.update(status='UNCLEAR',engineering_only=True)
        else:result['engineering_only']=False
        result.update(nuisance_replay=replay,elapsed_seconds=time.monotonic()-started,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(dev),
                      scope='Sampled-surface local/current-wrist proxies; exposed full-test exploratory screen; no nested consequence OOF or paired counterfactual/GT independence or policy utility.')
        save('result.json',result);save('fit_records.json',records)
        torch.save(dict(train=train,test=test,clusters=source['clusters'],arms=source['arms'],common=common.cpu(),actions=cpu_tree(actions),
                        norms=cpu_tree(norms),contact_state=contact_state.cpu(),mu=cpu_tree(mu),states=states,predictions=predictions,
                        candidates=candidates,permutation=permutation),args.run_dir/'diagnostic.pt')
        assert all(sha(ROOT/k)==v for k,v in manifest['code_sha256'].items())
        assert sha(args.dataset)==manifest['dataset_sha256'] and sha(args.forecast_run/'diagnostic.pt')==manifest['forecast_diagnostic_sha256']
        assert sum(path.stat().st_size for path in args.run_dir.rglob('*') if path.is_file())<40*1024**2
        manifest['run_status']='COMPLETED';print(json.dumps(dict(status=result['status'],gates=result['gates'],metrics=result['metrics'],elapsed=result['elapsed_seconds'])),flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',failure=f'{type(error).__name__}: {error}');raise
    finally:
        manifest.update(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started);save('manifest.json',manifest)


if __name__=='__main__':main()
