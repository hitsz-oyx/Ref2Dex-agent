#!/usr/bin/env python3
"""Full anatomical finger flow plus OOF-state/action-centered innovations."""
import argparse
from datetime import datetime, timezone
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
from relative_finger_consequence import finger_summary,fit_head
from spatial_action_fidelity import intrinsic_finger_flow
from probe_spatial_action_fidelity import load_inputs
from probe_execution_geometry import source_state
from geometric_consequence import standardize_fit,normalize
from spatial_consequence import SpatialConsequence,evaluate
from conditional_consequence import permute_within,contrast_scores
from consequence_sufficiency import cluster_gain
from probe_duration_response import current_design
from probe_geometric_consequence import cpu_tree
from probe_interventions import sha
from intervention import physical_targets

SPECS={
    'NomJointPlain':('NomJoint',False), 'NomJointCentered':('NomJoint',True),
    'NomFingerPlain':('NomFinger',False),'NomFingerCentered':('NomFinger',True),
    'PredJointCentered':('PredJoint',True),'PredFingerCentered':('PredFinger',True),
    'PredFingerShuffled':('PredFinger',True),
}


def nuisance_replay(p,g,source,device):
    n=len(source['arms']);rows=torch.arange(n,device=device);zero=torch.zeros(n,dtype=torch.long,device=device)
    action=torch.zeros(n,32,device=device);physical=torch.zeros(n,26,device=device)
    errors={};count=np.zeros(n,dtype=int)
    for prefix,norm in source['preprocess'].items():
        assert not set(source['clusters'][norm['fit']])&set(source['clusters'][norm['hold']])
        h=source_state(p,source,g,prefix,device)
        model=SpatialConsequence(h.shape[1],26).to(device)
        model.load_state_dict(source['states'][prefix+'_State']);model.eval()
        value=evaluate(model,h,action,physical,g,zero,rows)*norm['scale'].to(device)+source['base'].to(device)
        hold=norm['hold'];count[hold]+=1
        expected=source['oof_predictions']['State'][hold].to(device)
        errors[prefix]=float((value[hold]-expected).abs().max())
    assert (count==1).all() and max(errors.values())<1e-5
    assert np.max(np.abs(source['oof_predictions']['State'][source['test']].numpy()-source['full_predictions']['State'][source['test']]))<1e-6
    return errors


def prepare_inputs(p,bridge,g,h,source,forecast,device):
    nominal=g['targets'];predicted=forecast['oof_q']['Ha'].to(device)
    q0=p['history'][:,-1,6:18].to(device)
    raw=torch.cat((q0,nominal[:,0,6:],predicted[:,0,6:]),-1)/.32
    norm=standardize_fit(raw,torch.as_tensor(source['train'],device=device))
    common=torch.cat((h,normalize(raw,norm)),-1)
    action={}
    for prefix,q in (('Nom',nominal),('Pred',predicted)):
        action[prefix+'Joint']=torch.nn.functional.pad((q[:,:,6:]-q[:,:1,6:])/.32,(0,20))
        action[prefix+'Finger']=finger_summary(intrinsic_finger_flow(p,bridge,g,q))
        assert all(float(action[prefix+name][:,0].abs().max())==0 for name in ('Joint','Finger'))
    return common,action,norm


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--forecast-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();args.run_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(2)
    def save(name,value):(args.run_dir/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    paths=[Path(__file__).resolve(),TASK/'src/relative_finger_consequence.py',TASK/'src/spatial_action_fidelity.py',
        TASK/'tools/run/probe_spatial_action_fidelity.py',TASK/'src/execution_geometry.py',TASK/'src/spatial_consequence.py',
        TASK/'src/geometric_consequence.py',TASK/'src/conditional_consequence.py',TASK/'src/consequence_sufficiency.py',
        TASK/'src/intervention.py',TASK/'tools/run/probe_execution_geometry.py',TASK/'tools/audit/probe_duration_response.py',
        TASK/'tools/run/probe_conditional_consequence.py',TASK/'tools/run/probe_geometric_consequence.py',
        TASK/'tools/run/probe_interventions.py',ROOT/'src/task/ObjectInteractionCmv2/model.py']
    manifest=dict(run_status='STARTED',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        created_at=datetime.now(timezone.utc).isoformat(),seeds=[245,246,251],smoke=args.smoke,updates=1 if args.smoke else 300,
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),command=sys.argv,
        dataset_sha256=sha(args.dataset),forecast_manifest_sha256=sha(args.forecast_run/'manifest.json'),
        forecast_diagnostic_sha256=sha(args.forecast_run/'diagnostic.pt'),code_sha256={str(f.relative_to(ROOT)):sha(f) for f in paths},
        new_head_fits=7,reused_state_fits=4,simulation_executed=False,task_or_policy_executed=False)
    save('manifest.json',manifest)
    try:
        dev=torch.device('cuda:0');torch.cuda.set_device(dev);torch.cuda.reset_peak_memory_stats(dev)
        p,bridge,g,h,source,forecast,source_run,original_run=load_inputs(args,dev)
        manifest.update(source_run=str(source_run),original_run=str(original_run),
            source_diagnostic_sha256=sha(source_run/'diagnostic.pt'),original_diagnostic_sha256=sha(original_run/'diagnostic.pt'))
        assert torch.equal(physical_targets(p['before'],p['trajectory'][:,7]),source['z'])
        replay=nuisance_replay(p,g,source,dev)
        common,actions,extra_norm=prepare_inputs(p,bridge,g,h,source,forecast,dev)
        train,test=source['train'],source['test'];tr,te=(torch.as_tensor(v,device=dev) for v in (train,test))
        actual=torch.as_tensor(source['arms'],device=dev);rows=torch.arange(len(actual),device=dev)
        mu=source['oof_predictions']['State'].to(dev);z=source['z'].to(dev);scale=source['target_scale'].to(dev)
        target=(z-mu)/scale
        _,groups=current_design(p,p['episode_id'].numpy()//168)
        permutation=np.arange(len(actual))
        for ids in (train,test):permutation[ids]=permute_within(groups,ids,251)[ids]
        shuffled=actual[torch.as_tensor(permutation,device=dev)]
        states,records,predictions,candidates,means={}, {}, {}, {}, {}
        for name,(kind,centered) in SPECS.items():
            arm=shuffled if name=='PredFingerShuffled' else actual
            model,value,record=fit_head(common,actions[kind],arm,target,tr,centered,manifest['updates'])
            states[name]=cpu_tree(model.state_dict());records[name]=record
            predictions[name]=(mu+value[rows,arm]*scale).cpu().numpy()
            candidates[name]=(mu[te,None]+value[te]*scale).cpu().numpy()
            means[name]=float(value[te].mean(1).square().mean().sqrt())
            if centered:assert float(value.mean(1).abs().max())<1e-6
            if name=='PredFingerCentered':predictions['PredFinger_test_shuffled']=(mu+value[rows,shuffled]*scale).cpu().numpy()
            print(json.dumps(dict(fit=name,source_residual_mse=record['final_source_residual_mse'],elapsed=time.monotonic()-started)),flush=True)
        assert len({v['initial_hash'] for v in records.values()})==1
        for name,key in (('State','State'),('PriorJoint','Joint'),('PriorArm','Arm')):
            predictions[name]=source['full_predictions'][key]
        predictions['PriorPredJoint']=forecast['predictions']['PredictedJoint'];predictions['PriorPredSurface']=forecast['predictions']['PredictedSurface']
        errors={name:((pred[test]-source['z'].numpy()[test])/scale.cpu().numpy())**2 for name,pred in predictions.items()}
        metrics={name:dict(E_mse=float(err[:,:12].mean()),I_mse=float(err[:,12:].mean())) for name,err in errors.items()}
        pairs=[('NomJointCentered','NomJointPlain'),('NomFingerCentered','NomFingerPlain'),('NomFingerPlain','NomJointPlain'),
            ('NomFingerCentered','NomJointCentered'),('PredFingerCentered','NomFingerCentered'),('PredFingerCentered','PredJointCentered'),
            ('PredFingerCentered','PredFingerShuffled'),('PredFinger_test_shuffled','PredFingerCentered'),('PredFingerCentered','PriorJoint')]
        pairs += [(name,'State') for name in SPECS]
        comparisons={new+'_vs_'+control:{key:cluster_gain(errors[control][:,axes].mean(1),errors[new][:,axes].mean(1),source['clusters'][test],251)
            for key,axes in (('E',np.arange(12)),('I',np.arange(12,26)))} for new,control in pairs}
        contrasts={}
        for name,value in candidates.items():
            effect=(value[:,1:]-value[:,:1]).mean(0);beta=source['GT_contrasts']
            contrasts[name]={key:dict(raw=contrast_scores(beta,effect,scale.cpu().numpy(),axes),
                centered=contrast_scores(beta-beta.mean(0),effect-effect.mean(0),scale.cpu().numpy(),axes),
                plus_minus=contrast_scores(beta[::2]-beta[1::2],effect[::2]-effect[1::2],scale.cpu().numpy(),axes))
                for key,axes in (('E',np.arange(12)),('I',np.arange(12,26)))}
        primary='PredFingerCentered';c=comparisons[primary+'_vs_State']['I']
        A=c['gain']>=.05 and c['lower95']>0 and comparisons[primary+'_vs_PredFingerShuffled']['I']['gain']>=.03 and -comparisons['PredFinger_test_shuffled_vs_'+primary]['I']['gain']>=.03
        c=contrasts[primary]['I'];B=c['raw']['correlation'] is not None and c['raw']['correlation']>=.5 and c['raw']['signed_agreement']>=.65 and c['raw']['gain_vs_zero']>=.1 and c['centered']['gain_vs_zero']>0
        specific=all(comparisons[primary+'_vs_'+control]['I']['gain']>=.03 and comparisons[primary+'_vs_'+control]['I']['lower95']>0 for control in ('PredJointCentered','PriorJoint'))
        result=dict(status='UNCLEAR' if args.smoke else ('PROMISING' if A and B and specific else 'UNPROMISING'),engineering_only=args.smoke,
            gates=dict(A=A,B=B,geometry_specific=specific),metrics=metrics,comparisons=comparisons,contrasts=contrasts,
            nuisance_replay=replay,test_mean_candidate_residual_rms=means,elapsed_seconds=time.monotonic()-started,
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(dev),
            limits='Factual full-test exploratory screen; uniform candidate-centering convention, source OOF state nuisance. No nested consequence OOF, future physical input, independent GT contrast, downstream or policy claim.')
        save('result.json',result);save('fit_records.json',records)
        torch.save(dict(train=train,test=test,clusters=source['clusters'],arms=source['arms'],common=common.cpu(),actions=cpu_tree(actions),
            extra_norm=cpu_tree(extra_norm),states=states,predictions=predictions,candidates=candidates,permutation=permutation),args.run_dir/'diagnostic.pt')
        assert all(sha(ROOT/k)==v for k,v in manifest['code_sha256'].items())
        assert sha(args.dataset)==manifest['dataset_sha256'] and sha(args.forecast_run/'diagnostic.pt')==manifest['forecast_diagnostic_sha256']
        assert sum(f.stat().st_size for f in args.run_dir.rglob('*') if f.is_file())<20*1024**2
        manifest['run_status']='COMPLETED';print(json.dumps(dict(status=result['status'],gates=result['gates'],metrics=metrics,elapsed=result['elapsed_seconds'])),flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',failure=f'{type(error).__name__}: {error}');raise
    finally:
        manifest.update(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started);save('manifest.json',manifest)


if __name__=='__main__':main()
