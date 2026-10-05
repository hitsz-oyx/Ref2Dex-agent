#!/usr/bin/env python3
"""Ref11 State / measured endpoint / measured two-chunk oracle flow screen."""
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
from oracle_hand_flow import measured_flow_inputs,current_state,exact_observable_groups,fit,evaluate
from geometric_consequence import NominalSurfaceActions,persistence_baseline
from intervention import physical_targets
from conditional_consequence import permute_within
from consequence_sufficiency import cluster_gain
from probe_geometric_consequence import cpu_tree
from probe_interventions import sha

EXPECTED_SHA='138b99b21e5567616dbdb6556179cc87dc8b42b578cad7e127dc6d5590e17149'
SPECS={'State':'State','Endpoint':'Endpoint','Chunk':'Chunk','EndpointShuffled':'Endpoint','ChunkShuffled':'Chunk'}


def flow_permutation(p,train,test):
    groups=p['episode_id'].numpy()//168*12+p['motion_id'].numpy()*4+np.minimum((p['context'][:,3].numpy()*4).astype(int),3)
    permutation=np.arange(len(groups))
    for ids in (train,test):permutation[ids]=permute_within(groups,ids,257)[ids]
    return permutation,groups


def summarize(z,scale,predictions,test,clusters):
    errors={name:((value[test]-z[test])/scale)**2 for name,value in predictions.items()}
    metrics={name:dict(E_mse=float(err[:,:12].mean()),I_mse=float(err[:,12:].mean())) for name,err in errors.items()}
    pairs=[]
    for kind in ('Endpoint','Chunk'):
        pairs += [(kind,name) for name in ('State','TrainMean',kind+'Shuffled')]
        pairs += [(kind+'_test_shuffled',kind)]
    pairs += [('Chunk','Endpoint')]
    comparisons={new+'_vs_'+control:{key:cluster_gain(errors[control][:,axes].mean(1),errors[new][:,axes].mean(1),clusters[test],257)
        for key,axes in (('E',np.arange(12)),('I',np.arange(12,26)))} for new,control in pairs}
    links={};gates={}
    for kind in ('Endpoint','Chunk'):
        gates[kind]={}
        for channel in ('E','I'):
            state=comparisons[kind+'_vs_State'][channel];mean=comparisons[kind+'_vs_TrainMean'][channel]
            shuffled=comparisons[kind+'_vs_'+kind+'Shuffled'][channel];sensitivity=comparisons[kind+'_test_shuffled_vs_'+kind][channel]
            passed=state['gain']>=.05 and state['lower95']>0 and mean['gain']>=.03 and mean['lower95']>0 and shuffled['gain']>=.03 and shuffled['lower95']>0 and -sensitivity['gain']>=.03 and sensitivity['upper95']<0
            gates[kind][channel]=bool(passed)
        links[kind]='PROMISING' if all(gates[kind].values()) else 'UNPROMISING'
    return dict(status=links['Chunk'],prediction_links=links,gates=gates,metrics=metrics,comparisons=comparisons)


def paired_observable_scores(pairs,z,predictions,scale,clusters):
    envs=len(set(clusters[pairs[:,0]])) if len(pairs) else 0
    result=dict(pairs=len(pairs),environments=envs,status='UNCLEAR',scope='Exact observed snapshots only; simulator/hidden-contact state not restored. No donor-label transplant.')
    if len(pairs)<30 or envs<10:return result
    a,b=pairs.T;gt=(z[a]-z[b])/scale;active=np.abs(gt[:,12:])>=.1
    scores={}
    for name in ('State','Endpoint','Chunk'):
        value=(predictions[name][a]-predictions[name][b])/scale
        scores[name]={key:dict(gain_vs_zero=float(1-np.mean((value[:,axes]-gt[:,axes])**2)/max(np.mean(gt[:,axes]**2),1e-12)),
            correlation=float(np.corrcoef(value[:,axes].flatten(),gt[:,axes].flatten())[0,1]) if np.std(value[:,axes])>1e-10 else None)
            for key,axes in (('E',np.arange(12)),('I',np.arange(12,26)))}
        scores[name]['I']['signed_agreement']=float(np.mean(value[:,12:][active]*gt[:,12:][active]>0)) if active.any() else None
    result['scores']=scores
    c=scores['Chunk']['I'];positive=c['correlation'] is not None and c['correlation']>=.5 and c['gain_vs_zero']>=.1 and c['signed_agreement'] is not None and c['signed_agreement']>=.65
    result['status']='PROMISING' if positive else 'UNPROMISING'
    return result


def load_inputs(args,device):
    assert sha(args.dataset)==EXPECTED_SHA
    p=torch.load(args.dataset,map_location='cpu',weights_only=False)
    split=torch.load(args.split_run/'diagnostic.pt',map_location='cpu',weights_only=False)
    train,test,clusters=split['train'],split['test'],split['clusters']
    assert not set(clusters[train])&set(clusters[test]) and np.array_equal(clusters,p['episode_id'].numpy()%168)
    bridge=NominalSurfaceActions(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf',device)
    geometry,actions,geometry_checks=measured_flow_inputs(p,bridge)
    h,norms=current_state(p,geometry,torch.as_tensor(train,device=device))
    return p,train,test,clusters,geometry,actions,geometry_checks,h,norms


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--split-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();args.run_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();torch.set_num_threads(2)
    def save(name,value):(args.run_dir/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    geometry_manifest=ROOT/'outputs/cm-interaction-oracle/geometric-innovation-s241-r2/manifest.json'
    assets=json.loads(geometry_manifest.read_text())
    paths=[Path(__file__).resolve(),TASK/'src/oracle_hand_flow.py',TASK/'src/geometric_consequence.py',TASK/'src/intervention.py',
        TASK/'src/conditional_consequence.py',TASK/'src/consequence_sufficiency.py',TASK/'tools/run/probe_interventions.py',
        TASK/'tools/run/probe_geometric_consequence.py',ROOT/'src/task/CmResidual/dexplore_cm_geometry.py',
        ROOT/'src/task/CmResidual/v118_planner.py',ROOT/'src/task/CmResidual/surface_execution.py']
    manifest=dict(run_status='STARTED',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),created_at=datetime.now(timezone.utc).isoformat(),
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),command=sys.argv,seeds=[255,256,257],updates=1 if args.smoke else 300,
        smoke=args.smoke,dataset_sha256=sha(args.dataset),split_diagnostic_sha256=sha(args.split_run/'diagnostic.pt'),
        geometry_manifest_sha256=sha(geometry_manifest),hand_visual_sha256=assets['hand_visual_sha256'],
        urdf_sha256=sha(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'),
        code_sha256={str(path.relative_to(ROOT)):sha(path) for path in paths},new_fits=5,execution_predictor_used=False,
        native_action_or_PD_input=False,simulation_executed=False,task_or_policy_executed=False)
    save('manifest.json',manifest)
    try:
        assert all(sha(Path(k))==v for k,v in manifest['hand_visual_sha256'].items())
        dev=torch.device('cuda:0');torch.cuda.set_device(dev);torch.cuda.reset_peak_memory_stats(dev)
        p,train,test,clusters,geometry,actions,checks,h,norms=load_inputs(args,dev)
        tr=torch.as_tensor(train,device=dev);z=physical_targets(p['before'],p['trajectory'][:,7]).to(dev)
        mean=z[tr].mean(0);scale=z[tr].std(0,unbiased=False).clamp_min(.001);target=(z-mean)/scale
        permutation,groups=flow_permutation(p,train,test);perm=torch.as_tensor(permutation,device=dev)
        predictions,states,records={},{},{}
        for name,kind in SPECS.items():
            action=actions[kind][perm] if name.endswith('Shuffled') else actions[kind]
            model,value,record=fit(h,action,target,tr,manifest['updates'])
            predictions[name]=(value*scale+mean).cpu().numpy();states[name]=cpu_tree(model.state_dict());records[name]=record
            if name in ('Endpoint','Chunk'):predictions[name+'_test_shuffled']=(evaluate(model,h,actions[kind][perm])*scale+mean).cpu().numpy()
        assert len({v['initial_hash'] for v in records.values()})==1
        predictions['TrainMean']=mean.expand_as(z).cpu().numpy();predictions['Persistence']=persistence_baseline(p['before']).numpy()
        result=summarize(z.cpu().numpy(),scale.cpu().numpy(),predictions,test,clusters)
        pairs,support=exact_observable_groups(p,test)
        result.update(oracle_action=True,geometry_checks=checks,contrast_support=support,
            observable_contrasts=paired_observable_scores(pairs,z.cpu().numpy(),predictions,scale.cpu().numpy(),clusters),
            elapsed_seconds=time.monotonic()-started,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(dev),engineering_only=args.smoke,
            limits='Actual post-treatment hand-flow, fixed current object frame. Oracle prediction only, no prospective exogenous plan/paired simulator restoration/control/task/policy claim.')
        if args.smoke:result['status']='UNCLEAR'
        save('result.json',result);save('fit_records.json',records)
        torch.save(dict(train=train,test=test,clusters=clusters,h=h.cpu(),norms=cpu_tree(norms),geometry=geometry.cpu(),
            actions=cpu_tree(actions),z=z.cpu(),target_mean=mean.cpu(),target_scale=scale.cpu(),states=states,predictions=predictions,
            permutation=permutation,groups=groups,observable_pairs=pairs),args.run_dir/'diagnostic.pt')
        assert all(sha(ROOT/k)==v for k,v in manifest['code_sha256'].items()) and sha(args.dataset)==manifest['dataset_sha256']
        assert sha(args.split_run/'diagnostic.pt')==manifest['split_diagnostic_sha256']
        assert sum(path.stat().st_size for path in args.run_dir.rglob('*') if path.is_file())<25*1024**2
        manifest['run_status']='COMPLETED';print(json.dumps(dict(status=result['status'],gates=result['gates'],metrics=result['metrics'],geometry_checks=checks,contrast_support=support,elapsed=result['elapsed_seconds'])),flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',failure=f'{type(error).__name__}: {error}');raise
    finally:
        manifest.update(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started);save('manifest.json',manifest)


if __name__=='__main__':main()
