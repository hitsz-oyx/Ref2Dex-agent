#!/usr/bin/env python3
"""Fixed spatial OI-CmV2 nominal-action Decision Probe, strict environment OOF."""
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

ROOT=Path(__file__).resolve().parents[5]
TASK=ROOT/'src/task/cm-interaction-oracle'
sys.path[:0]=[str(ROOT),str(TASK/'src'),str(TASK/'tools/audit')]
from spatial_consequence import fit, evaluate, action_slots, UPDATES, SEED
from geometric_consequence import NominalSurfaceActions, persistence_baseline, pca_fit, pca_apply
from conditional_consequence import environment_folds, permute_within, contrast_scores, oracle_retention
from consequence_sufficiency import readouts, cluster_gain, PRIMARY
from probe_conditional_consequence import preprocess, cpu_state
from probe_geometric_consequence import cpu_tree
from probe_interventions import sha, standardized
from probe_gt_consequence_sufficiency import EXPECTED_SHA
from probe_duration_response import current_design, residual_fit
from audit_finger_amplitudes import finger_audit, markdown

VARIANTS=('State','Arm','Joint','Flow','Shuffled')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--oracle-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true',help='one update per fit, engineering only')
    args=parser.parse_args()
    args.run_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);started=time.monotonic()
    def save(name,value): (args.run_dir/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    sample_path=args.dataset.parent/'object_surface_sample.pt'
    geometry_source=ROOT/'outputs/cm-interaction-oracle/geometric-innovation-s241-r2/manifest.json'
    source=json.loads(geometry_source.read_text())
    paths=[Path(__file__),TASK/'src/spatial_consequence.py',TASK/'src/geometric_consequence.py',
        TASK/'src/conditional_consequence.py',TASK/'src/consequence_sufficiency.py',TASK/'src/intervention.py',
        TASK/'tools/run/probe_conditional_consequence.py',TASK/'tools/run/probe_interventions.py',
        TASK/'tools/audit/probe_duration_response.py',TASK/'tools/audit/audit_finger_amplitudes.py',
        ROOT/'src/task/ObjectInteractionCmv2/model.py',ROOT/'src/task/CmResidual/v118_planner.py',
        ROOT/'src/task/CmResidual/dexplore_cm_geometry.py',ROOT/'src/task/CmResidual/surface_execution.py']
    manifest=dict(run_status='STARTED',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        command=sys.argv,seeds=[245,246],device='cuda:0',physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},dataset_sha256=sha(args.dataset),
        sample_sha256=sha(sample_path),oracle_sha256=sha(args.oracle_run/'diagnostic.pt'),
        geometry_source_sha256=sha(geometry_source),hand_visual_sha256=source['hand_visual_sha256'],
        geometry_code_sha256={k:v for k,v in source['code_sha256'].items() if 'CmResidual/' in k or k.endswith('.urdf')},
        updates=1 if args.smoke else UPDATES,smoke=args.smoke,created_at=datetime.now(timezone.utc).isoformat())
    save('manifest.json',manifest)
    try:
        assert manifest['dataset_sha256']==EXPECTED_SHA
        assert manifest['oracle_sha256']=='16b6b0961c467e2836ca90cf0dc7285269570ac8fb1ed8129619e84091c63884'
        assert all(sha(Path(k))==v for k,v in manifest['hand_visual_sha256'].items())
        assert all(sha(ROOT/k)==v for k,v in manifest['geometry_code_sha256'].items())
        provenance=json.loads((args.dataset.parent/'geometry_provenance.json').read_text())
        assert manifest['sample_sha256']==provenance['sample_sha256']
        p=torch.load(args.dataset,map_location='cpu',weights_only=False)
        old=torch.load(args.oracle_run/'diagnostic.pt',map_location='cpu',weights_only=False)
        train,test,clusters=old['train'],old['test'],old['clusters']
        assert not set(clusters[train])&set(clusters[test])
        e,i,_,y,details=readouts(p);z=torch.cat((e,i),-1)
        assert details['risk'].all() and np.array_equal(y.numpy(),old['target'])
        assert (p['duration']==8).all() and (p['amplitude']==1).all()
        device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
        tr,te=(torch.as_tensor(ids,device=device) for ids in (train,test))
        urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
        geometry=NominalSurfaceActions(urdf,device).build(p)
        sample=torch.load(sample_path,weights_only=False)
        # Deterministic fixed coverage, no data-driven point selection.
        object_ids=torch.linspace(0,len(sample['points'])-1,64).round().long()
        geometry['obj_points']=sample['points'][object_ids].to(device)[None]
        geometry['obj_normals']=sample['normals'][object_ids].to(device)[None]
        actual=p['arm'].to(device);arms=actual.cpu().numpy();rows=torch.arange(len(arms),device=device)
        onehot=torch.nn.functional.one_hot(actual,15)[:,1:].float()
        folds=environment_folds(train,old['motion'],clusters,SEED)
        design,groups=current_design(p,old['waves'])
        zg,yg=z.to(device),y.to(device)
        base=persistence_baseline(p['before'].to(device))
        _,zm,zs=standardized(zg,tr);yn,ym,ys=standardized(yg,tr)
        full,oof,candidates,states,records,norms,permutations={},{},{},{},{},{},{}
        blank=torch.zeros_like(zg)
        oof={name:torch.full_like(zg,float('nan')) for name in VARIANTS}
        full_models={}
        def run(prefix,fit_np,hold_np,full_fit=False):
            ids=torch.as_tensor(fit_np,device=device);hold=torch.as_tensor(hold_np,device=device)
            h,_,hn=preprocess(p,ids,device)
            gn=pca_fit(geometry['geometry'],ids,16)
            h=torch.cat((h,pca_apply(geometry['geometry'],gn)),-1)
            _,_,scale=standardized(zg,ids)
            residual=(zg-base)/scale
            perm=np.arange(len(arms))
            for partition in (fit_np,hold_np): perm[partition]=permute_within(groups,partition,246)[partition]
            shuffled=actual[torch.as_tensor(perm,device=device)]
            norms[prefix]=cpu_tree(dict(h=hn,geometry=gn,scale=scale,fit=fit_np,hold=hold_np))
            permutations[prefix]=perm
            for name in VARIANTS:
                intended=shuffled if name=='Shuffled' else actual
                action,spatial_arms=action_slots(name,geometry,intended,onehot)
                model,pred,record=fit(h,action,blank,geometry,spatial_arms,residual,ids,manifest['updates'])
                raw=pred*scale+base
                states[prefix+'_'+name]=cpu_state(model.state_dict());records[prefix+'_'+name]=record
                oof[name][hold]=raw[hold]
                if full_fit:
                    full[name]=raw.cpu().numpy();full_models[name]=model
                    values=[]
                    for arm in range(15):
                        intended=torch.full_like(actual,arm)
                        oh=torch.nn.functional.one_hot(intended,15)[:,1:].float()
                        a,sa=action_slots(name,geometry,intended,oh)
                        values.append((evaluate(model,h,a,blank,geometry,sa,hold)*scale+base[hold]).cpu().numpy())
                    candidates[name]=np.stack(values,1)
                    if name=='Flow':
                        a,sa=action_slots('Flow',geometry,shuffled,onehot)
                        full['Flow_test_shuffled']=(evaluate(model,h,a,blank,geometry,sa,rows)*scale+base).cpu().numpy()
                print(json.dumps(dict(fit=prefix+'_'+name,train_mse=record['final_train_mse'],elapsed=time.monotonic()-started)),flush=True)
            assert len({records[prefix+'_'+name]['initial_hash'] for name in VARIANTS})==1
            return h
        h=run('full',train,test,True)
        for fold in range(3):run('fold'+str(fold),train[folds[train]!=fold],train[folds[train]==fold])
        for name in VARIANTS:
            assert torch.isfinite(oof[name][tr]).all()
            oof[name][te]=torch.as_tensor(full[name][test],device=device)
        full['TrainMean']=zm.expand_as(zg).cpu().numpy();full['Persistence']=base.cpu().numpy()
        errors={name:((value[test]-z.numpy()[test])/zs.cpu().numpy())**2 for name,value in full.items()}
        metrics={name:dict(E_mse=float(err[:,:12].mean()),I_mse=float(err[:,12:].mean()),joint_mse=float(err.mean())) for name,err in errors.items()}
        comparisons={}
        for new,control in (('Flow','State'),('Flow','Arm'),('Flow','Joint'),('Flow','Shuffled'),('Flow','TrainMean'),('Flow','Persistence'),('Flow_test_shuffled','Flow')):
            comparisons[new+'_vs_'+control]={key:cluster_gain(errors[control][:,axes].mean(1),errors[new][:,axes].mean(1),clusters[test],246)
                for key,axes in (('E',np.arange(12)),('I',np.arange(12,26)))}
        beta,_,_,rank=residual_fit(design[test],arms[test],z.numpy()[test],14)
        contrasts={}
        for name,value in candidates.items():
            effect=(value[:,1:]-value[:,:1]).mean(0)
            contrasts[name]={key:dict(raw=contrast_scores(beta,effect,zs.cpu().numpy(),axes),
                centered=contrast_scores(beta-beta.mean(0),effect-effect.mean(0),zs.cpu().numpy(),axes),
                plus_minus=contrast_scores(beta[::2]-beta[1::2],effect[::2]-effect[1::2],zs.cpu().numpy(),axes))
                for key,axes in (('E',np.arange(12)),('I',np.arange(12,26)))}
        task_predictions,task_states,task_errors,task_metrics={},{},{},{}
        task_h=torch.cat((h,(base-zm)/zs),-1)
        for name in ('H','Arm','Joint','Flow','GT','P_State','P_Flow','Flow_P_Flow'):
            variant=name if name in ('Arm','Joint','Flow') else ('Flow' if name=='Flow_P_Flow' else 'State')
            a,sa=action_slots(variant,geometry,actual,onehot)
            physical=blank if name not in ('GT','P_State','P_Flow','Flow_P_Flow') else ((zg-zm)/zs if name=='GT' else (oof['State' if name=='P_State' else 'Flow']-zm)/zs)
            model,pred,record=fit(task_h,a,physical,geometry,sa,yn,tr,manifest['updates'])
            raw=pred*ys+ym;err=((raw-yg)/ys).square().cpu().numpy()[test]
            task_predictions[name]=raw.cpu();task_states[name]=cpu_state(model.state_dict());task_errors[name]=err
            task_metrics[name]=dict(primary_mse=float(err[:,PRIMARY].mean()),physical_failure_mse=float(err[:,6].mean()),per_head=err.mean(0).tolist())
            records['task_'+name]=record
            print(json.dumps(dict(task=name,**task_metrics[name],elapsed=time.monotonic()-started)),flush=True)
        assert len({records['task_'+name]['initial_hash'] for name in task_predictions})==1
        task_comparisons={}
        for new,control in (('GT','H'),('P_Flow','H'),('P_Flow','Flow'),('Flow_P_Flow','Flow'),('P_Flow','P_State')):
            task_comparisons[new+'_vs_'+control]={key:cluster_gain(task_errors[control][:,axes].mean(1),task_errors[new][:,axes].mean(1),clusters[test],246)
                for key,axes in (('primary',PRIMARY),('physical_failure',(6,)))}
        retention=oracle_retention(task_errors['H'][:,PRIMARY].mean(1),task_errors['GT'][:,PRIMARY].mean(1),task_errors['P_Flow'][:,PRIMARY].mean(1),clusters[test],246)
        A=comparisons['Flow_vs_State']['I']['gain']>=.05 and comparisons['Flow_vs_State']['I']['lower95']>0 and comparisons['Flow_vs_Shuffled']['I']['gain']>=.03 and -comparisons['Flow_test_shuffled_vs_Flow']['I']['gain']>=.03
        cm=contrasts['Flow']['I']
        B=rank==14 and cm['raw']['correlation'] is not None and cm['raw']['correlation']>=.5 and cm['raw']['signed_agreement']>=.65 and cm['raw']['gain_vs_zero']>=.1 and cm['centered']['gain_vs_zero']>0
        geometry_gain=all(comparisons['Flow_vs_'+name]['I']['gain']>=.03 for name in ('Arm','Joint','TrainMean'))
        C=any(task_comparisons[name+'_vs_Flow']['primary']['gain']>=.05 and task_comparisons[name+'_vs_Flow']['primary']['lower95']>0 and task_comparisons[name+'_vs_Flow']['physical_failure']['gain']>=-.02 for name in ('P_Flow','Flow_P_Flow'))
        # An isolated win over a weak direct Flow is insufficient.
        unique=task_comparisons['P_Flow_vs_P_State']['primary']['gain']>=.05 and task_comparisons['P_Flow_vs_P_State']['primary']['lower95']>0
        result=dict(status='PROMISING' if A and B and geometry_gain and C and unique else ('UNPROMISING' if rank==14 else 'UNCLEAR'),
            engineering_only=args.smoke,gates=dict(A=A,B=B,geometry_specific=geometry_gain,C=C,unique_over_state=unique),
            predictor_metrics=metrics,predictor_comparisons=comparisons,contrasts=contrasts,contrast_rank=rank,
            task_metrics=task_metrics,task_comparisons=task_comparisons,oracle_retention=retention,
            fk_tip_error_max_m=geometry['fingertip_error_max_m'],elapsed_seconds=time.monotonic()-started,
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device),policy_executed=False,
            limitations='Exploratory exposed split; nominal PD endpoint sweep is not realized motion; from-scratch reduced V13 encoder and compact E12/I14 heads, fixed 300 updates; no policy utility.')
        save('result.json',result);save('fit_records.json',records)
        audit=finger_audit(p,design,arms);save('finger_amplitudes.json',audit)
        (args.run_dir/'finger_amplitudes.md').write_text(markdown(audit))
        torch.save(dict(train=train,test=test,clusters=clusters,folds=folds,arms=arms,z=z,y=y,base=base.cpu(),target_mean=zm.cpu(),target_scale=zs.cpu(),
            task_mean=ym.cpu(),task_scale=ys.cpu(),geometry=cpu_tree(geometry),preprocess=norms,permutations=permutations,
            states=states,task_states=task_states,full_predictions=full,oof_predictions=cpu_tree(oof),candidate_arrays=candidates,
            GT_contrasts=beta,task_predictions=task_predictions),args.run_dir/'diagnostic.pt')
        assert sha(args.dataset)==EXPECTED_SHA and sha(sample_path)==manifest['sample_sha256']
        assert all(sha(ROOT/k)==v for k,v in manifest['code_sha256'].items())
        manifest['run_status']='COMPLETED'
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',failure=f'{type(error).__name__}: {error}')
        raise
    finally:
        manifest.update(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
        save('manifest.json',manifest)


if __name__=='__main__':main()
