#!/usr/bin/env python3
"""Repair continuous-joint projection; reuse execution and oracle weights."""
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
sys.path[:0]=[str(ROOT),str(TASK/'src'),str(TASK/'tools/audit')]
from execution_geometry import (execution_inputs,fit_execution,endpoint_flows,verify_realized_fk,
    joint_units,joint_slots,spatial_variant)
from geometric_consequence import NominalSurfaceActions,pca_apply,ridge_predict
from execution_projection import continuous_mask,project_execution
from spatial_consequence import SpatialConsequence,evaluate,fit,UPDATES
from conditional_consequence import permute_within,contrast_scores
from consequence_sufficiency import cluster_gain
from probe_conditional_consequence import preprocess,cpu_state
from probe_geometric_consequence import cpu_tree
from probe_interventions import sha
from probe_duration_response import current_design


def to_device(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,dict):return {k:to_device(v,device) for k,v in value.items()}
    return value


def source_state(p,d,geometry,prefix,device):
    norm=to_device(d['preprocess'][prefix],device)
    h,_,_=preprocess(p,torch.as_tensor(norm['fit'],device=device),device,norm['h'])
    return torch.cat((h,pca_apply(geometry['geometry'],norm['geometry'])),-1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--original-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();args.run_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);started=time.monotonic()
    def save(name,value):(args.run_dir/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    original_manifest=json.loads((args.original_run/'manifest.json').read_text())
    original=torch.load(args.original_run/'diagnostic.pt',map_location='cpu',weights_only=False)
    args.source_run=Path(original['source_run'])
    source=json.loads((args.source_run/'manifest.json').read_text())
    paths=[Path(__file__).resolve(),TASK/'src/execution_geometry.py',TASK/'src/execution_projection.py',TASK/'tools/run/probe_execution_geometry.py',TASK/'src/spatial_consequence.py',
        TASK/'src/geometric_consequence.py',TASK/'src/conditional_consequence.py',TASK/'src/consequence_sufficiency.py',
        TASK/'src/intervention.py',TASK/'tools/run/probe_spatial_consequence.py',TASK/'tools/run/probe_conditional_consequence.py',
        TASK/'tools/run/probe_geometric_consequence.py',TASK/'tools/run/probe_interventions.py',
        TASK/'tools/audit/probe_duration_response.py',ROOT/'src/task/ObjectInteractionCmv2/model.py']
    manifest=dict(run_status='STARTED',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        device='cuda:0',physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),seeds=[245,246,247],command=sys.argv,smoke=args.smoke,
        updates=1 if args.smoke else UPDATES,dataset_sha256=sha(args.dataset),source_manifest_sha256=sha(args.source_run/'manifest.json'),
        source_diagnostic_sha256=sha(args.source_run/'diagnostic.pt'),code_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        created_at=datetime.now(timezone.utc).isoformat(),projection='bounded_only_preserve_continuous',
        original_manifest_sha256=sha(args.original_run/'manifest.json'),original_diagnostic_sha256=sha(args.original_run/'diagnostic.pt'),
        reused_execution_fits=8,reused_oracle_fits=2,new_spatial_fits=3)
    save('manifest.json',manifest)
    try:
        assert source['run_status']=='COMPLETED' and not source['smoke']
        assert original_manifest['run_status']=='COMPLETED' and not original_manifest['smoke']
        assert all(sha(ROOT/k)==v for k,v in original_manifest['code_sha256'].items())
        assert manifest['dataset_sha256']==source['dataset_sha256']
        assert all(sha(ROOT/k)==v for k,v in source['code_sha256'].items())
        assert all(sha(Path(k))==v for k,v in source['hand_visual_sha256'].items())
        p=torch.load(args.dataset,map_location='cpu',weights_only=False)
        d=torch.load(args.source_run/'diagnostic.pt',map_location='cpu',weights_only=False)
        device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
        bridge=NominalSurfaceActions(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf',device)
        g=bridge.build(p)
        continuous=continuous_mask(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf',device)
        assert continuous.nonzero().flatten().tolist()==[3,4,5]
        assert max(float((g[k].cpu()-d['geometry'][k]).abs().max()) for k in ('points','nominal_flow','geometry','joint'))==0
        g['obj_points']=d['geometry']['obj_points'].to(device);g['obj_normals']=d['geometry']['obj_normals'].to(device)
        train,test=d['train'],d['test'];clusters=d['clusters'];arms=d['arms']
        assert not set(clusters[train])&set(clusters[test])
        tr,te=(torch.as_tensor(ids,device=device) for ids in (train,test))
        rows=torch.arange(len(arms),device=device);actual=torch.as_tensor(arms,device=device);zero=torch.zeros_like(actual)
        units=joint_units(device);q0=p['history'][:,-1,:18].to(device);q8=p['native_q'][:,7].to(device)
        target=(q8-q0)/units
        realized_flow,realized_links=endpoint_flows(p,bridge,g,q8[:,None])
        fk_audit=verify_realized_fk(p,realized_links)
        execution_states,execution_norms,execution_records={}, {}, {}
        full_q,oof_q={}, {name:torch.full((len(arms),15,18),float('nan'),device=device) for name in ('H','Ha')}
        for prefix,norm in d['preprocess'].items():
            ids=torch.as_tensor(norm['fit'],device=device);hold=torch.as_tensor(norm['hold'],device=device)
            assert not set(clusters[norm['fit']])&set(clusters[norm['hold']])
            h=source_state(p,d,g,prefix,device)
            state,actions,en=execution_inputs(p,h,g,ids)
            execution_norms[prefix]=cpu_tree(en)
            for name in ('H','Ha'):
                model=to_device(original['execution_states'][prefix+'_'+name],device)
                assert all(torch.equal(en[k].cpu(),original['execution_norms'][prefix][k]) for k in en)
                values=[]
                for arm in range(15):
                    a=actions[:,arm] if name=='Ha' else torch.zeros_like(actions[:,0])
                    values.append(ridge_predict(torch.cat((state,a),-1),model))
                raw_q=q0[:,None]+torch.stack(values,1)*units
                clipped=project_execution(raw_q,bridge.lower,bridge.upper,continuous)
                execution_states[prefix+'_'+name]=cpu_tree(model)
                execution_records[prefix+'_'+name]=dict(source=len(ids),hold=len(hold),
                    source_q_mse=float(((clipped[ids,actual[ids]]-q8[ids])/units).square().mean()),
                    hold_clip_fraction=float((raw_q[hold]!=clipped[hold]).float().mean()))
                oof_q[name][hold]=clipped[hold]
                if prefix=='full':full_q[name]=clipped
        assert all(torch.isfinite(v).all() for v in oof_q.values())
        # source full fills outer-test; fold fits overwrite train OOF only.
        forecast_q=oof_q['Ha'];forecast_flow,_=endpoint_flows(p,bridge,g,forecast_q)
        factual_flow=forecast_flow[rows,actual][:,None]
        q_errors={name:((value[rows,actual]-q8)/units).square().cpu().numpy() for name,value in full_q.items()}
        q_errors['Nominal']=((g['targets'][rows,actual]-q8)/units).square().cpu().numpy()
        q_errors['Persistence']=target.square().cpu().numpy()
        flow_errors={}
        for name,q in (('H',full_q['H']),('Ha',full_q['Ha'])):
            f,_=endpoint_flows(p,bridge,g,q[rows,actual][:,None])
            flow_errors[name]=(f-realized_flow).square().mean((1,2,3)).cpu().numpy()
        flow_errors['Nominal']=(g['nominal_flow'][rows,actual][:,None]-realized_flow).square().mean((1,2,3)).cpu().numpy()
        flow_errors['Persistence']=realized_flow.square().mean((1,2,3)).cpu().numpy()
        execution_metrics={name:dict(q18_scaled_mse=float(err[test].mean()),finger12_scaled_mse=float(err[test,6:].mean()),
            surface_endpoint_rmse_mm=float(np.sqrt(flow_errors[name][test].mean())*1000)) for name,err in q_errors.items()}
        execution_comparisons={}
        for new,control in (('Ha','H'),('Ha','Nominal'),('Ha','Persistence')):
            execution_comparisons[new+'_vs_'+control]=dict(finger12=cluster_gain(q_errors[control][test,6:].mean(1),q_errors[new][test,6:].mean(1),clusters[test],247),
                surface=cluster_gain(flow_errors[control][test],flow_errors[new][test],clusters[test],247))
        h=source_state(p,d,g,'full',device)
        z=d['z'].to(device);base=d['base'].to(device);scale=d['target_scale'].to(device)
        target_z=(z-base)/scale;blank=torch.zeros_like(z);a_blank=torch.zeros(len(z),32,device=device)
        _,groups=current_design(p,p['episode_id'].numpy()//168)
        # Groups use source wave/motion/phase, exactly as the previous Probe.
        perm=np.arange(len(arms))
        for ids in (train,test):perm[ids]=permute_within(groups,ids,247)[ids]
        shuffled_arms=actual[torch.as_tensor(perm,device=device)]
        shuffle_flow=forecast_flow[rows,shuffled_arms][:,None]
        variants=dict(RealizedSurface=(a_blank,spatial_variant(g,realized_flow),zero),
            RealizedJoint=(joint_slots(q8-q0),g,zero),
            PredictedSurface=(a_blank,spatial_variant(g,factual_flow),zero),
            PredictedJoint=(joint_slots(forecast_q[rows,actual]-q0),g,zero),
            PredictedSurfaceShuffled=(a_blank,spatial_variant(g,shuffle_flow),zero))
        predictions,states,records,candidates={},{},{},{}
        source_replay={}
        # Reuse prior fixed State/Flow/Arm/Joint controls; do not refit exposed
        # baselines. Exact replay verifies identical inputs/scales beforehand.
        for name in ('State','Nominal','Arm','Joint'):
            source_variant='Flow' if name=='Nominal' else name
            predictions[name]=d['full_predictions'][source_variant]
            a=a_blank;sa=zero
            if name=='Nominal':sa=actual
            elif name=='Arm':a=torch.nn.functional.pad(torch.nn.functional.one_hot(actual,15)[:,1:].float(),(0,18))
            elif name=='Joint':a=torch.nn.functional.pad(g['joint'][rows,actual]/.32,(0,14))
            restored=SpatialConsequence(h.shape[1],26).to(device)
            restored.load_state_dict(d['states']['full_'+source_variant]);restored.eval()
            replay=(evaluate(restored,h,a,blank,g,sa,rows)*scale+base).cpu().numpy()
            source_replay[name]=float(np.max(np.abs(replay-predictions[name])))
            assert source_replay[name]<1e-6
        for name in ('RealizedSurface','RealizedJoint'):
            a,gg,sa=variants[name]
            restored=SpatialConsequence(h.shape[1],26).to(device)
            restored.load_state_dict(original['states'][name]);restored.eval()
            raw=(evaluate(restored,h,a,blank,gg,sa,rows)*scale+base).cpu().numpy()
            source_replay[name]=float(np.max(np.abs(raw-original['predictions'][name])))
            assert source_replay[name]<1e-6
            predictions[name]=original['predictions'][name]
        for name in ('PredictedSurface','PredictedJoint','PredictedSurfaceShuffled'):
            a,gg,sa=variants[name]
            model,pred,record=fit(h,a,blank,gg,sa,target_z,tr,manifest['updates'])
            raw=pred*scale+base
            predictions[name]=raw.cpu().numpy();states[name]=cpu_state(model.state_dict());records[name]=record
            if name=='PredictedSurface':
                values=[]
                for arm in range(15):
                    candidate_flow=forecast_flow[:,arm:arm+1]
                    values.append((evaluate(model,h,a_blank,blank,spatial_variant(g,candidate_flow),zero,te)*scale+base[te]).cpu().numpy())
                candidates[name]=np.stack(values,1)
                shuffled=(evaluate(model,h,a_blank,blank,spatial_variant(g,shuffle_flow),zero,rows)*scale+base).cpu().numpy()
                predictions['PredictedSurface_test_shuffled']=shuffled
            print(json.dumps(dict(fit=name,train_mse=record['final_train_mse'],elapsed=time.monotonic()-started)),flush=True)
        assert len({v['initial_hash'] for v in records.values()})==1
        original_records=json.loads((args.original_run/'fit_records.json').read_text())
        assert all(record['initial_hash']==original_records[name]['initial_hash'] for name,record in records.items())
        errors={name:((pred[test]-d['z'].numpy()[test])/scale.cpu().numpy())**2 for name,pred in predictions.items()}
        metrics={name:dict(E_mse=float(err[:,:12].mean()),I_mse=float(err[:,12:].mean()),joint_mse=float(err.mean())) for name,err in errors.items()}
        comparisons={}
        for new,control in (('RealizedSurface','Nominal'),('RealizedSurface','State'),('RealizedJoint','State'),('RealizedSurface','RealizedJoint'),
            ('PredictedSurface','Nominal'),('PredictedSurface','State'),('PredictedSurface','Joint'),('PredictedSurface','Arm'),
            ('PredictedSurface','PredictedJoint'),('PredictedSurface','PredictedSurfaceShuffled'),('PredictedSurface_test_shuffled','PredictedSurface')):
            comparisons[new+'_vs_'+control]={key:cluster_gain(errors[control][:,axes].mean(1),errors[new][:,axes].mean(1),clusters[test],247)
                for key,axes in (('E',np.arange(12)),('I',np.arange(12,26)))}
        value=candidates['PredictedSurface'];effect=(value[:,1:]-value[:,:1]).mean(0);beta=d['GT_contrasts']
        contrasts={key:dict(raw=contrast_scores(beta,effect,scale.cpu().numpy(),axes),
            centered=contrast_scores(beta-beta.mean(0),effect-effect.mean(0),scale.cpu().numpy(),axes),
            plus_minus=contrast_scores(beta[::2]-beta[1::2],effect[::2]-effect[1::2],scale.cpu().numpy(),axes))
            for key,axes in (('E',np.arange(12)),('I',np.arange(12,26)))}
        oracle=comparisons['RealizedSurface_vs_Nominal']['I']['gain']>=.1 and comparisons['RealizedSurface_vs_Nominal']['I']['lower95']>0
        execution=execution_comparisons['Ha_vs_Nominal']['surface']['gain']>=.2 and execution_comparisons['Ha_vs_H']['finger12']['gain']>=.1 and execution_comparisons['Ha_vs_H']['finger12']['lower95']>0
        A=comparisons['PredictedSurface_vs_State']['I']['gain']>=.05 and comparisons['PredictedSurface_vs_State']['I']['lower95']>0 and comparisons['PredictedSurface_vs_PredictedSurfaceShuffled']['I']['gain']>=.03 and -comparisons['PredictedSurface_test_shuffled_vs_PredictedSurface']['I']['gain']>=.03
        c=contrasts['I']
        B=c['raw']['correlation'] is not None and c['raw']['correlation']>=.5 and c['raw']['signed_agreement']>=.65 and c['raw']['gain_vs_zero']>=.1 and c['centered']['gain_vs_zero']>0
        specific=all(comparisons['PredictedSurface_vs_'+name]['I']['gain']>=.03 for name in ('Arm','Joint','PredictedJoint'))
        result=dict(status='UNCLEAR' if args.smoke else ('PROMISING' if A and B and specific else 'UNPROMISING'),engineering_only=args.smoke,
            gates=dict(realized_oracle=oracle,execution_predictability=execution,A=A,B=B,geometry_specific=specific),
            execution_metrics=execution_metrics,execution_comparisons=execution_comparisons,predictor_metrics=metrics,
            predictor_comparisons=comparisons,contrasts=contrasts,realized_fk=fk_audit,
            source_control_replay=source_replay,
            elapsed_seconds=time.monotonic()-started,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device),
            downstream_executed=False,policy_executed=False,
            limits='Realized motion is post-treatment diagnostic oracle, never selector input. Predicted execution uses source-only environment OOF. Exposed split, fixed300 updates, no task/causal utility claim.')
        save('result.json',result);save('fit_records.json',records)
        torch.save(dict(source_run=str(args.source_run),original_run=str(args.original_run),train=train,test=test,clusters=clusters,arms=arms,
            execution_records=execution_records,
            full_q=cpu_tree(full_q),oof_q=cpu_tree(oof_q),states=states,predictions=predictions,candidates=candidates,
            permutation=perm),args.run_dir/'diagnostic.pt')
        assert sha(args.original_run/'diagnostic.pt')==manifest['original_diagnostic_sha256']
        assert sha(args.dataset)==manifest['dataset_sha256'] and sha(args.source_run/'diagnostic.pt')==manifest['source_diagnostic_sha256']
        assert all(sha(ROOT/k)==v for k,v in manifest['code_sha256'].items())
        assert all(sha(ROOT/k)==v for k,v in source['code_sha256'].items())
        assert all(sha(Path(k))==v for k,v in source['hand_visual_sha256'].items())
        manifest['run_status']='COMPLETED';print(json.dumps(result,indent=2),flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',failure=f'{type(error).__name__}: {error}');raise
    finally:
        manifest.update(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started);save('manifest.json',manifest)


if __name__=='__main__':main()
