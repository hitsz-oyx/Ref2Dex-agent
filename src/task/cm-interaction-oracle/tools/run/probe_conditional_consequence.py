#!/usr/bin/env python3
"""Frozen ref9 conditional prediction and OOF consequence task-transfer Probe."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[5]
SCRIPT=Path(__file__).resolve()
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src/task/cm-interaction-oracle/src'))
sys.path.insert(0,str(ROOT/'src/task/cm-interaction-oracle/tools/audit'))
from consequence_sufficiency import readouts, cluster_gain, supported_ranking, HEADS, PRIMARY
from conditional_consequence import environment_folds, permute_within, task_slots, contrast_scores, oracle_retention
from intervention import per_finger_residuals, PER_FINGER_ARM_NAMES
from probe_interventions import standardized, sha, tensor_hash
from probe_gt_consequence_sufficiency import EXPECTED_SHA
from probe_early_hold import dropout_auc
from probe_amplitude_authority import audit_packet
from probe_duration_response import current_design, residual_fit
from audit_finger_amplitudes import finger_audit, markdown

EPOCHS=1500


def cpu_state(state):
    return {key:value.detach().cpu() for key,value in state.items()}


def preprocess(p, fit_ids, device, saved=None):
    raw=torch.cat((p['history'].flatten(1),p['actor_obs'],p['context'],p['base_action']),-1).to(device)
    if saved is None:
        hn,hm,hs=standardized(raw,fit_ids)
        _,_,v=torch.linalg.svd(hn[fit_ids],full_matrices=False)
        projection=v[:32].T
        hp,hpm,hps=standardized(hn@projection,fit_ids)
        before,bm,bs=standardized(p['before'].to(device),fit_ids)
        onehot=torch.nn.functional.one_hot(p['arm'].to(device),15)[:,1:].float()
        action,am,ass=standardized(onehot,fit_ids)
        saved=dict(h_mean=hm,h_scale=hs,projection=projection,hp_mean=hpm,hp_scale=hps,
            before_mean=bm,before_scale=bs,a_mean=am,a_scale=ass)
    else:
        saved={key:value.to(device) for key,value in saved.items()}
        hn=(raw-saved['h_mean'])/saved['h_scale']
        hp=(hn@saved['projection']-saved['hp_mean'])/saved['hp_scale']
        before=(p['before'].to(device)-saved['before_mean'])/saved['before_scale']
        onehot=torch.nn.functional.one_hot(p['arm'].to(device),15)[:,1:].float()
        action=(onehot-saved['a_mean'])/saved['a_scale']
    return torch.cat((hp,before),-1),action,saved


def fit(x,y,ids):
    torch.manual_seed(237)
    model=nn.Sequential(nn.Linear(x.shape[1],64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,y.shape[1])).to(x.device)
    initial=tensor_hash(torch.cat([v.flatten() for v in cpu_state(model.state_dict()).values()]))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001)
    losses=[]; gradients=[]
    for epoch in range(EPOCHS):
        loss=(model(x[ids])-y[ids]).square().mean()
        if not torch.isfinite(loss): raise ValueError('nonfinite loss')
        optimizer.zero_grad(); loss.backward()
        grad=nn.utils.clip_grad_norm_(model.parameters(),2.)
        if not torch.isfinite(grad): raise ValueError('nonfinite gradient')
        optimizer.step(); losses.append(float(loss.detach()))
        if epoch%100==0 or epoch==EPOCHS-1: gradients.append(dict(epoch=epoch+1,preclip_norm=float(grad)))
    model.eval()
    with torch.no_grad(): prediction=model(x)
    record=dict(initial_hash=initial,parameters=sum(v.numel() for v in model.parameters()),epochs=EPOCHS,
        train_loss=losses,gradient_norms=gradients,last200_fractional_change=1-losses[-1]/max(losses[-200],1e-12),
        final_train_loss=float((prediction[ids]-y[ids]).square().mean()))
    return model,prediction.detach(),record


def candidates(model,h,normalizers,target_mean,target_scale,ids,use_action):
    """Same decision states, all intended arms, no post-execution actions."""
    n=len(ids); values=[]
    with torch.no_grad():
        for arm in range(15):
            a=torch.zeros(n,14,device=h.device)
            if arm: a[:,arm-1]=1
            a=(a-normalizers['a_mean'])/normalizers['a_scale']
            if not use_action: a=torch.zeros_like(a)
            values.append((model(torch.cat((h[ids],a),-1))*target_scale+target_mean).cpu().numpy())
    return np.stack(values,1)


def restored_model(state,width,heads,device):
    model=nn.Sequential(nn.Linear(width,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,heads)).to(device)
    model.load_state_dict(state); model.eval(); return model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--oracle-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args(); args.run_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); started=time.monotonic()
    def save(name,value): (args.run_dir/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    code=[SCRIPT,SCRIPT.with_name('probe_interventions.py'),SCRIPT.with_name('probe_early_hold.py'),
        SCRIPT.with_name('probe_gt_consequence_sufficiency.py')]
    code.extend(ROOT/'src/task/cm-interaction-oracle/src'/name for name in
        ('conditional_consequence.py','consequence_sufficiency.py','intervention.py'))
    code.extend(ROOT/'src/task/cm-interaction-oracle/tools/audit'/name for name in
        ('probe_amplitude_authority.py','probe_duration_response.py','audit_finger_amplitudes.py'))
    manifest=dict(run_status='STARTED',command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dataset_sha256=sha(args.dataset),oracle_diagnostic_sha256=sha(args.oracle_run/'diagnostic.pt'),
        oracle_manifest_sha256=sha(args.oracle_run/'manifest.json'),collection_manifest_sha256=sha(args.dataset.parent/'manifest.json'),
        code_sha256={str(path.relative_to(ROOT)):sha(path) for path in code},physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        device='cuda:0',seeds=[237,238],epochs=EPOCHS,torch_version=torch.__version__,created_at=datetime.now(timezone.utc).isoformat())
    save('manifest.json',manifest)
    try:
        assert manifest['dataset_sha256']==EXPECTED_SHA
        assert manifest['oracle_diagnostic_sha256']=='16b6b0961c467e2836ca90cf0dc7285269570ac8fb1ed8129619e84091c63884'
        assert json.loads((args.oracle_run/'manifest.json').read_text())['run_status']=='COMPLETED'
        p=torch.load(args.dataset,map_location='cpu',weights_only=False)
        old=torch.load(args.oracle_run/'diagnostic.pt',map_location='cpu',weights_only=False)
        assert tuple(p['arm_names'])==PER_FINGER_ARM_NAMES
        save('engineering_audit.json',audit_packet(p,args.dataset,per_finger_residuals(.2)))
        e,i,_,y,details=readouts(p); z=torch.cat((e,i),-1)
        assert np.max(np.abs(y.numpy()-old['target']))==0
        train_np,test_np=old['train'],old['test']; clusters=old['clusters']; waves=old['waves']; motion=old['motion']; arms=p['arm'].numpy()
        assert np.array_equal(clusters,p['episode_id'].numpy()%168) and np.array_equal(arms,old['arms'])
        assert not set(clusters[train_np])&set(clusters[test_np]) and details['risk'].all()
        half=waves>=5; design,groups=current_design(p,waves)
        amplitudes=finger_audit(p,design,arms); save('finger_amplitudes.json',amplitudes)
        (args.run_dir/'finger_amplitudes.md').write_text(markdown(amplitudes))
        device=torch.device('cuda:0'); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        train=torch.as_tensor(train_np,device=device); test=torch.as_tensor(test_np,device=device)
        normal_keys=('h_mean','h_scale','projection','hp_mean','hp_scale','before_mean','before_scale','a_mean','a_scale')
        h,a,norm=preprocess(p,train,device,{key:old['normalizers'][key] for key in normal_keys})
        zg=z.to(device); zn,zm,zs=standardized(zg,train)
        yn,ym,ys=standardized(y.to(device),train)
        assert torch.allclose(zm,torch.cat((old['normalizers']['e_mean'],old['normalizers']['i_mean'])).to(device))
        folds=environment_folds(train_np,motion,clusters)
        records={}; predictor_states={}; permutations={}; preprocess_records={}; full_predictions={}; candidate_arrays={}
        oof={name:torch.full_like(zg,float('nan')) for name in ('H','Ha','Shuffled')}
        full_models={}
        def run_predictors(prefix,fit_np,hold_np,h_here,a_here,n_here,oof_mode=False):
            fit_ids=torch.as_tensor(fit_np,device=device); hold_ids=torch.as_tensor(hold_np,device=device)
            target,mean,scale=standardized(zg,fit_ids)
            fit_perm=permute_within(groups,fit_np,238)
            hold_perm=permute_within(groups,hold_np,238)
            permutation=np.arange(len(arms)); permutation[fit_np]=fit_perm[fit_np]; permutation[hold_np]=hold_perm[hold_np]
            permutations[prefix]=dict(fit=fit_np,hold=hold_np,permutation=permutation,
                fit_changed=float((arms[fit_np]!=arms[permutation[fit_np]]).mean()),hold_changed=float((arms[hold_np]!=arms[permutation[hold_np]]).mean()))
            preprocess_records[prefix]=dict(fit_ids=fit_np,hold_ids=hold_np,normalizers=cpu_state(n_here),target_mean=mean.cpu(),target_scale=scale.cpu(),
                source_arm_counts=np.bincount(arms[fit_np],minlength=15).tolist())
            assert min(preprocess_records[prefix]['source_arm_counts'])>=1, 'missing action category in predictor fit'
            for name in ('H','Ha','Shuffled'):
                apad=torch.zeros_like(a_here) if name=='H' else (a_here[torch.as_tensor(permutation,device=device)] if name=='Shuffled' else a_here)
                model,pred,record=fit(torch.cat((h_here,apad),-1),target,fit_ids)
                key=f'{prefix}_{name}'; records[key]=record; predictor_states[key]=cpu_state(model.state_dict())
                raw=pred*scale+mean
                if oof_mode: oof[name][hold_ids]=raw[hold_ids]
                else:
                    full_predictions[name]=raw.cpu().numpy(); full_models[name]=model
                    candidate_arrays[name]=candidates(model,h_here,n_here,mean,scale,hold_ids,name!='H')
                print(json.dumps(dict(predictor=key,last200_change=record['last200_fractional_change'],train_mse=record['final_train_loss'])),flush=True)
                if name=='Ha' and not oof_mode:
                    with torch.no_grad(): shuffled=(model(torch.cat((h_here,a_here[torch.as_tensor(permutation,device=device)]),-1))*scale+mean)
                    full_predictions['Ha_test_shuffled']=shuffled.cpu().numpy()
            assert len({records[f'{prefix}_{name}']['initial_hash'] for name in ('H','Ha','Shuffled')})==1
        run_predictors('full',train_np,test_np,h,a,norm)
        for fold in range(3):
            fit_np=train_np[folds[train_np]!=fold]; hold_np=train_np[folds[train_np]==fold]
            assert len(np.unique(clusters[hold_np]))>=20 and not set(clusters[fit_np])&set(clusters[hold_np])
            hf,af,nf=preprocess(p,torch.as_tensor(fit_np,device=device),device)
            run_predictors(f'fold{fold}',fit_np,hold_np,hf,af,nf,True)
        for name in oof:
            assert torch.isfinite(oof[name][train]).all()
            oof[name][test]=torch.as_tensor(full_predictions[name][test_np],device=device)
        predictor_errors={name:((raw[test_np]-z.numpy()[test_np])/zs.cpu().numpy())**2 for name,raw in full_predictions.items()}
        predictor_metrics={name:dict(E_mse=float(err[:,:12].mean()),I_mse=float(err[:,12:].mean()),joint_mse=float(err.mean()),
            raw_MAE_per_axis=np.abs(full_predictions[name][test_np]-z.numpy()[test_np]).mean(0).tolist()) for name,err in predictor_errors.items()}
        predictor_comparisons={}
        for base,new in (('H','Ha'),('Shuffled','Ha'),('Ha','Ha_test_shuffled')):
            predictor_comparisons[f'{new}_vs_{base}']={key:cluster_gain(predictor_errors[base][:,axes].mean(1),predictor_errors[new][:,axes].mean(1),clusters[test_np],238)
                for key,axes in (('E',np.arange(12)),('I',np.arange(12,26)),('EI',np.arange(26)))}
        beta_test,_,_,rank=residual_fit(design[test_np],arms[test_np],z.numpy()[test_np],14)
        contrast_metrics={}; factual_coefficients={}; candidate_contrasts={}
        for name in ('H','Ha','Shuffled'):
            candidate=(candidate_arrays[name][:,1:]-candidate_arrays[name][:,:1]).mean(0)
            factual_coefficients[name]=residual_fit(design[test_np],arms[test_np],full_predictions[name][test_np],14)[0]
            candidate_contrasts[name]=candidate
            contrast_metrics[name]={key:contrast_scores(beta_test,candidate,zs.cpu().numpy(),axes) for key,axes in (('E',np.arange(12)),('I',np.arange(12,26)),('EI',np.arange(26)))}
        assert np.max(np.abs(candidate_contrasts['H']))==0
        # Independently fit source-wave half using outer-TRAIN only; forecast OTHER-half outer-TEST.
        crosshalf={}; half_candidates={}; half_gt={}; half_predictions={}
        for source in (False,True):
            fit_np=train_np[half[train_np]==source]; hold_np=test_np[half[test_np]!=source]
            fit_ids=torch.as_tensor(fit_np,device=device); hold_ids=torch.as_tensor(hold_np,device=device)
            key=f'half{int(source)}'; hh,ah,nh=preprocess(p,fit_ids,device)
            target,mean,scale=standardized(zg,fit_ids)
            counts=np.bincount(arms[fit_np],minlength=15)
            assert counts.min()>=1
            model,pred,record=fit(torch.cat((hh,ah),-1),target,fit_ids)
            records[key]=record; predictor_states[key]=cpu_state(model.state_dict())
            preprocess_records[key]=dict(fit_ids=fit_np,hold_ids=hold_np,normalizers=cpu_state(nh),target_mean=mean.cpu(),target_scale=scale.cpu(),source_arm_counts=counts.tolist())
            raw=(pred*scale+mean).cpu().numpy(); half_predictions[key]=raw[hold_np]
            err=((raw[hold_np]-z.numpy()[hold_np])/zs.cpu().numpy())**2
            gt_half,_,_,half_rank=residual_fit(design[hold_np],arms[hold_np],z.numpy()[hold_np],14)
            cand=candidates(model,hh,nh,mean,scale,hold_ids,True)
            effect=(cand[:,1:]-cand[:,:1]).mean(0); half_candidates[key]=effect; half_gt[key]=gt_half
            crosshalf[key]=dict(source_half=int(source),fit_windows=len(fit_np),test_windows=len(hold_np),rank=half_rank,
                test_arm_counts=np.bincount(arms[hold_np],minlength=15).tolist(),E_mse=float(err[:,:12].mean()),I_mse=float(err[:,12:].mean()),
                contrast_status='available' if half_rank==14 else 'UNCLEAR',
                I_contrasts=contrast_scores(gt_half,effect,zs.cpu().numpy(),np.arange(12,26)) if half_rank==14 else None)
            print(json.dumps(dict(crosshalf=key,I_mse=crosshalf[key]['I_mse'],rank=half_rank)),flush=True)
        mediated={name:(value-zm)/zs for name,value in oof.items()}
        zero=torch.zeros_like(zn)
        inputs=dict(H=task_slots(h,a,zero),Ha=task_slots(h,a,zero,True),GT_HEI=task_slots(h,a,zn),
            P_H=task_slots(h,a,mediated['H']),P_Ha=task_slots(h,a,mediated['Ha']),P_Shuffled=task_slots(h,a,mediated['Shuffled']),
            Ha_P_Ha=task_slots(h,a,mediated['Ha'],True))
        task_predictions={}; task_states={}; task_errors={}; task_metrics={}; downstream_models={}
        rank_groups=motion*4+np.minimum((p['context'][:,3].numpy()*4).astype(int),3)
        for name,x in inputs.items():
            model,pred,record=fit(x,yn,train); records[f'task_{name}']=record
            raw=(pred*ys+ym).cpu().numpy(); error=((raw-y.numpy())/ys.cpu().numpy())**2
            task_predictions[name]=raw; task_states[name]=cpu_state(model.state_dict()); task_errors[name]=error[test_np]; downstream_models[name]=model
            task_metrics[name]=dict(primary_mse=float(error[test_np][:,PRIMARY].mean()),all8_mse=float(error[test_np].mean()),physical_failure_mse=float(error[test_np,6].mean()),
                physical_failure_auc=dropout_auc(raw[test_np,6],y.numpy()[test_np,6]),test_normalized_mse_per_head=error[test_np].mean(0).tolist(),
                train_primary_mse=float(error[train_np][:,PRIMARY].mean()),ranking=supported_ranking(raw,y.numpy(),test_np,rank_groups,arms))
            print(json.dumps(dict(task=name,primary_mse=task_metrics[name]['primary_mse'])),flush=True)
        assert len({records[f'task_{name}']['initial_hash'] for name in inputs})==1
        shuffled_z=(torch.as_tensor(full_predictions['Ha_test_shuffled'],device=device)-zm)/zs
        with torch.no_grad():
            raw=(downstream_models['P_Ha'](task_slots(h,a,shuffled_z))*ys+ym).cpu().numpy()
        task_predictions['P_Ha_test_shuffled']=raw; task_errors['P_Ha_test_shuffled']=(((raw-y.numpy())/ys.cpu().numpy())**2)[test_np]
        comparisons={}
        for base,new in (('H','Ha'),('H','GT_HEI'),('H','P_Ha'),('Ha','P_Ha'),('Ha','Ha_P_Ha'),('P_H','P_Ha'),('P_Shuffled','P_Ha'),('P_Ha','P_Ha_test_shuffled')):
            comparisons[f'{new}_vs_{base}']={key:cluster_gain(task_errors[base][:,axes].mean(1),task_errors[new][:,axes].mean(1),clusters[test_np],238)
                for key,axes in (('primary',PRIMARY),('physical_failure',(6,)),('contact',(3,)))}
        retained=oracle_retention(task_errors['H'][:,PRIMARY].mean(1),task_errors['GT_HEI'][:,PRIMARY].mean(1),task_errors['P_Ha'][:,PRIMARY].mean(1),clusters[test_np])
        frozen=restored_model(old['models']['HEI'],162,8,device); oldnorm={key:value.to(device) for key,value in old['normalizers'].items()}
        frozen_metrics={}; frozen_predictions={}
        with torch.no_grad():
            for name in ('H','Ha','Shuffled'):
                # full outer representation is the exact serialized ref8 basis; raw OOF test equals full fit.
                x=task_slots(h,a,mediated[name]); raw=(frozen(x)*oldnorm['y_scale']+oldnorm['y_mean']).cpu().numpy()
                error=((raw-old['target'])/oldnorm['y_scale'].cpu().numpy())**2
                frozen_predictions[name]=raw; loss=float(error[test_np][:,PRIMARY].mean())
                old_h=((old['predictions']['H']-old['target'])/oldnorm['y_scale'].cpu().numpy())**2
                old_gt=((old['predictions']['HEI']-old['target'])/oldnorm['y_scale'].cpu().numpy())**2
                frozen_metrics[name]=dict(primary_mse=loss,old_oracle_R=float((old_h[test_np][:,PRIMARY].mean()-loss)/(old_h[test_np][:,PRIMARY].mean()-old_gt[test_np][:,PRIMARY].mean())))
        pc=predictor_comparisons; cm=contrast_metrics['Ha']['I']
        support=rank==14 and all(min(np.bincount(arms[train_np[folds[train_np]!=f]],minlength=15))>=1 for f in range(3))
        shuffle_ok=all(v['fit_changed']>=.6 and v['hold_changed']>=.6 for v in permutations.values())
        A=(all(pc['Ha_vs_H'][key]['gain']>=.05 and pc['Ha_vs_H'][key]['lower95']>0 for key in ('I','EI')) and
            pc['Ha_vs_Shuffled']['I']['gain']>=.05 and -pc['Ha_test_shuffled_vs_Ha']['I']['gain']>=.05 and shuffle_ok)
        B=(cm['correlation'] is not None and cm['correlation']>=.5 and cm['signed_agreement'] is not None and cm['signed_agreement']>=.65 and cm['gain_vs_zero']>=.1)
        transfer_options=[name for name in ('P_Ha','Ha_P_Ha') if comparisons[f'{name}_vs_Ha']['primary']['gain']>=.05 and
            comparisons[f'{name}_vs_Ha']['primary']['lower95']>0 and comparisons[f'{name}_vs_Ha']['physical_failure']['gain']>=-.02]
        oracle_ok=comparisons['GT_HEI_vs_H']['primary']['lower95']>0 and retained['R'] is not None and retained['R']>=.25 and retained['valid_bootstrap_fraction']>=.95
        C=bool(transfer_options and comparisons['P_Ha_vs_P_H']['primary']['gain']>=.03 and
            -comparisons['P_Ha_test_shuffled_vs_P_Ha']['primary']['gain']>=.02 and oracle_ok)
        fitting_limited=[name for name,record in records.items() if record['last200_fractional_change']>.05]
        overall='PROMISING' if support and A and B and C else ('UNPROMISING' if support else 'UNCLEAR')
        key_fitting_limited=[name for name in fitting_limited if not name.startswith('half')]
        if key_fitting_limited and overall=='UNPROMISING': overall='UNCLEAR'
        oof_quality={name:dict(train_oof_E_mse=float(((value[train,:12]-zg[train,:12])/zs[:12]).square().mean()),
            train_oof_I_mse=float(((value[train,12:]-zg[train,12:])/zs[12:]).square().mean()),
            oof_mean=value[train].mean(0).cpu().tolist(),oof_std=value[train].std(0,unbiased=False).cpu().tolist(),
            test_mean=value[test].mean(0).cpu().tolist(),test_std=value[test].std(0,unbiased=False).cpu().tolist()) for name,value in oof.items()}
        result=dict(status=overall,gates=dict(support=bool(support),A=bool(A),B=bool(B),C=bool(C),shuffle_support=bool(shuffle_ok)),
            predictor_metrics=predictor_metrics,predictor_comparisons=pc,contrasts=contrast_metrics,crosshalf=crosshalf,
            downstream_metrics=task_metrics,downstream_comparisons=comparisons,transfer_options=transfer_options,
            oracle_retention=retained,frozen_ref8_plugin=frozen_metrics,OOF_quality=oof_quality,
            fitting_limited=fitting_limited,key_fitting_limited=key_fitting_limited,
            split=dict(train=len(train_np),test=len(test_np),train_environments=len(set(clusters[train_np])),test_environments=len(set(clusters[test_np]))),
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device),elapsed_seconds=time.monotonic()-started,
            limitations='Fixed categorical action/PCA/MLP; extra physical supervision/compute; no individual causal mediation or policy utility',policy_executed=False)
        save('result.json',result); save('fit_records.json',records)
        torch.save(dict(train=train_np,test=test_np,folds=folds,clusters=clusters,waves=waves,motion=motion,arms=arms,half=half,
            z=z,y=y,normalizers=cpu_state(dict(**norm,z_mean=zm,z_scale=zs,y_mean=ym,y_scale=ys)),preprocess=preprocess_records,
            permutations=permutations,full_predictions=full_predictions,oof_predictions={name:value.cpu() for name,value in oof.items()},
            predictor_states=predictor_states,candidate_arrays=candidate_arrays,candidate_contrasts=candidate_contrasts,
            factual_contrasts=factual_coefficients,GT_contrasts=beta_test,design=design,groups=groups,
            crosshalf_candidates=half_candidates,crosshalf_GT=half_gt,crosshalf_predictions=half_predictions,
            downstream_states=task_states,task_predictions=task_predictions,frozen_plugin_predictions=frozen_predictions),args.run_dir/'diagnostic.pt')
        assert sha(args.dataset)==EXPECTED_SHA and all(sha(ROOT/key)==value for key,value in manifest['code_sha256'].items())
        manifest['run_status']='COMPLETED'
        print(json.dumps({key:result[key] for key in ('status','gates','oracle_retention','fitting_limited')},indent=2),flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',failure=f'{type(error).__name__}: {error}'); raise
    finally:
        manifest.update(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
        save('manifest.json',manifest)


if __name__=='__main__': main()
