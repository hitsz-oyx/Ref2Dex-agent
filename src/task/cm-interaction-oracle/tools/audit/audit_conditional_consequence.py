#!/usr/bin/env python3
"""Replay saved ref9 wiring on GPU; file/OLS/bootstrap statistics on CPU, no fit."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'src/task/cm-interaction-oracle/tools/run'))
from probe_conditional_consequence import preprocess, restored_model, candidates, cpu_state
from consequence_sufficiency import readouts, PRIMARY, cluster_gain
from conditional_consequence import task_slots, contrast_scores, oracle_retention
from probe_interventions import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True); parser.add_argument('--oracle-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args(); output=args.run_dir/'engineering_replay.json'
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(2); torch.cuda.set_device(0); device=torch.device('cuda:0')
    p=torch.load(args.dataset,map_location='cpu',weights_only=False)
    d=torch.load(args.run_dir/'diagnostic.pt',map_location='cpu',weights_only=False)
    old=torch.load(args.oracle_run/'diagnostic.pt',map_location='cpu',weights_only=False)
    r=json.loads((args.run_dir/'result.json').read_text()); m=json.loads((args.run_dir/'manifest.json').read_text())
    records=json.loads((args.run_dir/'fit_records.json').read_text())
    assert m['run_status']=='COMPLETED' and m['dataset_sha256']==sha(args.dataset)
    assert all(sha(ROOT/key)==value for key,value in m['code_sha256'].items())
    assert m['oracle_diagnostic_sha256']==sha(args.oracle_run/'diagnostic.pt')
    tr,te=d['train'],d['test']; folds=d['folds']; clusters=d['clusters']; arms=d['arms']; groups=d['groups']
    assert np.array_equal(tr,old['train']) and np.array_equal(te,old['test']) and not set(clusters[tr])&set(clusters[te])
    e,i,_,y,details=readouts(p); z=torch.cat((e,i),-1)
    assert torch.equal(z,d['z']) and torch.equal(y,d['y']) and int(details['early_failure'].sum())==193
    assert len(records)==21 and all(v['epochs']==1500 and len(v['train_loss'])==1500 for v in records.values())
    max_norm_error=0.; max_prediction_error=0.; max_candidate_error=0.; shuffle_checked={}
    normals=d['normalizers']; zm,zs=normals['z_mean'].to(device),normals['z_scale'].to(device)
    ym,ys=normals['y_mean'].to(device),normals['y_scale'].to(device)
    for prefix,prep in d['preprocess'].items():
        fit_np,hold_np=prep['fit_ids'],prep['hold_ids']
        assert not set(fit_np)&set(hold_np)
        if prefix.startswith('fold'):
            assert set(fit_np)<=set(tr) and set(hold_np)<=set(tr)
            assert not set(clusters[fit_np])&set(clusters[hold_np])
            f=int(prefix[-1]); assert np.array_equal(hold_np,tr[folds[tr]==f])
        if prefix.startswith('half'):
            source=int(prefix[-1]); assert (d['half'][fit_np]==bool(source)).all() and (d['half'][hold_np]!=bool(source)).all()
            assert set(fit_np)<=set(tr) and set(hold_np)<=set(te)
        assert min(np.bincount(arms[fit_np],minlength=15))>=1
        mean=z[fit_np].mean(0); scale=z[fit_np].std(0,unbiased=False).clamp_min(.001)
        max_norm_error=max(max_norm_error,float((mean-prep['target_mean']).abs().max()),float((scale-prep['target_scale']).abs().max()))
        # Recompute feature means/std in float64 CPU to independently check fold provenance.
        raw=torch.cat((p['history'].flatten(1),p['actor_obs'],p['context'],p['base_action']),-1).double()
        for names,value in ((('h_mean','h_scale'),raw),(('before_mean','before_scale'),p['before'].double()),
            (('a_mean','a_scale'),torch.nn.functional.one_hot(p['arm'],15)[:,1:].double())):
            mu=value[fit_np].mean(0); sig=value[fit_np].std(0,unbiased=False).clamp_min(.001)
            assert torch.allclose(mu,prep['normalizers'][names[0]].double(),atol=2e-5,rtol=2e-5)
            assert torch.allclose(sig,prep['normalizers'][names[1]].double(),atol=2e-5,rtol=2e-5)
        h,a,n=preprocess(p,torch.as_tensor(fit_np,device=device),device,prep['normalizers'])
        mean,scale=prep['target_mean'].to(device),prep['target_scale'].to(device)
        if prefix.startswith('half'):
            model=restored_model(d['predictor_states'][prefix],118,26,device)
            with torch.no_grad(): raw=(model(torch.cat((h,a),-1))*scale+mean).cpu().numpy()
            max_prediction_error=max(max_prediction_error,float(np.max(np.abs(raw[hold_np]-d['crosshalf_predictions'][prefix]))))
            assert r['crosshalf'][prefix]['contrast_status']=='UNCLEAR' and r['crosshalf'][prefix]['I_contrasts'] is None
            continue
        permutation=d['permutations'][prefix]['permutation']
        for ids in (fit_np,hold_np):
            assert set(permutation[ids])==set(ids) and np.array_equal(groups[permutation[ids]],groups[ids])
        shuffle_checked[prefix]={key:d['permutations'][prefix][key] for key in ('fit_changed','hold_changed')}
        for name in ('H','Ha','Shuffled'):
            model=restored_model(d['predictor_states'][f'{prefix}_{name}'],118,26,device)
            apad=torch.zeros_like(a) if name=='H' else (a[torch.as_tensor(permutation,device=device)] if name=='Shuffled' else a)
            with torch.no_grad(): raw=(model(torch.cat((h,apad),-1))*scale+mean).cpu().numpy()
            saved=d['full_predictions'][name] if prefix=='full' else d['oof_predictions'][name].numpy()
            max_prediction_error=max(max_prediction_error,float(np.max(np.abs(raw[hold_np]-saved[hold_np]))))
            if prefix=='full':
                cand=candidates(model,h,n,mean,scale,torch.as_tensor(hold_np,device=device),name!='H')
                max_candidate_error=max(max_candidate_error,float(np.max(np.abs(cand-d['candidate_arrays'][name]))))
                if name=='Ha':
                    with torch.no_grad(): permraw=(model(torch.cat((h,a[torch.as_tensor(permutation,device=device)]),-1))*scale+mean).cpu().numpy()
                    max_prediction_error=max(max_prediction_error,float(np.max(np.abs(permraw[te]-d['full_predictions']['Ha_test_shuffled'][te]))))
    assert max_norm_error<2e-5 and max_prediction_error<1e-4 and max_candidate_error<1e-4
    # All downstream train features must be exact held-fold predictions, not in-sample fits.
    full=d['preprocess']['full']; h,a,_=preprocess(p,torch.as_tensor(tr,device=device),device,full['normalizers'])
    zc=(z.to(device)-zm)/zs; med={name:(v.to(device)-zm)/zs for name,v in d['oof_predictions'].items()}
    zero=torch.zeros_like(zc)
    slots=dict(H=task_slots(h,a,zero),Ha=task_slots(h,a,zero,True),GT_HEI=task_slots(h,a,zc),
        P_H=task_slots(h,a,med['H']),P_Ha=task_slots(h,a,med['Ha']),P_Shuffled=task_slots(h,a,med['Shuffled']),
        Ha_P_Ha=task_slots(h,a,med['Ha'],True))
    task_error=0.
    with torch.no_grad():
        for name,x in slots.items():
            model=restored_model(d['downstream_states'][name],162,8,device)
            raw=(model(x)*ys+ym).cpu().numpy()
            task_error=max(task_error,float(np.max(np.abs(raw-d['task_predictions'][name]))))
        oldmodel=restored_model(old['models']['HEI'],162,8,device)
        for name in ('H','Ha','Shuffled'):
            raw=(oldmodel(task_slots(h,a,med[name]))*old['normalizers']['y_scale'].to(device)+old['normalizers']['y_mean'].to(device)).cpu().numpy()
            task_error=max(task_error,float(np.max(np.abs(raw-d['frozen_plugin_predictions'][name]))))
    assert task_error<2e-5
    errs={name:(((pred-y.numpy())/normals['y_scale'].numpy())**2)[te] for name,pred in d['task_predictions'].items()}
    comp_error=0.
    for name,c in r['downstream_comparisons'].items():
        new,base=name.split('_vs_')
        for metric,axes in (('primary',PRIMARY),('physical_failure',(6,)),('contact',(3,))):
            recalculated=cluster_gain(errs[base][:,axes].mean(1),errs[new][:,axes].mean(1),clusters[te],238)
            comp_error=max(comp_error,max(abs(recalculated[key]-c[metric][key]) for key in ('gain','lower95','upper95')))
    assert comp_error<1e-7
    rr=oracle_retention(errs['H'][:,PRIMARY].mean(1),errs['GT_HEI'][:,PRIMARY].mean(1),errs['P_Ha'][:,PRIMARY].mean(1),clusters[te])
    assert abs(rr['R']-r['oracle_retention']['R'])<1e-7
    # Direct joint-design OLS, distinct from production residualized regression.
    design=d['design'][te]; indicators=(arms[te,None]==np.arange(1,15)[None]).astype(float)
    coeff=np.linalg.lstsq(np.column_stack((design,indicators)),z.numpy()[te],rcond=1e-10)[0][-14:]
    ols_error=float(np.max(np.abs(coeff-d['GT_contrasts']))); assert ols_error<1e-8
    result=dict(status='PASS',dataset_sha256=sha(args.dataset),source_commit=m['git_commit'],
        fold_target_normalization_max_error=max_norm_error,predictor_replay_max_error=max_prediction_error,
        candidates_replay_max_error=max_candidate_error,downstream_and_frozen_replay_max_error=task_error,
        bootstrap_error=comp_error,direct_OLS_max_error=ols_error,shuffle=shuffle_checked,
        diagnostic_sha256=sha(args.run_dir/'diagnostic.pt'),audit_code_sha256=sha(__file__),
        device='GPU saved-model replay only; CPU file/OLS/bootstrap, no fit',
        limits='Engineering consistency does not establish fit generalization, mediation or policy utility')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__': main()
