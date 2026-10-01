#!/usr/bin/env python3
"""Grouped probability reliability setup with unchanged physical NN weights."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission


def run(args):
    admission=gpu_admission(args.gpu)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (str(args.gpu),admission['uuid']):raise ValueError('explicit admitted GPU')
    import torch
    from src.task.CmResidual.contact_ranker import physical_history
    from src.task.CmResidual.contact_trajectory import TrajectoryNetwork,all_trajectories,decode_trajectories,trajectory_targets,retained_choice
    from src.task.CmResidual.paired_evaluation import fingerprint
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0');started=time.monotonic()
    checkpoint=args.fit/'trajectory.pt'
    if sha(checkpoint)!='f7c0b0f95d2344947bd872e0ef518bc0fa49ebc592a9c154bb101f55ebdd22e8':raise ValueError('physical NN drift')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);nn_before=fingerprint(saved['models'])
    paths=[Path(p) for p in saved['source_record_sha256']]
    if any(sha(p)!=saved['source_record_sha256'][str(p)] for p in paths):raise ValueError('record drift')
    code=[Path(__file__),ROOT/'src/task/CmResidual/contact_trajectory.py',ROOT/'src/task/CmResidual/contact_ranker.py']
    inputs={str(p.resolve()):sha(p) for p in paths+[checkpoint,args.fit/'split.json']+code}
    base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    if args.output.resolve().parent!=base:raise ValueError('owned setup required')
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(experiment_id='P-20261001-contact-supported-height-control',family='HF10',probe_index_in_family=2,
                  run_status='RUNNING',stage='GROUPED_PROBABILITY_SETUP',pid=os.getpid(),command=sys.argv,gpu=admission,
                  input_sha256=inputs,neural_training=False,probability_parameter_fitting=True,actor_training=False,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    def save(): (args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    try:
        payload=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
        data={k:torch.cat([p[k] for p in payload]) for k in ['history','rest_z','candidate_actions','assignment','motion_id','start_frame','trigger','state','future_state','future_contact']}
        uniform=torch.cat([torch.full((len(p['assignment']),),p['schema']=='ref2dex.randomized_contact_consequence.v1',dtype=torch.bool) for p in payload])
        groups=[f'{int(m)}/{int(f)}' for m,f in zip(data['motion_id'],data['start_frame'])]
        split=torch.tensor(json.loads((args.fit/'split.json').read_text())['row_split']);fit=(split==0).nonzero().flatten();cal=(split==1).nonzero().flatten()
        target=trajectory_targets(data['future_state'],data['future_contact'].all(-1),data['state'][:,38],data['rest_z'])
        initial=(data['state'][:,38]-data['rest_z'])*1000
        generator=torch.Generator().manual_seed(9841);fold_mapping={}
        for motion in sorted(set(data['motion_id'][cal].tolist())):
            keys=sorted({groups[i] for i in cal.tolist() if int(data['motion_id'][i])==motion})
            if len(keys)<3:raise ValueError('insufficient calibration groups')
            for index,key in enumerate([keys[i] for i in torch.randperm(len(keys),generator=generator).tolist()]):fold_mapping[key]=index%3
        fold=torch.tensor([fold_mapping[groups[i]] for i in cal.tolist()],device=device)
        (args.output/'calibration_split.json').write_text(json.dumps(dict(seed=9841,frame_folds=fold_mapping),indent=2)+'\n')
        h=(physical_history(data['history'],data['rest_z']).to(device)-saved['history_mean'].to(device))/saved['history_scale'].to(device)
        candidate=(data['candidate_actions'].to(device)-saved['action_mean'].to(device))/saved['action_scale'].to(device)
        context=torch.cat((data['candidate_actions'][:,4],torch.nn.functional.one_hot(data['motion_id'].long(),3).float(),
                           ((data['start_frame']+data['trigger']).float()/600)[:,None]),-1).to(device)
        context=(context-saved['context_mean'].to(device))/saved['context_scale'].to(device)
        truth={k:target[k][cal].to(device) for k in ['joint_contact','release','supported_change_mm']}
        def metrics(probability,y,prior):
            bins=[];ece=0.
            for b in range(10):
                mask=(probability>=b/10)&(probability<(b+1)/10 if b<9 else probability<=1)
                if mask.any():
                    prediction=float(probability[mask].mean());observed=float(y[mask].mean())
                    ece+=float(mask.float().mean())*abs(prediction-observed)
                    bins.append(dict(rows=int(mask.sum()),prediction=prediction,observed=observed))
            return dict(brier=float((probability-y).square().mean()),constant_brier=float((prior-y).square().mean()),descriptive_ece=ece,bins=bins)
        def calibrate(logits,y):
            parameter=torch.nn.Parameter(torch.zeros(2,device=device,dtype=torch.float64));calls=0
            optimizer=torch.optim.LBFGS([parameter],lr=1,max_iter=100,line_search_fn='strong_wolfe')
            def closure():
                nonlocal calls
                calls+=1;optimizer.zero_grad()
                scale=parameter[0].exp();bias=parameter[1]
                probability=(logits.double()*scale+bias).sigmoid().mean(0).clamp(1e-8,1-1e-8)
                loss=torch.nn.functional.binary_cross_entropy(probability,y.double())+1e-4*((scale-1).square()+bias.square())
                if not torch.isfinite(loss):raise ValueError('nonfinite probability fit')
                loss.backward();return loss
            optimizer.step(closure)
            result=dict(scale=float(parameter[0].exp()),bias=float(parameter[1]),closure_calls=calls,max_iterations=100)
            if not .01<=result['scale']<=100 or not torch.isfinite(parameter).all():raise ValueError('unbounded probability setup')
            return result
        calibration={};parameters={};crossfit={}
        for variant,states in saved['models'].items():
            members=[]
            with torch.no_grad(),torch.random.fork_rng(devices=[]):
                for state in states:
                    model=TrajectoryNetwork(state_only=variant=='state_only').to(device);model.load_state_dict(state);model.eval().requires_grad_(False)
                    chunks=[]
                    for offset in range(0,len(cal),128):
                        rows=cal[offset:offset+128]
                        chunks.append(all_trajectories(model,h[rows],candidate[rows],context[rows]))
                    members.append(torch.cat(chunks))
            raw=torch.stack(members).detach();rows=torch.arange(len(cal),device=device);assignment=data['assignment'][cal].to(device)
            factual=raw[:,rows,assignment];parameters[variant]={};crossfit[variant]={}
            for key,index in [('joint_contact',20),('release',21)]:
                logits=factual[:,:,index];y=truth[key];cv=torch.empty_like(y);details=[]
                for f in range(3):
                    train=fold!=f;test=fold==f
                    theta=calibrate(logits[:,train],y[train]);cv[test]=(logits[:,test]*theta['scale']+theta['bias']).sigmoid().mean(0)
                    details.append(dict(fold=f,fit_rows=int(train.sum()),test_rows=int(test.sum()),fit_positive=int(y[train].sum()),test_positive=int(y[test].sum()),**theta))
                prior=float(target[key][fit].mean())
                crossfit[variant][key]=dict(**metrics(cv,y,prior),folds=details)
                parameters[variant][key]=calibrate(logits,y)
            decoded=decode_trajectories(raw,saved['height_mean'].to(device),saved['height_scale'].to(device),initial[cal].to(device)[None,:,None],parameters[variant])
            predicted=decoded['supported_change_mm'].mean(0)[rows,assignment]
            error=(predicted-truth['supported_change_mm']).abs();margin=max(.5,float(error.median()))
            choice,_=retained_choice(decoded,margin,saved['release_supported'],score_key='supported_change_mm')
            calibration[variant]=dict(margin_mm=margin,signed_score_mae_mm=float(error.mean()),signed_error_q90_mm=float(torch.quantile(error,.9)),
                proposal_fraction=float((choice!=4).float().mean()),choice_counts=torch.bincount(choice,minlength=6).tolist(),
                scope='whole historical cal thresholds; reliability measured by initial-frame crossfit')
            print(json.dumps(dict(variant=variant,crossfit=crossfit[variant],calibration=calibration[variant])),flush=True)
        means=[];uniform_fit=fit[uniform[fit]]
        for arm in range(6):means.append(float(target['supported_change_mm'][uniform_fit[data['assignment'][uniform_fit]==arm]].mean()))
        best_fixed=max(range(6),key=lambda a:means[a])
        contact=crossfit['cm']['joint_contact'];risk=crossfit['cm']['release']
        gate=dict(contact_brier=contact['brier']<=.8*contact['constant_brier'],contact_reliability=contact['descriptive_ece']<=.05,
                  release_brier=risk['brier']<=risk['constant_brier'],class_support=saved['release_supported'],
                  nonzero_proposal=calibration['cm']['proposal_fraction']>0)
        gate['passed']=all(gate.values())
        if fingerprint(saved['models'])!=nn_before:raise ValueError('physical NN weights changed')
        saved.update(probability_calibration=parameters,calibration=calibration,best_fixed=best_fixed,
                     score_key='supported_change_mm',setup_gate=gate,physical_nn_sha256=sha(checkpoint),neural_weights_unchanged=True)
        torch.save(saved,args.output/'calibrated_trajectory.pt')
        if time.monotonic()-started>300 or sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())>8<<30:raise ValueError('setup budget')
        if any(sha(Path(p))!=v for p,v in inputs.items()):raise ValueError('setup source drift')
        result=dict(run_status='COMPLETED',label='UNCLEAR',setup_gate=gate,crossfit=crossfit,calibration=calibration,
                    probability_calibration=parameters,best_fixed=best_fixed,uniform_fit_signed_support_means_mm=means,
                    calibrated_sha256=sha(args.output/'calibrated_trajectory.pt'),physical_nn_sha256=sha(checkpoint),
                    neural_weights_unchanged=True,input_hashes_unchanged=True,elapsed_seconds=time.monotonic()-started,
                    decision='FRESH_SELECTOR_POOL' if gate['passed'] else 'STOP_HF10_LOCAL_ADJUSTMENTS',
                    old_mae_gate='FAILED_PRESERVED',utility_boundary='no fresh physical control data yet')
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
        manifest.update(run_status='COMPLETED',setup_gate=gate,neural_weights_unchanged=True,input_hashes_unchanged=True)
    except BaseException as error:manifest.update(run_status='FAILED',error=repr(error));raise
    finally:manifest['elapsed_seconds']=time.monotonic()-started;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=4)
    run(p.parse_args())
