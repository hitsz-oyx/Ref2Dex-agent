#!/usr/bin/env python3
"""GPU-only fixed-budget consequence learning with untouched held frame groups."""
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
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (str(args.gpu),admission['uuid']):
        raise ValueError('explicit admitted single GPU required')
    import torch
    from src.task.CmResidual.contact_ranker import (
        frame_group_split,physical_history,ConsequenceNetwork,all_predictions,
        shuffled_numeric_actions,policy_choice,randomized_value,
    )
    from src.task.CmResidual.physical_value_contract import private_initialization
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0');started=time.monotonic()
    collection=json.loads((args.collection/'run_manifest.json').read_text())
    if collection['run_status']!='COMPLETED':raise ValueError('collection not terminal')
    paths=[Path(p['directory'])/'records.pt' for p in collection['phases']]
    for phase,path in zip(collection['phases'],paths):
        if sha(path)!=phase['record_sha256']:raise ValueError('collection payload drift')
    sources=[Path(__file__),ROOT/'src/task/CmResidual/contact_ranker.py']
    inputs={str(p.resolve()):sha(p) for p in paths+sources}
    if args.output.resolve().parent!=args.collection.resolve():raise ValueError('model output must belong to collection')
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(run_status='RUNNING',pid=os.getpid(),command=sys.argv,gpu=admission,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  input_sha256=inputs,actor_training=False,value_training=False,
                  variants=['cm','state_only','action_shuffled'],updates_per_model=1000,ensemble_size=3,
                  wall_budget_seconds=3600,collection_seconds=collection['cumulative_seconds'])
    def save(): (args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if collection['cumulative_seconds']+time.monotonic()-started>3540:raise TimeoutError('collection+fit budget')
        if sum(p.stat().st_size for p in args.collection.rglob('*') if p.is_file())>8<<30:raise ValueError('storage budget')
        if any(sha(p)!=inputs[str(p.resolve())] for p in sources):raise ValueError('training source changed')
    save()
    try:
        payloads=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
        tensor_keys=['state','history','candidate_actions','assignment','propensity','rest_z','motion_id','start_frame','trigger']
        data={k:torch.cat([p[k] for p in payloads]) for k in tensor_keys}
        data['outcome']={k:torch.cat([p['outcome'][k] for p in payloads]) for k in payloads[0]['outcome']}
        episodes=[e for p in payloads for e in p['episode_id']]
        if not torch.allclose(data['propensity'],torch.full_like(data['propensity'],1/6)):raise ValueError('propensity drift')
        split,mapping=frame_group_split(data['motion_id'],data['start_frame'])
        split_record=dict(seed=9331,row_split=split.tolist(),group_split={f'{m}/{f}':v for (m,f),v in mapping.items()},
                          episode_ids=episodes,input_sha256={str(p):sha(p) for p in paths})
        (args.output/'split.json').write_text(json.dumps(split_record,indent=2)+'\n')
        split_stats={}
        for name,index in [('fit',0),('calibration',1),('holdout',2)]:
            mask=split==index; ep={episodes[i] for i in mask.nonzero().flatten().tolist()}
            counts=torch.bincount(data['assignment'][mask],minlength=6)
            stats=dict(rows=int(mask.sum()),episodes=len(ep),arms=counts.tolist(),
                       drop_eligible=int(data['outcome']['drop_eligible'][mask].sum()),
                       drops=int(data['outcome']['drop'][mask].sum()))
            split_stats[name]=stats
            if len(ep)<24 or len(set(data['motion_id'][mask].tolist()))!=3:raise ValueError('insufficient held frame/episode coverage')
            if index in (0,2) and int(counts.min())<15:raise ValueError('insufficient factual arm support')
        print(json.dumps(dict(split_stats=split_stats)),flush=True)
        fit,cal,hold=[(split==i).nonzero().flatten() for i in range(3)]
        h=physical_history(data['history'],data['rest_z'])
        mean=h[fit].mean((0,1));scale=h[fit].std((0,1),unbiased=False).clamp_min(.001)
        h=((h-mean)/scale).to(device)
        candidate=data['candidate_actions'].to(device)
        observed_action=candidate[fit,data['assignment'][fit]]
        action_mean=observed_action.mean(0)
        action_scale=observed_action.std(0,unbiased=False).clamp_min(.001)
        candidate=(candidate-action_mean)/action_scale
        # Base continuation depends on policy intent and reference phase.
        # These are pre-action fields, available equally to every predictor.
        context=torch.cat((data['candidate_actions'][:,4],torch.nn.functional.one_hot(data['motion_id'].long(),3).float(),
                           ((data['start_frame']+data['trigger']).float()/600)[:,None]),-1)
        context_mean=context[fit].mean(0);context_scale=context[fit].std(0,unbiased=False).clamp_min(.001)
        context=((context-context_mean)/context_scale).to(device)
        assignment=data['assignment'].long().to(device)
        target=torch.stack((data['outcome']['supported_lift_mm'],data['outcome']['contact_fraction'],data['outcome']['drop'].float()),-1).to(device)
        eligible=data['outcome']['drop_eligible'].to(device)
        lift_mean=target[fit,0].mean();lift_scale=target[fit,0].std(unbiased=False).clamp_min(.5)
        corrupted=shuffled_numeric_actions(candidate[fit],assignment[fit],seed=9431)
        drop_supported=(split_stats['fit']['drop_eligible']>=20 and split_stats['fit']['drops']>=3 and
                        split_stats['calibration']['drop_eligible']>=10 and split_stats['calibration']['drops']>=2)
        models={};states={};calibration={};choices={};metrics={}
        fits=fit.to(device)
        fit_groups=sorted({(int(data['motion_id'][i]),int(data['start_frame'][i])) for i in fit.tolist()})
        group_index=torch.tensor([fit_groups.index((int(data['motion_id'][i]),int(data['start_frame'][i]))) for i in fit.tolist()],device=device)
        def decode(raw):
            return torch.stack(((raw[:,:,0]*lift_scale+lift_mean).clamp_min(0),raw[:,:,1].sigmoid(),raw[:,:,2].sigmoid()),-1)
        def predict(pool,indices):
            pieces=[]
            for model in pool:
                chunks=[]
                for offset in range(0,len(indices),128):
                    rows=indices[offset:offset+128].to(device)
                    chunks.append(decode(all_predictions(model,h[rows],candidate[rows],context[rows])))
                pieces.append(torch.cat(chunks))
            return torch.stack(pieces)
        for variant in ('cm','state_only','action_shuffled'):
            pool=[]
            for member in range(3):
                check()
                with private_initialization(9461+member):model=ConsequenceNetwork(state_only=variant=='state_only').to(device)
                optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                generator=torch.Generator(device=device).manual_seed(9561+member)
                sampled_groups=torch.randint(len(fit_groups),(len(fit_groups),),generator=generator,device=device)
                group_weight=torch.bincount(sampled_groups,minlength=len(fit_groups)).float()
                row_weight=group_weight[group_index]
                for update in range(1000):
                    local=torch.multinomial(row_weight,64,replacement=True,generator=generator)
                    rows=fits[local]
                    if variant=='state_only':prediction=model(h[rows],context=context[rows])[torch.arange(len(rows),device=device),assignment[rows]]
                    else:
                        action=corrupted[local] if variant=='action_shuffled' else candidate[rows,assignment[rows]]
                        prediction=model(h[rows],action,context=context[rows])
                    loss=torch.nn.functional.smooth_l1_loss(prediction[:,0],(target[rows,0]-lift_mean)/lift_scale)
                    loss+=torch.nn.functional.binary_cross_entropy_with_logits(prediction[:,1],target[rows,1])
                    risk=eligible[rows]
                    if risk.any():loss+=torch.nn.functional.binary_cross_entropy_with_logits(prediction[risk,2],target[rows[risk],2])
                    if not torch.isfinite(loss):raise ValueError('nonfinite consequence training')
                    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                    if update%250==0:check()
                model.eval().requires_grad_(False);pool.append(model)
                print(json.dumps(dict(variant=variant,member=member,updates=1000,last_loss=float(loss))),flush=True)
            models[variant]=pool;states[variant]=[{k:v.detach().cpu() for k,v in m.state_dict().items()} for m in pool]
            with torch.no_grad():
                cp=predict(pool,cal);hp=predict(pool,hold)
                cm=cp.mean(0);hm=hp.mean(0)
                factual_cal=cm[torch.arange(len(cal),device=device),assignment[cal]]
                factual_hold=hm[torch.arange(len(hold),device=device),assignment[hold]]
                error=(factual_cal[:,0]-target[cal,0]).abs()
                margin=max(.5,float(error.median()))
                contact_mae=float((factual_cal[:,1]-target[cal,1]).abs().mean())
                cal_risk=eligible[cal]
                risk_brier=float((factual_cal[cal_risk,2]-target[cal[cal_risk.cpu()],2]).square().mean()) if cal_risk.any() else None
                q90=float(torch.quantile(error,.9))
                calibration[variant]=dict(margin_mm=margin,supported_lift_absolute_q90_mm=q90,
                                          factual_contact_mae=contact_mae,conditional_drop_brier=risk_brier,
                                          drop_supported=drop_supported,scope='factual calibration; effect dispersion heuristic')
                choice,_,_=policy_choice(hp,margin,drop_supported=drop_supported,eligible=eligible[hold])
                choices[variant]=choice.cpu()
                risk=eligible[hold]
                metrics[variant]=dict(factual_lift_rmse_mm=float((factual_hold[:,0]-target[hold,0]).square().mean().sqrt()),
                    factual_contact_mae=float((factual_hold[:,1]-target[hold,1]).abs().mean()),
                    factual_drop_brier=float((factual_hold[risk,2]-target[hold[risk.cpu()],2]).square().mean()) if risk.any() else None,
                    lift_interval_factual_coverage=float(((factual_hold[:,0]-target[hold,0]).abs()<=q90).float().mean()))
        means=[]
        for arm in range(6):means.append(float(target[fit[data['assignment'][fit]==arm],0].mean()))
        best_fixed=max(range(6),key=lambda a:means[a])
        choices['always_base']=torch.full((len(hold),),4,dtype=torch.long)
        choices['best_fixed']=torch.full((len(hold),),best_fixed,dtype=torch.long)
        held_episodes=[episodes[i] for i in hold.tolist()]
        policy={}
        for name,choice in choices.items():
            policy[name]={key:randomized_value(choice,data['assignment'][hold],data['outcome'][key][hold].float(),held_episodes)
                          for key in ('supported_lift_mm','contact_fraction','drop')}
            policy[name]['intervention_fraction']=float((choice!=4).float().mean())
            policy[name]['choice_counts']=torch.bincount(choice,minlength=6).tolist()
        cm_value=policy['cm'];controls=[k for k in policy if k!='cm']
        effects={k:cm_value['supported_lift_mm']['value']-policy[k]['supported_lift_mm']['value'] for k in controls}
        contact_ok=all(cm_value['contact_fraction']['value']>=policy[k]['contact_fraction']['value']-.05 for k in controls)
        drop_ok=all(cm_value['drop']['value']<=policy[k]['drop']['value']+.05 for k in controls)
        matched=cm_value['supported_lift_mm']
        supported=(matched['matched_windows']>=32 and matched['matched_episodes']>=10)
        coverage=.05<=cm_value['intervention_fraction']<=.80
        calibration_ok=(calibration['cm']['factual_contact_mae']<=.20 and metrics['cm']['lift_interval_factual_coverage']>=.80 and drop_supported)
        passed=min(effects.values())>=.5 and contact_ok and drop_ok and supported and coverage and calibration_ok
        payload=dict(schema='ref2dex.contact_ranker.v1',models=states,history_mean=mean.cpu(),history_scale=scale.cpu(),
                     lift_mean=lift_mean.cpu(),lift_scale=lift_scale.cpu(),calibration=calibration,
                     context_mean=context_mean,context_scale=context_scale,
                     action_mean=action_mean.cpu(),action_scale=action_scale.cpu(),
                     group_split=split_record['group_split'],best_fixed=best_fixed,drop_supported=drop_supported,
                     source_record_sha256={str(p):sha(p) for p in paths},execution='cached candidate2 + base8',
                     frozen_expert_route='src/task/CmResidual/configs/hf02_temporal_canonical_route.json')
        torch.save(payload,args.output/'ranker.pt')
        report=dict(run_status='COMPLETED',label='UNCLEAR' if not supported or not drop_supported else ('PROMISING' if passed else 'UNPROMISING'),
                    decision='DIRECT_REPLANNING_PROBE' if passed else 'REVIEW_PHYSICAL_RANKING',split_stats=split_stats,
                    factual_metrics=metrics,calibration=calibration,policy=policy,best_fixed=best_fixed,
                    effects_mm=effects,gate=dict(passed=passed,contact=contact_ok,drop=drop_ok,
                    support=supported,coverage=coverage,calibration=calibration_ok),
                    true_individual_regret='unavailable in randomized single-action data',baseline_status='PARTIAL',cm_utility='UNPROVEN')
        check()
        if any(sha(Path(p))!=v for p,v in inputs.items()):raise ValueError('training input mutated')
        report.update(elapsed_seconds=time.monotonic()-started,cumulative_probe_seconds=collection['cumulative_seconds']+time.monotonic()-started,
                      input_hashes_unchanged=True,ranker_sha256=sha(args.output/'ranker.pt'))
        (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        manifest.update(run_status='COMPLETED',label=report['label'],decision=report['decision'],input_hashes_unchanged=True)
        print(json.dumps(report),flush=True)
    except BaseException as error:manifest.update(run_status='FAILED',error=repr(error));raise
    finally:manifest['elapsed_seconds']=time.monotonic()-started;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=4)
    run(p.parse_args())
