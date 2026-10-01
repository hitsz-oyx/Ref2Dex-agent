#!/usr/bin/env python3
"""Terminal mechanism audit; never fits models or selects new thresholds."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission


def run(args):
    admission=gpu_admission(args.gpu)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (str(args.gpu),admission['uuid']):
        raise ValueError('explicit idle GPU required')
    import torch
    from src.task.CmResidual.contact_ranker import physical_history,ConsequenceNetwork,all_predictions,policy_choice,randomized_value
    torch.set_num_threads(2);device=torch.device('cuda:0');started=time.monotonic()
    fit=args.fit;checkpoint=fit/'ranker.pt'
    model_hash=sha(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    report=json.loads((fit/'results.json').read_text())
    if report['ranker_sha256']!=model_hash:raise ValueError('checkpoint changed')
    paths=[Path(p) for p in saved['source_record_sha256']]
    if any(sha(p)!=saved['source_record_sha256'][str(p)] for p in paths):raise ValueError('data changed')
    payload=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
    keys=['history','candidate_actions','assignment','rest_z','motion_id','start_frame','trigger']
    data={k:torch.cat([p[k] for p in payload]) for k in keys}
    y=torch.cat([p['outcome']['supported_lift_mm'] for p in payload])
    eligible=torch.cat([p['outcome']['drop_eligible'] for p in payload])
    drop=torch.cat([p['outcome']['drop'] for p in payload])
    episodes=[e for p in payload for e in p['episode_id']]
    splits=torch.tensor(json.loads((fit/'split.json').read_text())['row_split'])
    h=physical_history(data['history'],data['rest_z'])
    h=((h-saved['history_mean'])/saved['history_scale']).to(device)
    actions=((data['candidate_actions']-saved['action_mean'])/saved['action_scale']).to(device)
    context=torch.cat((data['candidate_actions'][:,4],torch.nn.functional.one_hot(data['motion_id'].long(),3).float(),
                       ((data['start_frame']+data['trigger']).float()/600)[:,None]),-1)
    context=((context-saved['context_mean'].cpu())/saved['context_scale'].cpu()).to(device)
    output={'checkpoint_sha256':model_hash,'label':report['label'],'split_audits':{},
            'scope':'terminal descriptive audit, no refit or threshold search; no individual counterfactual labels'}
    def quantile(v):return dict(zip(['q0','q50','q90','q100'],torch.quantile(v.float(),torch.tensor([0.,.5,.9,1.])).tolist())) if len(v) else None
    for name,index in [('fit',0),('calibration',1),('holdout',2)]:
        rows=(splits==index).nonzero().flatten();variants={}
        assigned=data['assignment'][rows];target=y[rows]
        ep=[episodes[i] for i in rows.tolist()]
        with torch.no_grad():
            for variant,states in saved['models'].items():
                predictions=[]
                for state in states:
                    model=ConsequenceNetwork(state_only=variant=='state_only').to(device)
                    model.load_state_dict(state);model.eval()
                    chunks=[]
                    for start in range(0,len(rows),128):
                        r=rows[start:start+128]
                        raw=all_predictions(model,h[r],actions[r],context[r])
                        chunks.append(torch.stack(((raw[:,:,0]*saved['lift_scale'].to(device)+saved['lift_mean'].to(device)).clamp_min(0),
                                                   raw[:,:,1].sigmoid(),raw[:,:,2].sigmoid()),-1))
                    predictions.append(torch.cat(chunks))
                pred=torch.stack(predictions)
                choice,mean,lower=policy_choice(pred,saved['calibration'][variant]['margin_mm'],
                                               drop_supported=saved['drop_supported'],eligible=eligible[rows].to(device))
                choice=choice.cpu();mean=mean.cpu();lower=lower.cpu()
                intervene=choice!=4
                matched=(choice==assigned)&intervene;base_matched=(assigned==4)&intervene
                # Paired policy contrasts use the same randomized records.
                delta=6*((choice==assigned).float()-(assigned==4).float())*target
                effect=randomized_value(torch.zeros(len(rows),dtype=torch.long),torch.zeros(len(rows),dtype=torch.long),delta/6,ep)
                effects_by_group={}
                for i,r in enumerate(rows.tolist()):
                    group=f"{int(data['motion_id'][r])}/{int(data['start_frame'][r])}"
                    effects_by_group.setdefault(group,[]).append(i)
                avg=float(delta.mean());g=len(effects_by_group)
                sums=torch.tensor([float((delta[idx]-avg).sum()) for idx in effects_by_group.values()])
                se=float((sums.square().sum()*g/max(g-1,1)).sqrt())/len(rows)
                factual=mean[torch.arange(len(rows)),assigned,0]
                largest=delta.abs().argsort(descending=True)[:5]
                variants[variant]=dict(intervention_windows=int(intervene.sum()),choice_counts=torch.bincount(choice,minlength=6).tolist(),
                    policy_value_mm=randomized_value(choice,assigned,target,ep),effect_vs_base_mm=effect['value'],
                    paired_episode95=effect['approximate_episode_cluster95'],paired_frame_group95=[avg-1.96*se,avg+1.96*se],
                    selected_intervention_matches=int(matched.sum()),base_intervention_matches=int(base_matched.sum()),
                    selected_observed_mm=quantile(target[matched]),base_observed_mm=quantile(target[base_matched]),
                    predicted_selected_gain_mm=quantile((mean[torch.arange(len(rows)),choice,0]-mean[:,4,0])[intervene]),
                    predicted_pool_range_mm=quantile(mean[:,:,0].amax(-1)-mean[:,:,0].amin(-1)),
                    factual_rmse_mm=float((factual-target).square().mean().sqrt()),
                    eligible_policy_matches=int(((choice==assigned)&eligible[rows]).sum()),
                    eligible_policy_matched_drops=int(((choice==assigned)&eligible[rows]&drop[rows].bool()).sum()),
                    no_matched_drop_is_not_zero_risk=True,
                    largest_effect_contributions=[dict(row=int(rows[i]),episode=ep[i],assignment=int(assigned[i]),choice=int(choice[i]),
                        actual_lift_mm=float(target[i]),contribution_mm=float(delta[i])/len(rows)) for i in largest.tolist()])
        output['split_audits'][name]=dict(rows=len(rows),frame_groups=len(effects_by_group),variants=variants,
            outcome_mm=quantile(target))
    output.update(elapsed_seconds=time.monotonic()-started,input_hashes_unchanged=sha(checkpoint)==model_hash and
                  all(sha(p)==saved['source_record_sha256'][str(p)] for p in paths),gpu=admission)
    if args.output.exists():raise ValueError('audit output exists')
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=4);run(p.parse_args())
