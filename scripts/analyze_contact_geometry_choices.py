#!/usr/bin/env python3
"""Fixed propensity policy opportunity; no unseen candidate oracle/regret."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha
from analyze_orientation_feedback_opportunity import clustered_interval


def choose(predictions,reference):
    import torch
    score=predictions[...,1]*10
    point=score.mean(0);relative=score-score[:,:,reference,None]
    std=relative.std(0,unbiased=False)
    risk=torch.sigmoid(predictions[...,0]).mean(0)
    valid=(point-point[:,reference,None]>=1+1.645*std)&(risk<=risk[:,reference,None]+.05)
    best=point.masked_fill(~valid,-torch.inf).argmax(-1)
    choice=torch.where(valid.any(-1),best,torch.full_like(best,reference))
    return choice,dict(valid=valid,score_mm=point,relative_std_mm=std,predicted_loss=risk)


def support(policy,assignment,episode,group,take):
    match=take&(policy==assignment)
    return dict(matches=int(match.sum()),episodes=len(set(episode[match])),initial_groups=len(set(group[match])))


def run(args):
    begin=time.monotonic()
    if args.output.exists():raise ValueError('unique analysis output required')
    import torch
    from qualify_contact_geometry_source import load
    torch.set_num_threads(2)
    records,counts,adequate,source_hashes=load(args.source,args.source_audit)
    fit=json.loads((args.fit/'run_manifest.json').read_text());result=json.loads((args.fit/'results.json').read_text())
    if fit['run_status']!='COMPLETED' or result['run_status']!='COMPLETED':raise ValueError('terminal fit required')
    fit_audit=json.loads(args.fit_audit.read_text())
    if not fit_audit.get('audit_passed') or fit_audit['run_status']!='COMPLETED' or fit_audit['fit_manifest_sha256']!=sha(args.fit/'run_manifest.json'):
        raise ValueError('independent actual NN/label fit audit required')
    if any(sha(Path(k))!=v for k,v in fit['input_sha256'].items()):raise ValueError('fit source drift')
    bundle_path=args.fit/'candidate_predictions.pt'
    if sha(bundle_path)!=result['candidate_predictions_sha256']:raise ValueError('candidate prediction drift')
    bundle=torch.load(bundle_path,map_location='cpu',weights_only=False)
    if bundle['schema']!='ref2dex.contact_geometry_offline_candidates.v1' or bundle['source_manifest_sha256']!=sha(args.source/'run_manifest.json'):
        raise ValueError('fixed candidate source required')
    assignment=torch.cat([b['assignment'] for b in records]).numpy()
    propensity=torch.cat([b['propensity'] for b in records]).numpy()
    allocation=torch.cat([b['allocation'] for b in records]).numpy()
    probability=torch.cat([b['allocation_probabilities'] for b in records]).numpy()
    score=torch.cat([b['outcome']['supported_change_mm'] for b in records]).numpy()
    buckets=torch.cat([b['split_group_bucket'] for b in records]).numpy()
    initial=torch.cat([b['outcome']['initially_clear'] for b in records]).numpy()
    episode=np.array(sum([b['episode_id'] for b in records],[]))
    group=np.array([f'{int(i)}/{int(j)}' for b in records for i,j in zip(b['motion_id'],b['start_frame'])])
    risk=[];contact=[]
    for b in records:
        clear=b['future_clearance']>=.002
        was_clear=torch.cat(((b['initial_clearance']>=.002)[:,None],clear[:,:-1]),1).cummax(-1).values
        risk.append((was_clear&~clear).any(-1).float())
        contact.append(b['future_contact'][:,-3:].all(-1).all(-1).float())
    risk=torch.cat(risk).numpy();contact=torch.cat(contact).numpy()
    splits={'fit':buckets<50,'cal':(buckets>=50)&(buckets<70),'held':buckets>=70}
    fixed_values={str(k):float((((assignment==k)/propensity)*score)[splits['fit']].mean()) for k in (0,1)}
    reference=max((0,1),key=lambda k:fixed_values[str(k)])
    choices={mode:choose(pred,reference)[0].numpy() for mode,pred in bundle['predictions'].items()}
    n=len(score);choices.update(always_base=np.zeros(n,dtype=int),best_fixed=np.full(n,reference,dtype=int))
    # Predictions are actual NN outputs, checked independently by the fit audit; this check pins its source.
    if sha(args.fit/'contact_geometry_consequence.pt')!=result['checkpoint_sha256']:
        raise ValueError('model checkpoint drift')
    comparisons={};nulls={};policy_support={};coverage={};values={}
    for name,take in splits.items():
        if name=='fit':continue
        slots=(0,8) if reference==0 else (1,9)
        null_available=bool((take&(allocation==slots[0])).any() and (take&(allocation==slots[1])).any())
        null_values=((allocation==slots[0])/probability[:,slots[0]]-(allocation==slots[1])/probability[:,slots[1]])*score
        null=float(null_values[take].mean()) if null_available else None
        threshold=max(1.,2*abs(null)) if null_available else None
        nulls[name]=dict(reference=reference,slots=list(slots),difference_mm=null,
                         group90=clustered_interval(null_values[take],group[take],12652) if null_available else None,
                         required_gain_mm=threshold,available=null_available)
        policy_support[name]={k:support(v,assignment,episode,group,take) for k,v in choices.items()}
        coverage[name]=dict(changed_fixed=float((choices['cm'][take]!=reference).mean()),
                            changed_base=float((choices['cm'][take]!=0).mean()),
                            nonreference_windows=int((choices['cm'][take]!=reference).sum()))
        values[name]={k:float(((v==assignment)/propensity*score)[take].mean()) for k,v in choices.items()}
        comparisons[name]={}
        for control in ('state_only','shuffled','always_base','best_fixed','direct_score'):
            weights=(choices['cm']==assignment)/propensity-(choices[control]==assignment)/propensity
            difference=weights*score
            effect=float(difference[take].mean());group90=clustered_interval(difference[take],group[take],12653)
            ep90=clustered_interval(difference[take],episode[take],12654)
            adequate_pair=all(v['matches']>=24 and v['episodes']>=12 and v['initial_groups']>=8
                              for v in [policy_support[name]['cm'],policy_support[name][control]])
            risk_difference=float((weights*risk)[take].mean());contact_difference=float((weights*contact)[take].mean())
            gate=dict(support=adequate_pair,null_available=null_available,score=threshold is not None and effect>=threshold,
                      group90_positive=group90 is not None and group90[0]>0,
                      episode90_positive=ep90 is not None and ep90[0]>0,
                      loss_point_nonregression=risk_difference<=.02,joint_point_nonregression=contact_difference>=-.05,
                      changes_fixed10pct=coverage[name]['changed_fixed']>=.1)
            gate['passed']=all(gate.values())
            comparisons[name][control]=dict(score_difference_mm=effect,group90=group90,episode90=ep90,
                                             loss_difference=risk_difference,joint_difference=contact_difference,gate=gate)
    passed=adequate and result['gate']['passed'] and all(c['gate']['passed'] for c in comparisons['held'].values())
    all_support=adequate and all(c['gate']['support'] and c['gate']['null_available'] for c in comparisons['held'].values())
    label='PROMISING' if passed else ('UNPROMISING' if all_support else 'UNCLEAR')
    mean_pool={}
    for name,take in splits.items():
        mean_pool[name]=dict(base=values.get(name,{}).get('always_base'),cup=float(((assignment==1)/propensity*score)[take].mean()),
                             randomized_mixture_pool=float(((assignment>=2)/.6*score)[take].mean()))
    r=dict(experiment_id='P-20261002-contact-geometry-action-information',run_status='COMPLETED',label=label,
           information_label=result['label'],information_gate=result['gate'],best_fixed=reference,fit_fixed_values_mm=fixed_values,
           reference_null=nulls,policy_support=policy_support,coverage=coverage,ipw_values_mm=values,
           comparisons=comparisons,descriptive_pool_values_mm=mean_pool,counts=counts,
           elapsed_seconds=time.monotonic()-begin,prior_cumulative_seconds=fit_audit['cumulative_seconds'],
           input_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),args.fit/'run_manifest.json',args.fit/'results.json',bundle_path,
                                                           args.fit/'contact_geometry_consequence.pt',args.source/'run_manifest.json',args.source_audit,args.fit_audit]},
           scope='fixed held randomized candidate IPW opportunity only; no individual counterfactual ranking/regret, actual replan utility, learned actor or stable grasp claim')
    args.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('run_status','label','information_label','best_fixed','reference_null','coverage','comparisons')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','source-audit','fit','fit-audit','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
