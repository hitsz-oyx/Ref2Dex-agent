#!/usr/bin/env python3
"""Fixed HF21 actual-policy opportunity, risk, null and physical actuation."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha
from analyze_orientation_feedback_opportunity import clustered_interval
from structured_opportunity_statistics import contributions, duplicate_contrast


def pd_targets(actions,position,offset,scale):
    """Independent native linear map and actual dependent-joint couplings."""
    raw=actions.copy();raw[...,6:]=(raw[...,6:]+1)*.5
    targets=offset+scale*raw
    targets[...,:6]+=position[...,:6]
    for dst,src,ratio in ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)):
        targets[...,dst]=targets[...,src]*ratio
    return targets


def arrays(record):
    def value(key):return record[key].numpy()
    q=np.concatenate((value('state')[:,None,:18],value('future_state')[:,:-1,:18]),1)
    offset,scale=value('pd_offset'),value('pd_scale')
    cup=value('expert_bank')[:,:,1].copy()
    cup[:,:,3:6]=np.clip((value('hold_target')[:,None,3:6]-q[:,:,3:6]-offset[3:6])/scale[3:6],-1,1)
    cup_pd=pd_targets(cup,q,offset,scale)
    executed_delta=np.max(np.abs(value('actual_pd_targets')-cup_pd),axis=-1)
    candidate_pd=pd_targets(value('candidate_actions'),value('state')[:,None,:18],offset,scale)
    proposed_delta=np.max(np.abs(candidate_pd[:,2]-candidate_pd[:,1]),axis=-1)
    clearance=value('future_clearance')>=.002
    past=np.concatenate(((value('initial_clearance')>=.002)[:,None],clearance[:,:-1]),1)
    past=np.maximum.accumulate(past,axis=-1)
    return dict(mu=value('nuisance_forecasts').astype(np.float64).mean(1)[:,:5],
        trigger=value('trigger'),
        assignment=value('assignment'),propensity=value('propensity').astype(np.float64),allocation=value('allocation'),
        probability=value('allocation_probabilities').astype(np.float64),bucket=value('split_group_bucket'),
        episode=np.array(record['episode_id']),
        group=np.array([f'{int(m)}/{int(s)}' for m,s in zip(record['motion_id'],record['start_frame'])]),
        score=record['outcome']['supported_change_mm'].numpy().astype(np.float64),
        initial=record['outcome']['initially_clear'].numpy(),
        risk=np.any(past&~clearance,axis=-1).astype(float),
        contact=np.all(value('future_contact')[:,-3:],axis=(-1,-2)).astype(float),
        changed_cup=proposed_delta>1e-5,
        actual_changed_cup=np.max(executed_delta,axis=-1)>1e-5,
        actual_changed_cup_steps=np.sum(executed_delta>1e-5,axis=-1))


def support(match,take,data):
    rows=match&take
    return dict(matches=int(rows.sum()),episodes=len(set(data['episode'][rows])),
                initial_groups=len(set(data['group'][rows])))


def run(args):
    started=time.monotonic()
    if args.output.exists():raise ValueError('unique analysis artifact required')
    manifest_path=args.source/'run_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest['run_status']!='COMPLETED' or manifest.get('child_exit_code')!=0:
        raise ValueError('authoritative complete fixed source panel required')
    if [p['seed'] for p in manifest['phases']]!=list(range(611,619)):
        raise ValueError('fixed8seeds required; no partial panel')
    if any(sha(Path(k))!=v for k,v in manifest['input_sha256'].items()):
        raise ValueError('source input drift')
    import torch
    torch.set_num_threads(2)
    parts=[];hashes={str(manifest_path.resolve()):sha(manifest_path)}
    for phase in manifest['phases']:
        if phase['run_status']!='COMPLETED' or phase.get('native_exit_code')!=0 or phase.get('audit_exit_code')!=0:
            raise ValueError('native and independent full audit terminal required')
        directory=Path(phase['directory']);record_path=directory/'records.pt';planning=directory/'planning.pt'
        audit_path=Path(phase['audit']);audit=json.loads(audit_path.read_text())
        control=args.source/'phase-controls'/f"seed{phase['seed']}"/'run_manifest.json'
        if sha(record_path)!=phase['result']['record_sha256'] or sha(planning)!=phase['result']['planning_sha256']:
            raise ValueError('actual records/planning hash')
        if sha(audit_path)!=phase['audit_sha256'] or audit['run_status']!='COMPLETED':
            raise ValueError('independent audit hash')
        if audit['run_manifest_sha256']!=sha(control) or len(audit['phases'])!=1:
            raise ValueError('audit of this actual completed phase')
        record=torch.load(record_path,map_location='cpu',weights_only=False)
        if record['schema']!='ref2dex.structured_contact_opportunity_source.v1' or record['seed']!=phase['seed']:
            raise ValueError('truthful source schema/seed')
        if record['future_done'].any() or len(record['state'])!=audit['phases'][0]['physical']['rows']:
            raise ValueError('complete independently accepted rows')
        parts.append(arrays(record))
        for path in (record_path,planning,audit_path,control):hashes[str(path.resolve())]=sha(path)
    data={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    n=len(data['score']);chosen=data['assignment'];p=data['propensity']
    splits=dict(fit=data['bucket']<50,cal=(data['bucket']>=50)&(data['bucket']<70),held=data['bucket']>=70)
    outcomes=np.stack((data['score'],data['risk'],data['contact']),-1)
    aipw=contributions(chosen,outcomes,data['mu'])
    fixed_values={str(a):float(aipw[splits['fit'],a,0].mean()) for a in (0,1)}
    reference=max((0,1),key=lambda a:fixed_values[str(a)])
    matches=dict(cm=chosen==2,state_only=chosen==1,shuffled=chosen==4,direct_score=chosen==3,
                 always_base=chosen==0,best_fixed=chosen==reference)
    weights={k:v.astype(float)/p for k,v in matches.items()}
    arms_by_name=dict(cm=2,state_only=1,shuffled=4,direct_score=3,always_base=0,best_fixed=reference)
    augmented={k:aipw[:,a] for k,a in arms_by_name.items()}
    raw_ht={}; repeats={}
    counts={};values={};policy_support={};nulls={};coverage={};comparisons={}
    for name,take in splits.items():
        arms=np.bincount(chosen[take],minlength=8).tolist()
        count=dict(rows=int(take.sum()),episodes=len(set(data['episode'][take])),
                   initial_groups=len(set(data['group'][take])),arms=arms,
                   early=int((take&~data['initial']).sum()),initially_clear=int((take&data['initial']).sum()))
        count['adequate']=(count['rows']>=150 and count['episodes']>=32 and count['initial_groups']>=8
                           and all(arms[a]>=24 for a in range(5)))
        counts[name]=count
        if name=='fit':continue
        policy_support[name]={k:support(v,take,data) for k,v in matches.items()}
        values[name]={k:float(v[take,0].mean()) for k,v in augmented.items()}
        raw_ht[name]=dict(values_mm={k:float((v*data['score'])[take].mean()) for k,v in weights.items()},comparisons={})
        ep, epcounts=np.unique(data['episode'][take],return_counts=True)
        repeats[name]=dict(episodes_with_replanning=int((epcounts>=2).sum()),maximum_windows_per_episode=int(epcounts.max()),
            subsequent_windows=int(np.maximum(epcounts-1,0).sum()))
        null_pair={}
        for key,arm,left,right in (('base',0,(0,),(1,)),('cup',1,(2,3),(4,5))):
            available=all(np.any(take&np.isin(data['allocation'],slots)) for slots in (left,right))
            difference=duplicate_contrast(data['allocation'],outcomes,data['mu'],arm,left,right)[:,0]
            null_pair[key]=dict(slots=[list(left),list(right)],available=bool(available),
                difference_mm=float(difference[take].mean()) if available else None,
                group90=clustered_interval(difference[take],data['group'][take],12752) if available else None)
        available=all(v['available'] for v in null_pair.values())
        threshold=max(1.,*(2*abs(v['difference_mm']) for v in null_pair.values())) if available else None
        nulls[name]=dict(pairs=null_pair,required_gain_mm=threshold,available=available,
                        scope='duplicate-slot random allocation differences, not exact-state simulator replay')
        actual=take&matches['cm'];changed=actual&data['actual_changed_cup']
        coverage[name]=dict(proposed_cup_pd_change_fraction=float(data['changed_cup'][take].mean()),
            actual_cm_windows=int(actual.sum()),actual_non_cup_pd_windows=int(changed.sum()),
            actual_non_cup_pd_steps=int(data['actual_changed_cup_steps'][actual].sum()),
            actual_cm_total_steps=int(actual.sum())*10,
            actual_non_cup_fraction=float(changed.sum()/actual.sum()) if actual.any() else None)
        comparisons[name]={}
        for control in ('state_only','shuffled','always_base','best_fixed','direct_score'):
            w=weights['cm']-weights[control]
            difference=(augmented['cm']-augmented[control])[:,0]
            score=float(difference[take].mean())
            risk=float((augmented['cm']-augmented[control])[take,1].mean())
            joint=float((augmented['cm']-augmented[control])[take,2].mean())
            raw_ht[name]['comparisons'][control]=dict(score_difference_mm=float((w*data['score'])[take].mean()),
                loss_difference=float((w*data['risk'])[take].mean()),joint_difference=float((w*data['contact'])[take].mean()))
            group_ci=clustered_interval(difference[take],data['group'][take],12753)
            episode_ci=clustered_interval(difference[take],data['episode'][take],12754)
            sufficient=all(s['matches']>=24 and s['episodes']>=12 and s['initial_groups']>=8
                           for s in (policy_support[name]['cm'],policy_support[name][control]))
            gate=dict(support=sufficient,null_available=available,
                score=threshold is not None and score>=threshold,
                group90_positive=group_ci is not None and group_ci[0]>0,
                episode90_positive=episode_ci is not None and episode_ci[0]>0,
                loss_point_nonregression=risk<=.02,joint_point_nonregression=joint>=-.05,
                proposed_physical_change10pct=coverage[name]['proposed_cup_pd_change_fraction']>=.1,
                actual_non_cup_windows12=coverage[name]['actual_non_cup_pd_windows']>=12)
            gate['passed']=all(gate.values())
            comparisons[name][control]=dict(score_difference_mm=score,group90=group_ci,episode90=episode_ci,
                loss_difference=risk,joint_difference=joint,gate=gate)
    adequate=all(v['adequate'] for v in counts.values())
    support_ok=adequate and all(v['gate']['support'] and v['gate']['null_available'] for split in ('cal','held') for v in comparisons[split].values())
    passed=adequate and all(v['gate']['passed'] for split in ('cal','held') for v in comparisons[split].values())
    label='PROMISING' if passed else ('UNPROMISING' if support_ok else 'UNCLEAR')
    hashes[str(Path(__file__).resolve())]=sha(Path(__file__))
    hashes[str((ROOT/'scripts/structured_opportunity_statistics.py').resolve())]=sha(ROOT/'scripts/structured_opportunity_statistics.py')
    hashes[str((ROOT/'scripts/analyze_orientation_feedback_opportunity.py').resolve())]=sha(ROOT/'scripts/analyze_orientation_feedback_opportunity.py')
    result=dict(experiment_id='P-20261002-structured-contact-opportunity',run_status='COMPLETED',label=label,
        fixed_reference=reference,fit_fixed_values_mm=fixed_values,counts=counts,source_support_adequate=adequate,
        policy_support=policy_support,coverage=coverage,reference_null=nulls,aipw_values_mm=values,comparisons=comparisons,raw_ht=raw_ht,replanning=repeats,
        estimator='pre-fixed old-fit ensemble AIPW; residual corrected by known allocation probability',
        elapsed_seconds=time.monotonic()-started,prior_cumulative_seconds=manifest['cumulative_seconds'],
        input_sha256=hashes,device='cpu',device_reason='pure physical label/actuation reconstruction and IPW statistics; no neural computations',
        scope='known randomized local H10 response over the repeated-observation behavior-state distribution; no full closed-loop policy causal value, learned actor or stable grasp claim')
    if result['elapsed_seconds']+manifest['cumulative_seconds']>3600:
        raise TimeoutError('fixed whole slot analysis budget')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('run_status','label','fixed_reference','counts','coverage','reference_null','comparisons')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
