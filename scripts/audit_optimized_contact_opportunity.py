#!/usr/bin/env python3
"""Independent float64 policy/null/bootstrap and native-amplitude result audit."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha


def interval(samples,groups,seed):
    # Dictionary grouping and vectorized cluster draws, independent of analysis helper.
    keys=sorted(set(groups));index={key:i for i,key in enumerate(keys)}
    totals=np.zeros(len(keys));counts=np.zeros(len(keys))
    for value,key in zip(samples,groups):
        totals[index[key]]+=value;counts[index[key]]+=1
    if len(keys)<2:return None
    draws=np.random.default_rng(seed).integers(0,len(keys),size=(1000,len(keys)))
    estimates=totals[draws].sum(axis=1)/counts[draws].sum(axis=1)
    return np.quantile(estimates,[.05,.95]).tolist()


def target(action,q,offset,scale):
    output=np.empty(18,dtype=np.float32)
    for j in range(18):
        value=action[j] if j<6 else np.float32((np.float32(1)+action[j])*np.float32(.5))
        output[j]=np.float32(offset[j]+np.float32(scale[j]*value))
        if j<6:output[j]=np.float32(output[j]+q[j])
    output[7]=np.float32(output[6]*np.float32(1.05))
    output[9]=np.float32(output[8]*np.float32(1.05))
    output[11]=np.float32(output[10]*np.float32(1.05))
    output[13]=np.float32(output[12]*np.float32(1.05))
    output[16]=np.float32(output[15]*np.float32(.6))
    output[17]=np.float32(output[15]*np.float32(.8))
    return output


def run(args):
    started=time.monotonic()
    if args.output.exists():raise ValueError('unique opportunity audit artifact required')
    result=json.loads(args.result.read_text())
    if result['run_status']!='COMPLETED':raise ValueError('terminal statistical analysis required')
    if any(sha(Path(k))!=v for k,v in result['input_sha256'].items()):
        raise ValueError('result inputs drift')
    manifest=json.loads((args.source/'run_manifest.json').read_text())
    if manifest['run_status']!='COMPLETED' or manifest.get('child_exit_code')!=0:
        raise ValueError('terminal12phase source required')
    import torch
    torch.set_num_threads(2)
    rows=[];score_error=0.
    for phase in manifest['phases']:
        record=torch.load(Path(phase['directory'])/'records.pt',map_location='cpu',weights_only=False)
        for i in range(len(record['state'])):
            pre=record['state'][i].numpy();future=record['future_state'][i].numpy()
            clr=record['future_clearance'][i].numpy();contacts=record['future_contact'][i].numpy()
            rest=np.float32(record['rest_z'][i]);clear_threshold=np.float32(.002)
            support=all(bool(contacts[t,0] and contacts[t,1]) and clr[t]>=clear_threshold for t in (7,8,9))
            final=np.float32(max(np.float32(min(future[7:,38])-rest),0))
            initial=np.float32(max(np.float32(pre[38]-rest),0))
            score=np.float32(np.float32(np.float32(final*np.float32(support))-initial)*np.float32(1000))
            saved=float(record['outcome']['supported_change_mm'][i]);score_error=max(score_error,abs(float(score)-saved))
            was_clear=bool(record['initial_clearance'][i]>=clear_threshold);loss=False
            for t in range(10):
                now=bool(clr[t]>=clear_threshold)
                loss|=was_clear and not now;was_clear|=now
            q=pre[:18];offset=record['pd_offset'].numpy();scale=record['pd_scale'].numpy()
            commands=record['candidate_actions'][i].numpy()
            proposed=np.max(np.abs(target(commands[2],q,offset,scale)-target(commands[1],q,offset,scale)))>1e-5
            changed_steps=0
            for t in range(10):
                q=pre[:18] if t==0 else future[t-1,:18]
                cup=record['expert_bank'][i,t,1].numpy().copy()
                for j in (3,4,5):
                    value=np.float32(np.float32(np.float32(record['hold_target'][i,j])-q[j])-offset[j])
                    cup[j]=np.clip(np.float32(value/scale[j]),-1,1)
                cup_pd=target(cup,q,offset,scale)
                changed_steps+=int(np.max(np.abs(record['actual_pd_targets'][i,t].numpy()-cup_pd))>1e-5)
            motion=int(record['motion_id'][i]);start=int(record['start_frame'][i])
            bucket=int(hashlib.sha256(f'12651/{motion}/{start}'.encode()).hexdigest()[:8],16)%100
            if bucket!=int(record['split_group_bucket'][i]):raise ValueError('independent split group')
            rows.append(dict(arm=int(record['assignment'][i]),p=float(record['propensity'][i]),
                slot=int(record['allocation'][i]),slot_p=record['allocation_probabilities'][i].numpy().astype(float),
                bucket=bucket,episode=record['episode_id'][i],group=f'{motion}/{start}',score=saved,
                risk=float(loss),joint=float(all(contacts[t,0] and contacts[t,1] for t in (7,8,9))),
                initially_clear=bool(record['outcome']['initially_clear'][i]),
                proposed=bool(proposed),actual_changed=changed_steps>0,changed_steps=changed_steps))
    if score_error>2e-5:raise ValueError('independent physical score labels')
    discrepancies=[];max_error=0.
    def check(actual,expected,path='root'):
        nonlocal max_error
        if isinstance(expected,dict):
            for key,value in expected.items():check(actual[key],value,path+'/'+key)
        elif isinstance(expected,list):
            if len(actual)!=len(expected):discrepancies.append(path+' length')
            else:
                for j,value in enumerate(expected):check(actual[j],value,path+'/'+str(j))
        elif isinstance(expected,bool) or isinstance(expected,str) or expected is None:
            if actual!=expected:discrepancies.append(path)
        else:
            error=abs(float(actual)-float(expected));max_error=max(max_error,error)
            if error>2e-7:discrepancies.append(path)
    fit=[r for r in rows if r['bucket']<50]
    fixed={str(arm):float(np.mean([r['score']/r['p'] if r['arm']==arm else 0. for r in fit])) for arm in (0,1)}
    reference=max((0,1),key=lambda a:fixed[str(a)])
    check(result['fit_fixed_values_mm'],fixed);check(result['fixed_reference'],reference)
    controls={'cm':2,'state_only':1,'shuffled':4,'direct_score':3,'always_base':0,'best_fixed':reference,'random_pool':None}
    adequate=True;held_support=False;held_pass=False
    for name in ('fit','cal','held'):
        chosen=[r for r in rows if ('fit' if r['bucket']<50 else ('cal' if r['bucket']<70 else 'held'))==name]
        groups=np.array([r['group'] for r in chosen]);episodes=np.array([r['episode'] for r in chosen]);size=len(chosen)
        arms=[sum(r['arm']==arm for r in chosen) for arm in range(8)]
        counts=dict(rows=size,episodes=len(set(episodes)),initial_groups=len(set(groups)),arms=arms,
                    early=sum(not r['initially_clear'] for r in chosen),initially_clear=sum(r['initially_clear'] for r in chosen))
        counts['adequate']=size>=150 and counts['episodes']>=32 and counts['initial_groups']>=8 and all(v>=24 for v in arms[:5]) and sum(arms[5:])>=60
        check(result['counts'][name],counts);adequate&=counts['adequate']
        if name=='fit':continue
        policy={};inverse={};values={}
        for key,arm in controls.items():
            selected=[r for r in chosen if (r['arm']>=5 if arm is None else r['arm']==arm)]
            policy[key]=dict(matches=len(selected),episodes=len(set(r['episode'] for r in selected)),initial_groups=len(set(r['group'] for r in selected)))
            inverse[key]=np.array([(1/.3 if r['arm']>=5 else 0.) if arm is None else (1/r['p'] if r['arm']==arm else 0.) for r in chosen])
            values[key]=float(np.mean(inverse[key]*[r['score'] for r in chosen]))
        check(result['policy_support'][name],policy);check(result['ipw_values_mm'][name],values)
        null={}
        for key,slots in (('base',(0,8)),('cup',(1,9))):
            available=all(any(r['slot']==slot for r in chosen) for slot in slots)
            samples=np.array([r['score']*((1/r['slot_p'][slots[0]] if r['slot']==slots[0] else 0.)-(1/r['slot_p'][slots[1]] if r['slot']==slots[1] else 0.)) for r in chosen])
            null[key]=dict(slots=list(slots),available=available,difference_mm=float(samples.mean()) if available else None,
                           group90=interval(samples,groups,12752) if available else None)
        available=all(v['available'] for v in null.values())
        threshold=max([1.]+[2*abs(v['difference_mm']) for v in null.values()]) if available else None
        check(result['reference_null'][name],dict(pairs=null,required_gain_mm=threshold,available=available))
        actual=[r for r in chosen if r['arm']==2];changed=sum(r['actual_changed'] for r in actual)
        coverage=dict(proposed_cup_pd_change_fraction=sum(r['proposed'] for r in chosen)/size,
            actual_cm_windows=len(actual),actual_non_cup_pd_windows=changed,
            actual_non_cup_pd_steps=sum(r['changed_steps'] for r in actual),actual_cm_total_steps=10*len(actual),
            actual_non_cup_fraction=changed/len(actual) if actual else None)
        check(result['coverage'][name],coverage)
        pair_support=[];pair_gates=[]
        for control in ('state_only','shuffled','always_base','best_fixed','direct_score','random_pool'):
            contrast=inverse['cm']-inverse[control]
            samples=contrast*np.array([r['score'] for r in chosen]);effect=float(samples.mean())
            risk=float(np.mean(contrast*[r['risk'] for r in chosen]));joint=float(np.mean(contrast*[r['joint'] for r in chosen]))
            group_ci=interval(samples,groups,12753);ep_ci=interval(samples,episodes,12754)
            enough=all(v['matches']>=24 and v['episodes']>=12 and v['initial_groups']>=8 for v in (policy['cm'],policy[control]))
            gate=dict(support=enough,null_available=available,score=threshold is not None and effect>=threshold,
                group90_positive=group_ci is not None and group_ci[0]>0,episode90_positive=ep_ci is not None and ep_ci[0]>0,
                loss_point_nonregression=risk<=.02,joint_point_nonregression=joint>=-.05,
                proposed_physical_change10pct=coverage['proposed_cup_pd_change_fraction']>=.1,actual_non_cup_windows12=changed>=12)
            gate['passed']=all(gate.values())
            check(result['comparisons'][name][control],dict(score_difference_mm=effect,group90=group_ci,episode90=ep_ci,
                loss_difference=risk,joint_difference=joint,gate=gate))
            pair_support.append(enough and available);pair_gates.append(gate['passed'])
        if name=='held':held_support=all(pair_support);held_pass=all(pair_gates)
    check(result['source_support_adequate'],adequate)
    check(result['label'],'PROMISING' if adequate and held_pass else ('UNPROMISING' if adequate and held_support else 'UNCLEAR'))
    if discrepancies:raise ValueError('independent result mismatch '+repr(discrepancies))
    elapsed=time.monotonic()-started
    cumulative=result['prior_cumulative_seconds']+result['elapsed_seconds']+elapsed
    if cumulative>3600:raise TimeoutError('whole slot independent audit budget')
    audit=dict(run_status='COMPLETED',audit_passed=True,rows=len(rows),physical_score_max_error_mm=score_error,
        statistics_max_error=max_error,result_sha256=sha(args.result),auditor_sha256=sha(Path(__file__)),
        elapsed_seconds=elapsed,cumulative_seconds=cumulative,
        scope='independent labels, native actuation, known-probability values, all fixed gates and cluster intervals; no scope promotion')
    args.output.write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','result','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
