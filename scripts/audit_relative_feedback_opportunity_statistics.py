#!/usr/bin/env python3
"""Scalar independent physical-label/ratio/selection/gate audit of the fixed screen."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha


def run(args):
    begin=time.monotonic();torch.set_num_threads(2)
    m=json.loads((args.run/'run_manifest.json').read_text());r=json.loads(args.results.read_text())
    if m['run_status']!='COMPLETED' or r['source_manifest_sha256']!=sha(args.run/'run_manifest.json'):
        raise ValueError('terminal pinned source')
    rows=[]
    for phase in m['phases']:
        path=Path(phase['directory'])/'records.pt'
        if sha(path)!=phase['result']['record_sha256']:raise ValueError('record drift')
        b=torch.load(path,map_location='cpu',weights_only=False)
        if b['smoke_only']:raise ValueError('excluded engineering entered statistics')
        for i in range(len(b['state'])):
            a=int(b['assignment'][i]);p=float(b['propensity'][i]);rest=float(b['rest_z'][i]);weight=float(b['mass_kg'][i])*b['gravity_magnitude']
            pair=[];clears=b['future_clearance'][i].double().numpy();seen=bool(b['initial_clearance'][i]>=.002);loss=False
            for t in range(10):
                hand=max(float(np.linalg.norm(f)) for f in b['future_hand_force'][i,t].double().numpy())/weight
                obj=float(np.linalg.norm(b['future_object_force'][i,t].double().numpy()))/weight
                pair.append(hand>.1 and obj>.1)
                seen=seen or clears[t]>=.002;loss=loss or (seen and clears[t]<.002)
            supported=all(pair[-3:]) and all(c>=.002 for c in clears[-3:])
            height=max(min(float(x) for x in b['future_state'][i,-3:,38])-rest,0)*supported-max(float(b['state'][i,38])-rest,0)
            slot=int(b['allocation'][i]);bucket=int(b['split_group_bucket'][i]);pre=b['state'][i].double().numpy();changed=False
            for t in range(10):
                nominal=pre[:3]+b['pd_offset'][:3].double().numpy()+b['pd_scale'][:3].double().numpy()*b['expert_bank'][i,t,1,:3].double().numpy()
                changed=changed or bool(np.max(np.abs(nominal-b['actual_pd_targets'][i,t,:3].double().numpy()))>1e-5)
                pre=b['future_state'][i,t].double().numpy()
            rows.append(dict(arm=a,p=p,slot=slot,bucket=bucket,ep=b['episode_id'][i],
                group=f"{int(b['motion_id'][i])}/{int(b['start_frame'][i])}",y=[height*1000,float(not all(pair)),float(loss),float(all(pair[-3:]))],changed=changed))
    fit=[x for x in rows if x['bucket']<50];held=[x for x in rows if x['bucket']>=75]
    def values(records,slot=False):
        output=[]
        for arm in range(6 if slot else 4):
            matches=[x for x in records if x['slot' if slot else 'arm']==arm]
            denominator=sum(6. if slot else 1/x['p'] for x in matches)
            output.append([sum((6. if slot else 1/x['p'])*x['y'][metric] for x in matches)/denominator for metric in range(4)] if denominator else [float('nan')]*4)
        return np.array(output)
    fv,hv,sv=values(fit),values(held),values(held,True)
    eligible=[]
    for a in range(4):
        group=[x for x in fit if x['arm']==a]
        if len(group)>=24 and len(set(x['ep'] for x in group))>=8 and len(set(x['group'] for x in group))>=4:eligible.append(a)
    candidates=[a for a in (2,3) if a in eligible]
    selected=max(candidates,key=lambda a:fv[a,0]) if candidates else None
    best=max(eligible,key=lambda a:fv[a,0]) if eligible else None
    if selected!=r['selected_feedback_option'] or best!=r['best_fixed_option']:raise ValueError('fit-only fixed selection')
    errors=[]
    for a in (2,3):
        for ref in (0,1):
            expected=np.array(list(r['comparisons'][f'{a}-{ref}']['delta'].values()))
            errors.extend(np.abs(expected-(hv[a]-hv[ref])).tolist())
    if max(errors)>3e-5:raise ValueError('scalar physical/Hajek effects drift')
    slot_counts=[sum(x['slot']==a for x in held) for a in range(6)]
    if slot_counts!=r['held_duplicate_slot_counts']:raise ValueError('duplicate slot support')
    null_available=min(slot_counts[:4])>=8
    threshold=max(2.,2*abs(sv[0,0]-sv[1,0]),2*abs(sv[2,0]-sv[3,0])) if null_available else None
    if threshold is not None and abs(threshold-r['required_cup_uplift_mm'])>3e-5:raise ValueError('repeat-noise gate drift')
    for cluster,seed in [('group',15671),('ep',15672)]:
        labels=sorted(set(x[cluster] for x in held));clusters=[[x for x in held if x[cluster]==label] for label in labels]
        numerators=np.array([[[sum(x['y'][k]/x['p'] for x in group if x['arm']==a) for k in range(4)] for a in range(4)] for group in clusters])
        denominators=np.array([[sum(1/x['p'] for x in group if x['arm']==a) for a in range(4)] for group in clusters])
        rng=np.random.default_rng(seed);samples=[]
        for _ in range(1000):
            chosen=rng.integers(0,len(labels),len(labels));num=sum(numerators[i] for i in chosen);den=sum(denominators[i] for i in chosen)
            samples.append(np.divide(num,den[:,None],out=np.full_like(num,np.nan),where=den[:,None]>0))
        samples=np.array(samples);key='episode' if cluster=='ep' else cluster
        for a in (2,3):
            for ref in (0,1):
                difference=samples[:,a]-samples[:,ref];finite=np.isfinite(difference).all(-1)
                interval=np.quantile(difference[finite],[.05,.95],axis=0).T
                expected=r['comparisons'][f'{a}-{ref}']['confidence90'][key]
                if int(finite.sum())!=expected['valid_replicates']:raise ValueError('cluster valid bootstrap count')
                errors.extend(np.abs(interval-np.array(expected['intervals'])).flatten().tolist())
    if max(errors)>3e-5:raise ValueError('independent cluster intervals drift')
    counts={a:[x for x in held if x['arm']==a] for a in range(4)}
    adequate=(selected is not None and len(held)>=128 and len(set(x['ep'] for x in held))>=32 and len(set(x['group'] for x in held))>=8
        and all(len(counts[a])>=24 and len(set(x['ep'] for x in counts[a]))>=12 and len(set(x['group'] for x in counts[a]))>=8 for a in (selected,0,1)))
    gates=dict(support=bool(adequate),duplicate_reference_support=null_available)
    if selected is not None:
        c,b=hv[selected]-hv[1],hv[selected]-hv[0]
        gates.update(cup_height_over_noise=threshold is not None and c[0]>=threshold,base_height_at_least_2mm=b[0]>=2,
            both_reference_cluster90_positive=all(r['comparisons'][f'{selected}-{ref}']['confidence90'][key]['valid_replicates']>=900
                and r['comparisons'][f'{selected}-{ref}']['confidence90'][key]['intervals'][0][0]>0 for ref in (0,1) for key in ('group','episode')),
            contact_loss_nonregression=c[1]<=.02,geometric_loss_nonregression=c[2]<=.02,terminal_joint_nonregression=c[3]>=-.02,
            actual_pd_changes=sum(x['changed'] for x in held if x['arm']==selected)>=12)
    else:gates['fit_feedback_candidate_available']=False
    label='PROMISING' if all(gates.values()) else ('UNPROMISING' if adequate and null_available else 'UNCLEAR')
    if gates!=r['gates'] or label!=r['label']:raise ValueError('original gate/classification')
    result=dict(run_status='COMPLETED',passed=True,rows=len(rows),scalar_effect_and_interval_max_error=max(errors),
        selected_feedback=selected,best_fixed=best,classification=label,original_gates_verified=True,
        elapsed_seconds=time.monotonic()-begin,results_sha256=sha(args.results),
        scope='independent scalar rawforce/height, fixed ratios and fitselection, paired-cluster intervals and original gates')
    if args.output.exists():raise ValueError('unique statistics audit')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
