#!/usr/bin/env python3
"""Frozen fit selection and held opportunity screen with paired cluster bootstrap."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha
METRICS=('height_mm','contact_loss','geometric_loss','joint_presence')


def data(run):
    m=json.loads((run/'run_manifest.json').read_text())
    if m['run_status']!='COMPLETED' or m['smoke_only'] or [p['seed'] for p in m['phases']]!=list(range(671,677)):
        raise ValueError('complete fixed independent six-seed source')
    if any(sha(Path(k))!=v for k,v in m['input_sha256'].items()):raise ValueError('immutable source inputs drift')
    records=[]
    for phase in m['phases']:
        path=Path(phase['directory'])/'records.pt'
        if phase['run_status']!='COMPLETED' or phase['native_exit_code'] or phase['audit_exit_code'] or sha(path)!=phase['result']['record_sha256']:
            raise ValueError('terminal audited factual record')
        if sha(phase['audit'])!=phase['audit_sha256']:raise ValueError('audit drift')
        b=torch.load(path,map_location='cpu',weights_only=False)
        if b['smoke_only'] or b['seed']!=phase['seed'] or b['schema']!='ref2dex.contact_relative_feedback_source.v1' or b['future_done'].any():
            raise ValueError('engineering or contaminated science source')
        records.append(b)
    def array(key):return torch.cat([b[key] for b in records]).numpy()
    state,future,clr,contact,rest=array('state'),array('future_state'),array('future_clearance'),array('future_contact'),array('rest_z')
    joint=contact.all(-1);terminal=joint[:,-3:].all(-1)&(clr[:,-3:]>=.002).all(-1)
    score=(np.maximum(future[:,-3:,38].min(-1)-rest,0)*terminal-np.maximum(state[:,38]-rest,0))*1000
    seen=np.maximum.accumulate(np.concatenate(((array('initial_clearance')>=.002)[:,None],clr>=.002),1),axis=1)[:,1:]
    geometric=(seen&(clr<.002)).any(-1)
    motion,start=array('motion_id'),array('start_frame')
    d=dict(assignment=array('assignment'),allocation=array('allocation'),propensity=array('propensity'),
        episode=np.array(sum([b['episode_id'] for b in records],[])),group=np.array([f'{i}/{j}' for i,j in zip(motion,start)]),
        bucket=array('split_group_bucket'),values=np.stack((score,(~joint).any(-1),geometric,joint[:,-3:].all(-1)),-1).astype(np.float64),
        initially_clear=(array('initial_clearance')>=.002)&(state[:,38]-rest>=.03))
    pre=np.concatenate((state[:,None],future[:,:-1]),1);bank=array('expert_bank');pd=array('actual_pd_targets')
    offset,scale=records[0]['pd_offset'].numpy(),records[0]['pd_scale'].numpy()
    cup_xyz=pre[:,:,:3]+offset[:3]+scale[:3]*bank[:,:,1,:3]
    d['pd_changed']=(np.abs(pd[:,:,:3]-cup_xyz).max(-1)>1e-5).any(-1)
    return m,d


def support(d,mask,arm):
    take=mask&(d['assignment']==arm)
    return dict(windows=int(take.sum()),episodes=len(set(d['episode'][take])),groups=len(set(d['group'][take])))


def estimates(d,mask,slot=False):
    choice=d['allocation'][mask] if slot else d['assignment'][mask]
    p=np.full(int(mask.sum()),1/6) if slot else d['propensity'][mask]
    weights=np.stack([(choice==a)/p for a in range(6 if slot else 4)],-1)
    sums=weights.T@d['values'][mask];counts=weights.sum(0)
    values=np.divide(sums,counts[:,None],out=np.full_like(sums,np.nan),where=counts[:,None]>0)
    ht=sums/max(1,int(mask.sum()))
    return values,ht,weights


def intervals(d,mask,cluster,seed):
    _,_,weights=estimates(d,mask)
    keys,inverse=np.unique(d[cluster][mask],return_inverse=True)
    totals=np.zeros((len(keys),4,4));counts=np.zeros((len(keys),4))
    for a in range(4):
        np.add.at(totals[:,a],inverse,weights[:,a,None]*d['values'][mask]);np.add.at(counts[:,a],inverse,weights[:,a])
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(1000):
        chosen=rng.integers(0,len(keys),len(keys));num=totals[chosen].sum(0);den=counts[chosen].sum(0)
        samples.append(np.divide(num,den[:,None],out=np.full_like(num,np.nan),where=den[:,None]>0))
    return np.stack(samples)


def run(args):
    begin=time.monotonic();torch.set_num_threads(2);m,d=data(args.run)
    fit=d['bucket']<50;cal=(d['bucket']>=50)&(d['bucket']<75);held=d['bucket']>=75
    fv,_,_=estimates(d,fit);fit_report={};eligible=[]
    for a in range(4):
        s=support(d,fit,a);ok=s['windows']>=24 and s['episodes']>=8 and s['groups']>=4
        fit_report[str(a)]=dict(support=s,height_mm=float(fv[a,0]) if np.isfinite(fv[a,0]) else None,eligible=ok)
        if ok:eligible.append(a)
    candidates=[a for a in (2,3) if a in eligible]
    selected=max(candidates,key=lambda a:fv[a,0]) if candidates else None
    fixed=max(eligible,key=lambda a:fv[a,0]) if eligible else None
    hv,ht,_=estimates(d,held);sv,_,_=estimates(d,held,True)
    slots=np.bincount(d['allocation'][held],minlength=6)
    null_available=bool((slots[:4]>=8).all())
    null_base=float(sv[0,0]-sv[1,0]) if null_available else None
    null_cup=float(sv[2,0]-sv[3,0]) if null_available else None
    required=max(2.,2*abs(null_base),2*abs(null_cup)) if null_available else None
    bootstrap={key:intervals(d,held,key,seed) for key,seed in (('group',15671),('episode',15672))} if held.any() else {}
    comparisons={}
    for option in (2,3):
        for reference in (0,1):
            delta=hv[option]-hv[reference];ci={}
            for key,samples in bootstrap.items():
                difference=samples[:,option]-samples[:,reference];finite=np.isfinite(difference).all(-1)
                ci[key]=dict(valid_replicates=int(finite.sum()),intervals=np.quantile(difference[finite],[.05,.95],axis=0).T.tolist()) if finite.any() else None
            comparisons[f'{option}-{reference}']=dict(delta=dict(zip(METRICS,delta.tolist())),
                raw_ht_delta=dict(zip(METRICS,(ht[option]-ht[reference]).tolist())),confidence90=ci)
    held_support={str(a):support(d,held,a) for a in range(4)}
    total=dict(windows=int(held.sum()),episodes=len(set(d['episode'][held])),groups=len(set(d['group'][held])))
    adequate=(selected is not None and total['windows']>=128 and total['episodes']>=32 and total['groups']>=8
        and all(held_support[str(a)]['windows']>=24 and held_support[str(a)]['episodes']>=12 and held_support[str(a)]['groups']>=8 for a in (selected,0,1)))
    gates=dict(support=bool(adequate),duplicate_reference_support=null_available)
    if selected is not None:
        c,b=comparisons[f'{selected}-1'],comparisons[f'{selected}-0']
        gates.update(cup_height_over_noise=required is not None and c['delta']['height_mm']>=required,
            base_height_at_least_2mm=b['delta']['height_mm']>=2,
            both_reference_cluster90_positive=all(x['confidence90'].get(k) is not None
                and x['confidence90'][k]['valid_replicates']>=900 and x['confidence90'][k]['intervals'][0][0]>0 for x in (c,b) for k in ('group','episode')),
            contact_loss_nonregression=c['delta']['contact_loss']<=.02,
            geometric_loss_nonregression=c['delta']['geometric_loss']<=.02,
            terminal_joint_nonregression=c['delta']['joint_presence']>=-.02,
            actual_pd_changes=int((held&(d['assignment']==selected)&d['pd_changed']).sum())>=12)
    else:gates['fit_feedback_candidate_available']=False
    label='PROMISING' if all(gates.values()) else ('UNPROMISING' if adequate and null_available else 'UNCLEAR')
    splits={}
    for name,mask in (('fit',fit),('cal',cal),('held',held)):
        value,_,_=estimates(d,mask)
        splits[name]=dict(rows=int(mask.sum()),episodes=len(set(d['episode'][mask])),groups=len(set(d['group'][mask])),
            arms=np.bincount(d['assignment'][mask],minlength=4).tolist(),initial_clear_arms=np.bincount(d['assignment'][mask&d['initially_clear']],minlength=4).tolist(),
            values={str(a):dict(zip(METRICS,[float(x) if np.isfinite(x) else None for x in value[a]])) for a in range(4)})
    clear_take=held&d['initially_clear'];clear_values,_,_=estimates(d,clear_take)
    clear_support={str(a):support(d,clear_take,a) for a in range(4)}
    drop_supported=selected is not None and min(clear_support[str(a)]['windows'] for a in (selected,1))>=20
    out=dict(experiment_id=m['experiment_id'],run_status='COMPLETED',label=label,rows=len(d['assignment']),episodes=len(set(d['episode'])),
        metrics=list(METRICS),primary_estimator='Hajek known-propensity ratio; fixed constant probabilities',
        fit_selection=fit_report,selected_feedback_option=selected,best_fixed_option=fixed,split=splits,
        held_support=held_support,comparisons=comparisons,held_duplicate_slot_counts=slots.tolist(),
        held_base_null_mm=null_base,held_cup_null_mm=null_cup,required_cup_uplift_mm=required,gates=gates,
        actual_selected_held_pd_changed_windows=int((held&(d['assignment']==selected)&d['pd_changed']).sum()) if selected is not None else 0,
        initial_clear=dict(support=clear_support,comparison_supported=drop_supported,
            selected_vs_cup_geometric_loss=float(clear_values[selected,2]-clear_values[1,2]) if drop_supported else None,
            scope='H10 initial-clear proxy; no final stable-grasp/drop claim'),
        elapsed_seconds=time.monotonic()-begin,source_manifest_sha256=sha(args.run/'run_manifest.json'),
        scope='fit-selected executable candidate opportunity only, mixed behavior conditional windows; no individual oracle, Cm information/controller/actor utility not established')
    if args.output.exists():raise ValueError('unique analysis output')
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out,indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
