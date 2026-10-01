#!/usr/bin/env python3
"""Frozen held-out IPW opportunity screen; no individual oracle or NN fit."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha


def split_group(motion,start):
    return int(hashlib.sha256(f'9851/{motion}/{start}'.encode()).hexdigest()[:8],16)%100


def clustered_interval(values,groups,seed=9852):
    keys,inverse=np.unique(groups,return_inverse=True);totals=np.zeros(len(keys));counts=np.zeros(len(keys))
    np.add.at(totals,inverse,values);np.add.at(counts,inverse,1)
    if len(keys)<2:return None
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(1000):
        chosen=rng.integers(0,len(keys),len(keys));samples.append(float(totals[chosen].sum()/counts[chosen].sum()))
    return np.quantile(samples,[.05,.95]).tolist()


def support(arm,chosen,episode,group):
    match=chosen==arm
    return dict(matches=int(match.sum()),episodes=len(set(episode[match])),initial_groups=len(set(group[match])))


def compare(option,d,mask,required_gain,null_available):
    chosen=d['assignment'][mask];p=d['propensity'][mask];ep=d['episode'][mask];group=d['group'][mask];n=int(mask.sum())
    sides={str(a):support(a,chosen,ep,group) for a in [option,4]}
    weights=(chosen==option)/p-(chosen==4)/p
    score=weights*d['score'][mask];risk=weights*d['risk'][mask];contact=weights*d['contact'][mask]
    effect=float(score.mean()) if n else None
    frame=clustered_interval(score,group) if n else None;episode=clustered_interval(score,ep) if n else None
    adequate=all(v['matches']>=48 and v['episodes']>=12 and v['initial_groups']>=8 for v in sides.values())
    gates=dict(support=adequate,null_available=null_available,
               signed_retention_uplift=effect is not None and required_gain is not None and effect>=required_gain,
               frame90_positive=frame is not None and frame[0]>0,episode90_positive=episode is not None and episode[0]>0,
               lost_clearance_risk_point_bound=n>0 and float(risk.mean())<=.02,
               last3_joint_force_point_bound=n>0 and float(contact.mean())>=-.05)
    gates['passed']=all(gates.values())
    label='PROMISING' if gates['passed'] else ('UNPROMISING' if adequate and null_available else 'UNCLEAR')
    return dict(option=int(option),already_clear_windows=n,support=sides,score_uplift_mm=effect,frame90=frame,episode90=episode,
                lost_clearance_risk_difference=float(risk.mean()) if n else None,joint_force_difference=float(contact.mean()) if n else None,
                gate=gates,label=label)


def run(args):
    torch.set_num_threads(2);m=json.loads((args.run/'run_manifest.json').read_text())
    if m['run_status']!='COMPLETED' or m['smoke_only'] or len(m['phases'])!=12:raise ValueError('all frozen12science phases required')
    if not all(sha(Path(k))==v for k,v in m['input_sha256'].items()):raise ValueError('source/input drift')
    records=[]
    for p in m['phases']:
        path=Path(p['directory'])/'records.pt'
        if p['run_status']!='COMPLETED' or sha(path)!=p['result']['record_sha256']:raise ValueError('record drift')
        b=torch.load(path,map_location='cpu',weights_only=False)
        if b['schema']!='ref2dex.wrist_feedback_options.v1' or b['future_done'].any() or b['cm_used']:raise ValueError('actual-label contract')
        records.append(b)
    keys=['assignment','propensity','allocation','allocation_probabilities','state','future_state','future_contact','future_clearance','initial_clearance','rest_z','motion_id','start_frame']
    tensors={k:torch.cat([b[k] for b in records]) for k in keys};episode=np.array(sum([b['episode_id'] for b in records],[]))
    motion=tensors['motion_id'].tolist();starts=tensors['start_frame'].tolist()
    groups=np.array([f'{i}/{j}' for i,j in zip(motion,starts)]);bucket=np.array([split_group(i,j) for i,j in zip(motion,starts)])
    split=np.where(bucket<50,'fit',np.where(bucket<70,'cal','held'))
    initial=(tensors['initial_clearance']>=.002)&(tensors['state'][:,38]-tensors['rest_z']>=.03)
    pair=tensors['future_contact'].all(-1);clear=tensors['future_clearance']>=.002
    score=((tensors['future_state'][:,-3:,38].amin(-1)-tensors['rest_z']).clamp_min(0)*pair[:,-3:].all(-1)*clear[:,-3:].all(-1)-(tensors['state'][:,38]-tensors['rest_z']).clamp_min(0))*1000
    d=dict(assignment=tensors['assignment'].numpy(),propensity=tensors['propensity'].numpy(),episode=episode,group=groups,score=score.numpy(),
           risk=(~clear.all(-1)).float().numpy(),contact=pair[:,-3:].all(-1).float().numpy())
    probabilities=tensors['allocation_probabilities'];row=torch.arange(len(score));actual=probabilities[row,tensors['assignment']]+torch.where(tensors['assignment']==4,probabilities[:,8],torch.zeros(len(row)))
    if not torch.allclose(actual,tensors['propensity'],atol=1e-7):raise ValueError('actual allocation propensity drift')
    fit=(split=='fit')&initial.numpy();held=(split=='held')&initial.numpy();fit_values={};eligible=[]
    for arm in range(8):
        supp=support(arm,d['assignment'][fit],episode[fit],groups[fit]);value=float(((d['assignment'][fit]==arm)/d['propensity'][fit]*d['score'][fit]).mean()) if fit.any() else None
        ok=supp['matches']>=24 and supp['episodes']>=8 and supp['initial_groups']>=4
        fit_values[str(arm)]=dict(support=supp,ipw_value_mm=value,eligible=ok)
        if ok:eligible.append(arm)
    fixed=max(eligible,key=lambda arm:fit_values[str(arm)]['ipw_value_mm']) if eligible else None
    allocation=tensors['allocation'].numpy()[held];allocp=probabilities.numpy()[held];local=d['score'][held]
    null_available=bool((allocation==4).any() and (allocation==8).any());null_diff=None;null_interval=None
    if null_available:
        values=((allocation==4)/allocp[:,4]-(allocation==8)/allocp[:,8])*local
        null_diff=float(values.mean());null_interval=clustered_interval(values,groups[held])
    threshold=max(2.,2*abs(null_diff)) if null_available else None
    anchored=[a for a in eligible if a in (6,7)]
    selected_anchored=max(anchored,key=lambda a:fit_values[str(a)]['ipw_value_mm']) if anchored else None
    primary=compare(selected_anchored,d,held,threshold,null_available) if selected_anchored is not None else None
    secondary=compare(fixed,d,held,threshold,null_available) if fixed is not None else None
    variants={str(a):compare(a,d,held,threshold,null_available) for a in (6,7)}
    label=primary['label'] if primary is not None else 'UNCLEAR'
    summaries={}
    for key in ['fit','cal','held']:
        take=split==key;summaries[key]=dict(rows=int(take.sum()),episodes=len(set(episode[take])),initial_groups=len(set(groups[take])),already_clear=int((take&initial.numpy()).sum()),
            arms=np.bincount(d['assignment'][take],minlength=8).tolist(),already_clear_arms=np.bincount(d['assignment'][take&initial.numpy()],minlength=8).tolist())
    result=dict(experiment_id=m['experiment_id'],run_status='COMPLETED',label=label,rows=len(row),episodes=len(set(episode)),split=summaries,
                fit_fixed_candidate_values=fit_values,best_fixed_option=fixed,selected_anchored_option=selected_anchored,primary_fit_selected_anchored_vs_base=primary,descriptive_anchored_variants=variants,secondary_fit_fixed_vs_base=secondary,
                held_base_null_difference_mm=null_diff,held_base_null_frame90=null_interval,required_uplift_mm=threshold,
                total_seconds_including_engineering=m['cumulative_seconds'],bytes_including_engineering=m['output_bytes'],
                scope='known-propensity randomized local full10plan consequences on observed already-clear prestates; no individual oracle/regret, no learned Cm or two-step MPC deployment, no full-task success or formal safety proof',
                claim='C3 OPEN; eight full-window candidates, unqualified fixed options reported rather than presumed comparable')
    if args.output.exists():raise ValueError('analysis exists')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['fit_fixed_candidate_values','split']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
