#!/usr/bin/env python3
"""Known-propensity local contrasts on fresh, repeatedly observed states."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha


def contrast(treated,outcome,episodes,frames):
    import torch
    y=outcome.double();t=treated.bool();delta=2*(2*t.double()-1)*y
    value=float(delta.mean())
    result=dict(effect=value,treated_rows=int(t.sum()),control_rows=int((~t).sum()),
                treated_mean=float(y[t].mean()) if t.any() else None,
                control_mean=float(y[~t].mean()) if (~t).any() else None)
    for name,ids in [('episode',episodes),('frame',frames)]:
        groups={}
        for i,group in enumerate(ids):groups.setdefault(group,[]).append(i)
        sums=[float((delta[index]-value).sum()) for index in groups.values()]
        n=len(groups);se=math.sqrt(sum(s*s for s in sums)*n/max(n-1,1))/max(len(y),1)
        result[name+'_cluster90']=[value-1.645*se,value+1.645*se]
        result[name+'_cluster95']=[value-1.96*se,value+1.96*se]
        result[name+'_groups']=n
    return result


def run(args):
    import torch
    torch.set_num_threads(2)
    manifest=json.loads((args.collection/'run_manifest.json').read_text())
    if manifest['run_status']!='COMPLETED' or manifest['experiment_id']!='P-20261001-targeted-contact-interventions':
        raise ValueError('not completed targeted collection')
    rows=[];latency=[]
    for phase in manifest['phases']:
        path=Path(phase['directory'])/'records.pt'
        if sha(path)!=phase['record_sha256']:raise ValueError('record drift')
        p=torch.load(path,map_location='cpu',weights_only=False)
        if p['schema']!='ref2dex.targeted_contact_consequence.v1':raise ValueError('wrong intervention schema')
        a=p['assignment'];trace=p['policy_trace'];proposal=trace['proposed_arm'].long();treated=trace['treatment'].bool()
        active=proposal!=4
        if not torch.equal(a,torch.where(treated,proposal,4)):raise ValueError('allocation')
        if not torch.equal(p['propensity'],torch.where(active,.5,1.)):raise ValueError('propensity')
        expected=p['candidate_actions'][torch.arange(len(a)),a]
        if not torch.equal(p['actual_action'][:,:2],expected[:,None].expand(-1,2,-1)):raise ValueError('action not executed')
        if not torch.equal(p['history'][:,-1,:49],p['state']) or p['future_done'].any():raise ValueError('state/window')
        if not all(torch.isfinite(v).all() for v in (p['state'],p['future_state'],p['actual_action'])):raise ValueError('nonfinite')
        rows.append(p);latency.extend(p['inference_latency'])
    episodes=[e for p in rows for e in p['episode_id']]
    frames=[f'{int(m)}/{int(f)}' for p in rows for m,f in zip(p['motion_id'],p['start_frame'])]
    trace={k:torch.cat([p['policy_trace'][k] for p in rows]) for k in rows[0]['policy_trace']}
    active=trace['proposed_arm'].long()!=4;treat=trace['treatment'].bool()
    outcome={k:torch.cat([p['outcome'][k] for p in rows]) for k in rows[0]['outcome']}
    checkpoint_paths=[Path(p) for p in manifest['input_sha256'] if Path(p).name=='ranker.pt']
    if len(checkpoint_paths)!=1 or sha(checkpoint_paths[0])!=rows[0]['ranker_sha256']:raise ValueError('proposer drift')
    old_groups=torch.load(checkpoint_paths[0],map_location='cpu',weights_only=False)['group_split']
    old_split=torch.tensor([old_groups.get(frame,-1) for frame in frames])
    strata={}
    for name,code in [('old_fit_frames',0),('old_calibration_frames',1),('old_held_frames',2),('new_frames',-1)]:
        mask=active&(old_split==code);idx=mask.nonzero().flatten().tolist()
        strata[name]=dict(active_windows=len(idx),all_windows=int((old_split==code).sum()),
                         scope='descriptive initial-frame overlap audit; no new threshold/model selection',
                         contrast=contrast(treat[mask],outcome['supported_lift_mm'][mask].float(),
                                           [episodes[i] for i in idx],[frames[i] for i in idx]) if idx else None)
    ep=[episodes[i] for i in active.nonzero().flatten().tolist()]
    fr=[frames[i] for i in active.nonzero().flatten().tolist()]
    if not active.any():raise ValueError('no proposed interventions')
    result={k:contrast(treat[active],outcome[k][active].float(),ep,fr)
            for k in ['supported_lift_mm','contact_fraction','drop']}
    eligible=active&outcome['drop_eligible'];risk={}
    for name,mask in [('treated',eligible&treat),('control',eligible&~treat)]:
        risk[name]=dict(rows=int(mask.sum()),events=int(outcome['drop'][mask].sum()),
                       episodes=len({episodes[i] for i in mask.nonzero().flatten().tolist()}))
    risk_supported=all(v['rows']>=20 and v['events']>=3 for v in risk.values())
    all_windows={}
    for key in ['supported_lift_mm','contact_fraction','drop']:
        # Inactive proposals are identical to base, hence their contrast is
        # exactly zero. This still concerns mixed-history observed states.
        value=contrast(treat,outcome[key].float()*active.float(),episodes,frames)
        all_windows[key]={k:v for k,v in value.items() if k not in ['treated_mean','control_mean']}
    eligible_rows=eligible.nonzero().flatten().tolist()
    conditional_drop=contrast(treat[eligible],outcome['drop'][eligible].float(),
                              [episodes[i] for i in eligible_rows],[frames[i] for i in eligible_rows]) if eligible_rows else None
    per_episode={}
    for e in episodes:per_episode[e]=per_episode.get(e,0)+1
    arm_episodes={name:len({episodes[i] for i in mask.nonzero().flatten().tolist()})
                  for name,mask in [('treated',active&treat),('control',active&~treat)]}
    support=int(active.sum())>=128 and min(result['supported_lift_mm']['treated_rows'],result['supported_lift_mm']['control_rows'])>=48 and min(arm_episodes.values())>=24
    lift=result['supported_lift_mm'];lift_ok=lift['effect']>=.5 and lift['frame_cluster90'][0]>0
    contact_ok=result['contact_fraction']['effect']>=-.05;drop_ok=result['drop']['effect']<=.05
    gate=dict(support=support,lift=lift_ok,contact=contact_ok,drop=drop_ok,risk_support=risk_supported,
              passed=support and lift_ok and contact_ok and drop_ok and risk_supported)
    actual=torch.cat([p['actual_action'] for p in rows]);candidate=torch.cat([p['candidate_actions'] for p in rows])
    delta=(actual[:,0]-candidate[:,4]).norm(dim=-1)
    duration=torch.tensor([v['milliseconds'] for v in latency])
    output=dict(experiment_id=manifest['experiment_id'],run_status='COMPLETED',
                label='UNCLEAR' if not support or not risk_supported else ('PROMISING' if gate['passed'] else 'UNPROMISING'),
                local_lift_label='UNCLEAR' if not support else ('PROMISING' if lift_ok and contact_ok else 'UNPROMISING'),
                gate=gate,contrasts=result,conditional_risk_support=risk,
                all_window_local_contrasts=all_windows,eligible_drop_contrast=conditional_drop,
                initial_frame_overlap_audit=strata,
                risk_interval_boundary='zero observed events can yield degenerate descriptive intervals; risk gate still requires actual event support',
                decision='ACTION_EFFECT_LEARNING_DESIGN' if gate['passed'] else 'REVIEW_CANDIDATE_EXECUTION_AND_EFFECT_REPRESENTATION',
                total_windows=len(episodes),episodes=len(per_episode),active_windows=int(active.sum()),
                proposal_fraction=float(active.float().mean()),actual_change_fraction=float((delta>1e-7).float().mean()),
                nonbase_executed_windows=int((delta>1e-7).sum()),action_difference_l2_quantiles=torch.quantile(delta[delta>1e-7],torch.tensor([0.,.5,.9,1.])).tolist() if (delta>1e-7).any() else None,
                proposal_counts=torch.bincount(trace['proposed_arm'].long(),minlength=6).tolist(),
                arm_episodes=arm_episodes,repeated_decision_episodes=sum(v>=2 for v in per_episode.values()),
                decision_count_distribution={str(c):sum(v==c for v in per_episode.values()) for c in sorted(set(per_episode.values()))},
                inference_batch_ms_quantiles=torch.quantile(duration,torch.tensor([0.,.5,.9,1.])).tolist(),
                collection_seconds=manifest['cumulative_seconds'],frozen_checkpoint_sha256=rows[0]['ranker_sha256'],
                record_sha256={str(Path(p['directory'])/'records.pt'):p['record_sha256'] for p in manifest['phases']},
                actor_training=False,cm_training=False,individual_regret='unavailable',
                boundary='local effect at fresh frozen proposals in randomized sequential mixed histories; no pure-policy utility or formal stable-grasp claim')
    if args.output.exists():raise ValueError('results already exist')
    args.output.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
