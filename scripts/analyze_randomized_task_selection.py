#!/usr/bin/env python3
"""Full first-episode task metrics, independently audited from compact trajectories."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha

POLICIES=('actor','random','global','conditional')
METRICS=('retained_success','stable_success','drop_after_success','max_hold_seconds','mean_lift_meters','decision_count','nonzero_corrections')


def audit_episode(data,episode):
    env=episode['environment'];active=data['active'][:,env].numpy();height=data['height'][:,env].numpy()[active]
    contact=data['contact'][:,env].numpy()[active].all(-1);initial=np.float32(data['initial_height'][env])
    if len(height)!=episode['steps'] or not bool(data['done'][active,env][-1]):raise ValueError('terminal trace mismatch')
    run=lost=max_run=0;stable=drop=False
    for h,c in zip(height,contact):
        previous=stable;run=run+1 if h-initial>=np.float32(.03) and c else 0
        lost=0 if c else min(lost+1,6);max_run=max(max_run,run);stable=stable or run>=45
        drop=drop or (previous and (h-initial<np.float32(.02) or lost>=6))
    expected=dict(stable_success=stable,drop_after_success=drop,retained_success=stable and not drop)
    if any(episode[key]!=value for key,value in expected.items()):raise ValueError('retention/drop audit mismatch')
    np.testing.assert_allclose(episode['max_hold_seconds'],max_run/30,atol=2e-6)
    np.testing.assert_allclose(episode['mean_lift_meters'],np.maximum(height-initial,0).mean(),atol=2e-6)


def statistic(values,policies):
    return np.stack([values[policies==policy].mean(0) for policy in range(4)])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args();torch.set_num_threads(2)
    root=args.directory;manifest=json.loads((root/'run_manifest.json').read_text())
    if manifest['run_status']!='COLLECTION_COMPLETED':raise ValueError('nonterminal collection')
    if manifest['panels']!=[[286,496],[286,497],[287,496],[287,497]]:raise ValueError('test cohort drift')
    rows=[];counts={};windows={};audited=0
    for actor,evaluation in manifest['panels']:
        directory=root/f't{actor}_s{evaluation}';r=json.loads((directory/'results.json').read_text())
        if not r['all_episodes_complete'] or sha(directory/'episodes.pt')!=r['episodes_sha256'] or sha(directory/'decisions.pt')!=r['decisions_sha256']:raise ValueError('incomplete/drifted episodes')
        data=torch.load(directory/'episodes.pt',map_location='cpu',weights_only=False);windows[(actor,evaluation)]=data
        expected=torch.randint(4,(768,),generator=torch.Generator().manual_seed(11000+evaluation))
        if not torch.equal(expected,data['assignment']) or len(data['episodes'])!=768:raise ValueError('assignment/coverage drift')
        decisions=torch.load(directory/'decisions.pt',map_location='cpu',weights_only=False)
        for episode in data['episodes']:
            audit_episode(data,episode);audited+=1;env=episode['environment']
            chosen=decisions['environment']==env;ticks=decisions['tick'][chosen]
            if len(ticks)>10 or len(ticks)!=episode['decision_count'] or (ticks<10).any() or ticks.remainder(2).any():raise ValueError('decision budget drift')
            if len(ticks)>1 and (ticks[1:]-ticks[:-1]<6).any():raise ValueError('correction spacing drift')
            if (decisions['gap'][chosen]>.02).any() or not decisions['state'][chosen,49:51].bool().all():raise ValueError('acquisition drift')
            selected=decisions['selected'][chosen]
            if int((selected!=0).sum())!=episode['nonzero_corrections']:raise ValueError('correction count drift')
            if episode['policy']==0 and selected.any():raise ValueError('actor control changed')
            rows.append(dict(training_seed=actor,evaluation_seed=evaluation,**episode))
        counts[f't{actor}_s{evaluation}']=r['assignment_counts']
    if any(min(group)<32 for panel in counts.values() for group in panel.values()):
        (root/'results.json').write_text(json.dumps(dict(run_status='COMPLETED',label='UNCLEAR',reason='ASSIGNED_GROUP_COVERAGE',counts=counts),indent=2)+'\n');return
    # Arrange shared environment IDs across actors, then resample pairs.
    rng=np.random.default_rng(12026);point=np.zeros((4,len(METRICS)));boot=np.zeros((2000,4,len(METRICS)));strata=[]
    for evaluation in (496,497):
        first=windows[(286,evaluation)];second=windows[(287,evaluation)]
        if not torch.equal(first['motion'],second['motion']) or not torch.equal(first['assignment'],second['assignment']):raise ValueError('shared block drift')
        for motion in range(3):
            groups=[]
            for actor in (286,287):
                group=sorted((r for r in rows if r['training_seed']==actor and r['evaluation_seed']==evaluation and r['motion']==motion),key=lambda r:r['environment'])
                groups.append(group)
            if [r['environment'] for r in groups[0]]!=[r['environment'] for r in groups[1]]:raise ValueError('paired IDs drift')
            policy=np.array([r['policy'] for r in groups[0]])
            values=np.array([[[r[key] for key in METRICS] for r in group] for group in groups],dtype=float).mean(0)
            point+=statistic(values,policy)/6;draw=rng.integers(len(policy),size=(2000,len(policy)))
            for replicate in range(2000):boot[replicate]+=statistic(values[draw[replicate]],policy[draw[replicate]])/6
            strata.append(dict(evaluation_seed=evaluation,motion=motion,policy_counts=np.bincount(policy,minlength=4).tolist()))
    effects={}
    for control in range(3):
        delta=(point[3]-point[control]);replicates=boot[:,3]-boot[:,control]
        effects[POLICIES[control]]={key:dict(point=float(delta[j]*(100 if j<3 else 1)),
            lower95=float(np.quantile(replicates[:,j],.05)*(100 if j<3 else 1)),
            upper95=float(np.quantile(replicates[:,j],.95)*(100 if j<3 else 1)),
            central95=(np.quantile(replicates[:,j],[.025,.975])*(100 if j<3 else 1)).tolist()) for j,key in enumerate(METRICS)}
    gates={f'retention_gain_{control}':effects[control]['retained_success']['point']>=3 and effects[control]['retained_success']['lower95']>0 for control in POLICIES[:3]}
    gates['drop_increase_actor_at_most_2pp']=effects['actor']['drop_after_success']['upper95']<=2
    grouped={}
    for dimension in ('training_seed','evaluation_seed','motion'):
        grouped[dimension]={str(value):{policy:{metric:float(np.mean([r[metric] for r in rows if r[dimension]==value and r['policy']==index])) for metric in METRICS}
            for index,policy in enumerate(POLICIES)} for value in sorted({r[dimension] for r in rows})}
    result=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',gates=gates,
        complete_episodes=len(rows),policy_means={policy:{metric:float(point[i,j]) for j,metric in enumerate(METRICS)} for i,policy in enumerate(POLICIES)},
        conditional_minus_control=effects,assignment_counts=counts,subgroups=grouped,
        bootstrap=dict(replicates=2000,seed=12026,paired_actor_environment_blocks=True,strata=strata),
        boundary='exploratory one-object proxy retained task utility; no novel method or journal readiness claim')
    torch.save(dict(rows=rows,policy_means=torch.from_numpy(point),bootstrap_policy_means=torch.from_numpy(boot)),root/'analysis.pt')
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'trajectory_audit.json').write_text(json.dumps(dict(status='PASS',episodes_independently_recomputed=audited,
        decision_budget_and_acquisition_verified=True),indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
