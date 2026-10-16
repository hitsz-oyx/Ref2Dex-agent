"""Audit all same-reset scale evaluation pairs from stored native logs (CPU only)."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[5]


def holds(mask):
    longest=tail=0
    for active in mask:
        tail=tail+1 if active else 0;longest=max(longest,tail)
    return longest,tail


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bc',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve()
    if out.exists() or ROOT/'outputs/wam-rl-policy' not in out.parents:raise ValueError('Fresh Task output')
    left=json.loads((a.bc/'manifest.json').read_text());right=json.loads((a.candidate/'manifest.json').read_text())
    if left['status']!='COMPLETED' or right['status']!='COMPLETED':raise ValueError('Completed suites required')
    if left['shards']!=right['shards'] or left['seed']!=right['seed']:raise ValueError('Same frozen panel/reset required')
    rows=[];mismatches=[];initial={k:0. for k in ('hand','object','q','dq','hand_root','observation','goal','success','reset')}
    transitions={metric:dict(both=0,bc_only=0,candidate_only=0,neither=0) for metric in ('held60','held60_at_end')}
    for shard in range(len(left['results'])):
        folders=[root/('shard%02d'%shard) for root in (a.bc,a.candidate)]
        identities=[json.loads((folder/'environment-objects.json').read_text()) for folder in folders]
        if identities[0]!=identities[1]:raise ValueError('Object mapping differs')
        with np.load(folders[0]/'last-episode.npz') as z:bc={k:z[k] for k in z.files}
        with np.load(folders[1]/'last-episode.npz') as z:candidate={k:z[k] for k in z.files}
        invalid=np.zeros(len(identities[0]),bool)
        for key in initial:
            error=float(np.max(np.abs(bc[key][0]-candidate[key][0])));initial[key]=max(initial[key],error)
            invalid|=np.any((bc[key][0]!=candidate[key][0]).reshape(len(invalid),-1),axis=1)
        mismatches.extend(dict(shard=shard,env=int(env),object_code=identities[0][env]['object_code']) for env in np.flatnonzero(invalid))
        for env,identity in enumerate(identities[0]):
            pair={}
            for role,data in [('bc',bc),('candidate',candidate)]:
                distance=np.linalg.norm(data['object'][:,env,:3]-data['goal'][:,env],axis=-1)
                lift=data['object'][:,env,2]-data['object'][0,env,2]
                longest,tail=holds((distance<=.05)&(lift>=.1))
                row=dict(shard=shard,env=env,role=role,object_code=identity['object_code'],split=identity['split'],initial_state_exact_match=not bool(invalid[env]),
                    held60=longest>=60,held60_at_end=tail>=60,longest_hold_frames=longest,endpoint_hold_frames=tail,
                    lift10=bool(lift.max()>=.1),reached5cm=bool((distance<=.05).any()),final_within5cm=bool(distance[-1]<=.05),
                    final_goal_distance_m=float(distance[-1]),max_lift_m=float(lift.max()),final_lift_m=float(lift[-1]),
                    residual_rms=float(np.sqrt(np.mean((data['action'][:,env]-data['base'][:,env])**2))))
                rows.append(row);pair[role]=row
            for metric in transitions:
                flags=(pair['bc'][metric],pair['candidate'][metric])
                name={(True,True):'both',(True,False):'bc_only',(False,True):'candidate_only',(False,False):'neither'}[flags]
                transitions[metric][name]+=1
    aggregate={}
    for role,suite in [('bc',left),('candidate',right)]:
        selected=[r for r in rows if r['role']==role]
        aggregate[role]=dict(episodes=len(selected),**{key:sum(r[key] for r in selected) for key in ('held60','held60_at_end','lift10','reached5cm','final_within5cm')})
        if aggregate[role]['held60']!=suite['totals']['held60_with_lift10cm'] or aggregate[role]['final_within5cm']!=suite['totals']['final_within5cm']:
            raise ValueError('Raw independent metric differs from reported metric')
    out.mkdir(parents=True)
    report=dict(status='UNCLEAR',initial_state_max_errors=initial,initial_state_exact_match=not mismatches,aggregate=aggregate,
        pairing_status='PASS' if not mismatches else 'FAILED_ACTUAL_INITIAL_STATE',initial_state_mismatch_envs=mismatches,
        paired_transitions=transitions if not mismatches else None,pairs=len(rows)//2,
        source_bc=str(a.bc.resolve()),source_candidate=str(a.candidate.resolve()),
        scope='All stored paired native trajectories; position hold proxies, not contact certification',model_compute=False)
    (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'cases.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    scratch=ROOT/'tmp'/out.name;scratch.mkdir(parents=True,exist_ok=True);os.environ['MPLCONFIGDIR']=str(scratch/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4));keys=['lift10','reached5cm','held60','held60_at_end']
    for offset,role in [(-.18,'bc'),(.18,'candidate')]:
        x=np.arange(len(keys))+offset;ax.bar(x,[aggregate[role][key]/aggregate[role]['episodes'] for key in keys],.36,label=role)
    ax.set_xticks(np.arange(len(keys)));ax.set_xticklabels(['Lift10cm','Reach5cm','Hold1s','Hold1s to end'])
    ax.set_ylim(0,1);ax.set_ylabel('Fraction of paired rollouts');ax.legend();ax.grid(axis='y',alpha=.2)
    if mismatches:ax.set_title('Descriptive only: actual initial states differ')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig);print(json.dumps(report))
    if mismatches:sys.exit(2)


if __name__=='__main__':main()
