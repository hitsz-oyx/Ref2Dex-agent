"""Compare fixed-budget credit arms using saved traces and checkpoint identities."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--intervention',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[5]
    if args.output.exists() or root/'outputs/trajectory-policy' not in args.output.resolve().parents:
        raise ValueError('fresh task-owned comparison required')
    paths=[args.control,args.intervention]
    manifests=[json.loads((p/'manifest.json').read_text()) for p in paths]
    fields=('physical_gpu','seed','updates','high_steps','envs','episode_controls','engineering_smoke','gamma',
        'actor_lr','value_lr','clip_ratio','epochs','target_joint_kl','actor_backtrack_factor','actor_backtrack_trials','matmul_precision','native_controller')
    if any(m['status']!='COMPLETED' or m['engineering_smoke'] for m in manifests):
        raise ValueError('two full completed task runs required')
    if any(manifests[0][key]!=manifests[1][key] for key in fields):
        raise ValueError('fixed run conditions differ')
    if [m['gae_lambda'] for m in manifests]!=[.95,1.]:
        raise ValueError('declared lambda.95/1 arms required')
    # Data/weights remain frozen; historical code SHA is tied to each Git identity.
    for path,digest in manifests[0]['input_sha256'].items():
        if not path.endswith('.py') and manifests[1]['input_sha256'].get(path)!=digest:
            raise ValueError('fixed input identity differs: '+path)
    for p,m in zip(paths,manifests):
        if sha(p/'final.pt')!=m['final_checkpoint_sha256']:
            raise ValueError('final checkpoint drift')
    highs=[];lows=[];reports=[]
    for p,m in zip(paths,manifests):
        with np.load(p/'high.npz') as f:
            h={k:f[k] for k in ('history','c','mean','std','reward','value','advantage','update','low_index')}
        with np.load(p/'low.npz') as f:
            l={k:f[k] for k in ('q','dq','hand','obj','velocity','action','pd_targets','reward','held','surface_gap','clipped','episode_start')}
        highs.append(h);lows.append(l)
        # Decode-neutral batch-normalized startup credit and observed holding.
        starts=[]
        for start in np.unique(l['episode_start']):
            ix=np.flatnonzero(l['episode_start']==start);j=np.flatnonzero(h['low_index']==start)
            if len(j)!=1:raise ValueError('unique startup query required')
            j=int(j[0]);u=int(h['update'][j]);a=h['advantage'][u*16:(u+1)*16]
            normalized=(a-a.mean())/max(a.std(),1e-6)
            streak=np.zeros(16,int);longest=streak.copy()
            for held in l['held'][ix]:
                streak=np.where(held,streak+1,0);longest=np.maximum(longest,streak)
            for row in np.flatnonzero(longest>=45):
                starts.append(dict(episode_start=int(start),row=int(row),maximum_held=int(longest[row]),
                    startup_normalized_advantage=float(normalized[j%16,row])))
        r=json.loads((p/'result.json').read_text())
        reports.append(dict(git_commit=m['git_commit'],gae_lambda=m['gae_lambda'],elapsed_s=m['elapsed_s'],
            interactions=r['environment_interactions'],held_steps=r['held_steps'],clipping_rate=r['clipping_rate'],
            parameter_change=r['actor_parameter_change'],longheld_startups=starts,
            positive_startup_credit=sum(s['startup_normalized_advantage']>0 for s in starts),
            completed_stable_held=sum(k>=433 and terminal for ep in r['completed_episodes'] for k,terminal in zip(ep['maximum_held'],ep['terminal_held']))))
    # First rollout precedes any update: actual samples and native physics must match.
    first={}
    for key in ('history','c','mean','std','reward','value'):
        err=float(abs(highs[0][key][:16]-highs[1][key][:16]).max());first['high_'+key]=err
        if err>2e-5:raise ValueError('pre-update sample/state mismatch: '+key+' '+str(err))
    for key in ('q','dq','hand','obj','velocity','action','pd_targets','reward','surface_gap'):
        err=float(abs(lows[0][key][:128]-lows[1][key][:128]).max());first['low_'+key]=err
        if err>2e-5:raise ValueError('pre-update physical/control mismatch: '+key+' '+str(err))
    report=dict(status='UNCLEAR',matched_fixed_fields=fields,pre_update_maximum_errors=first,arms=reports,
        claim='One-seed exploratory comparison of training signals; frozen policy evaluation required for performance')
    args.output.mkdir(parents=True)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    files=[p/name for p in paths for name in ('manifest.json','result.json','high.npz','low.npz','final.pt')]+[Path(__file__)]
    (args.output/'manifest.json').write_text(json.dumps(dict(status='COMPLETED',device='cpu',reason='File/hash and saved trace statistics only',input_sha256={str(p.resolve()):sha(p) for p in files}),indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
