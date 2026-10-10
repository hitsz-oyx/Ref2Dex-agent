"""Post-hoc matched credit/update replay on frozen sampled rollouts; no physics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/consequence-evaluator/src')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args()
    if args.output.exists() or ROOT/'outputs/trajectory-policy' not in args.output.resolve().parents:
        raise ValueError('fresh task-owned diagnostic required')
    manifest=json.loads((args.training/'manifest.json').read_text())
    if manifest['status']!='COMPLETED' or manifest['engineering_smoke']:
        raise ValueError('frozen completed task training required')
    for path,digest in manifest['input_sha256'].items():
        if sha(path)!=digest:
            raise ValueError('frozen source/input drift: '+path)
    with np.load(args.training/'high.npz') as f:
        high={key:f[key] for key in f.files}
    with np.load(args.training/'low.npz') as f:
        low={key:f[key] for key in ('episode_start','held')}
    # Select ALL rows with >=45 consecutive held and their actual episode starts.
    selected=[]
    for start in np.unique(low['episode_start']):
        indices=np.flatnonzero(low['episode_start']==start)
        streak=np.zeros(16,int);longest=streak.copy()
        for h in low['held'][indices]:
            streak=np.where(h,streak+1,0);longest=np.maximum(longest,streak)
        rows=np.flatnonzero(longest>=45)
        if len(rows):
            j=np.flatnonzero(high['low_index']==start)
            if len(j)!=1:
                raise ValueError('unique recorded startup chunk required')
            for row in rows:
                selected.append(dict(high_index=int(j[0]),row=int(row),maximum_held=int(longest[row]),episode_start=int(start)))
    if len(selected)!=7:
        raise ValueError('predeclared seven long-held sampled rows differ')
    state=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
    util,memory=map(int,state.split(','))
    if util>10 or memory>512:
        raise ValueError('GPU not idle: '+state)
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import torch
    from trajectory_policy.actor import load_actor
    from trajectory_policy.ppo import HistoryValue
    import trajectory_policy.ppo as ppo
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    args.output.mkdir(parents=True)
    files=[args.training/name for name in ('manifest.json','high.npz','low.npz','monitor.jsonl')]+[Path(__file__)]
    hashes={str(p.resolve()):sha(p) for p in files}
    run=dict(status='RUNNING',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),physical_gpu=args.gpu,
        input_sha256=hashes,selected=selected,lambdas=[.95,1.],actor_lr=6.25e-8,fresh_adam=True,
        claim='Matched post-hoc credit/update diagnostic, not on-policy improvement; Adam moments are freshly initialized in both arms')
    (args.output/'manifest.json').write_text(json.dumps(run,indent=2)+'\n')
    started=time.monotonic();results=[]
    for u in sorted({s['high_index']//16 for s in selected}):
        path=args.training/'actors'/('u%04d.pt'%u);hashes[str(path.resolve())]=sha(path)
        packet=torch.load(path,map_location='cuda',weights_only=False)
        batch={key:torch.as_tensor(high[key][u*16:(u+1)*16],device='cuda') for key in
            ('history','c','mean','std','logp','value','reward','done','duration')}
        bootstrap=torch.as_tensor(high['bootstrap'][u*16],device='cuda')
        for lam in (.95,1.):
            actor=load_actor(packet,'cuda').train()
            value=HistoryValue(actor.history_mean,actor.history_scale).to('cuda')
            value.load_state_dict(packet['value_model'])
            actor_optimizer=torch.optim.Adam(actor.parameters(),lr=6.25e-8)
            value_optimizer=torch.optim.Adam(value.parameters(),lr=3e-4)
            update=ppo.update(actor,value,batch,bootstrap,actor_optimizer,value_optimizer,gae_lambda=lam)
            raw=update['advantage'];normalized=(raw-raw.mean())/raw.std(unbiased=False).clamp_min(1e-6)
            rows=[]
            with torch.no_grad():
                for s in selected:
                    if s['high_index']//16!=u:continue
                    k=s['high_index']%16;row=s['row']
                    h=batch['history'][k,row:row+1];c=batch['c'][k,row:row+1]
                    distribution=actor.distribution(h)
                    logp=float(distribution.log_prob(c).sum(-1))
                    xyz=(distribution.loc[0].reshape(24,12)[:8,:3]-batch['mean'][k,row].reshape(24,12)[:8,:3])*.01
                    rows.append(dict(s,raw_advantage=float(raw[k,row]),normalized_advantage=float(normalized[k,row]),
                        logprob_change=logp-float(batch['logp'][k,row]),prefix_xyz_mean_change_mm=float(xyz.square().mean().sqrt()*1000)))
            results.append(dict(update=u,lam=lam,actor_steps=update['actor_steps'],joint_kl=update['joint_action_kl'],actor_lr=update['actor_lr'],rows=rows))
            print(json.dumps(results[-1]),flush=True)
    if any(sha(path)!=digest for path,digest in hashes.items()):
        raise ValueError('diagnostic inputs drift')
    aggregate={str(lam):dict(positive_advantages=sum(r['normalized_advantage']>0 for a in results if a['lam']==lam for r in a['rows']),
        probability_increases=sum(r['logprob_change']>0 for a in results if a['lam']==lam for r in a['rows']),
        mean_logprob_change=float(np.mean([r['logprob_change'] for a in results if a['lam']==lam for r in a['rows']]))) for lam in (.95,1.)}
    report=dict(status='UNCLEAR',aggregate=aggregate,arms=results,elapsed_s=time.monotonic()-started,claim=run['claim'])
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    run.update(status='COMPLETED',input_sha256=hashes,elapsed_s=report['elapsed_s'])
    (args.output/'manifest.json').write_text(json.dumps(run,indent=2)+'\n')
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':
    def deadline(signum,frame):
        raise TimeoutError('credit replay exceeded30s')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(30)
    main()
