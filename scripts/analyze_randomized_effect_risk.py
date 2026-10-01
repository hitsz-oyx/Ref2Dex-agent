#!/usr/bin/env python3
"""Frozen counterfactual predictions and identified randomized risk differences."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import admission,sha
from scripts.analyze_fresh_causal_transfer import bridge_and_limits,predict
from src.task.CmResidual.randomized_effect_risk import pseudo_contrast,risk_difference


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args()
    gpu=admission(4)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu['uuid']:raise ValueError('admitted GPU mismatch')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    root=args.directory;manifest=json.loads((root/'run_manifest.json').read_text())
    if manifest['run_status']!='COLLECTION_COMPLETED':raise ValueError('collection not complete')
    bridge,limits=bridge_and_limits();models=Path(manifest['frozen_models'])
    risks=[];by_seed=[];all_predictions={};all_pseudo=[];rows=[];windows={};axis_scores=[]
    for training,evaluation in manifest['panels']:
        name=f't{training}_s{evaluation}';directory=root/name
        record=json.loads((directory/'results.json').read_text());filename=directory/'windows.pt'
        if not record['all_windows_complete'] or sha(filename)!=record['windows_sha256']:raise ValueError('incomplete/drifted windows')
        data=torch.load(filename,weights_only=False,map_location='cpu');windows[(training,evaluation)]=data
        if not data['complete'].all() or len(data['state'])!=768:raise ValueError('partial panel forbidden')
        if (data['gap']>.02).any():raise ValueError('acquisition mismatch')
        if (data['trigger']<10).any() or data['trigger'].remainder(2).any():raise ValueError('trigger schedule mismatch')
        for m in range(3):
            chosen=data['motion']==m;count=torch.bincount(data['arm'][chosen],minlength=7)
            if count.tolist()!=[37,37,37,37,36,36,36]:raise ValueError('assignment balance drift')
            if not torch.allclose(data['propensity'][chosen],(count/256).expand(256,-1)):raise ValueError('propensity drift')
        expected=data['reference_action'].clone()
        for label in range(1,7):
            chosen=data['arm']==label;axis=(label-1)//2
            expected[chosen,axis]+=.01 if label%2 else -.01
        if not torch.equal(expected,data['action']):raise ValueError('pulse/action interpretation drift')
        response=data['next_state'][:,36:39]-data['state'][:,36:39]-data['state'][:,43:46]/30
        pseudo=pseudo_contrast(response.double(),data['arm'],data['propensity'].double())
        all_pseudo.append(pseudo);panel_risk=[];panel_axes=[]
        for seed in (411,412,413):
            contrasts={}
            for method in ('nominal_motion','raw_command'):
                checkpoint=torch.load(models/f'effect_{method}_s{seed}.pt',weights_only=False,map_location='cpu')
                columns=[]
                for axis in range(3):
                    plus=data['reference_action'].clone();minus=plus.clone();plus[:,axis]+=.01;minus[:,axis]-=.01
                    if (plus.abs()>1).any() or (minus.abs()>1).any():raise ValueError('candidate clipping')
                    columns.append(predict(data['state'],plus,bridge,limits,checkpoint)-predict(data['state'],minus,bridge,limits,checkpoint))
                contrasts[method]=torch.stack(columns,-1).double()
                all_predictions.setdefault(f'{method}_s{seed}',[]).append(contrasts[method])
            nominal,raw=contrasts['nominal_motion'],contrasts['raw_command'];zero=torch.zeros_like(raw)
            metrics=[];axis_metrics=[]
            for first,second in ((nominal,raw),(nominal,zero),(raw,zero)):
                scores=risk_difference(first,second,pseudo)*1e6;metrics.append(scores)
                axis_metrics.append((first.square()-second.square()-2*(first-second)*pseudo).sum(1)*1e6)
            panel_risk.append(torch.stack(metrics,-1));panel_axes.append(torch.stack(axis_metrics,-1))
        panel_risk=torch.stack(panel_risk,1) # [env, optimization seed, comparison]
        risks.append(panel_risk.mean(1));by_seed.append(panel_risk);axis_scores.append(torch.stack(panel_axes,1).mean(1))
        rows.extend(dict(training_seed=training,evaluation_seed=evaluation,environment=int(env),motion=int(m))
                    for env,m in zip(data['environment'],data['motion']))
        print(json.dumps(dict(panel=name,rows=768)),flush=True)
    risk=torch.cat(risks).numpy();seed_risk=torch.cat(by_seed).numpy();axis_risk=torch.cat(axis_scores).numpy()
    comparisons=('nominal_minus_command','nominal_minus_zero','command_minus_zero')
    strata=[]
    for evaluation in (492,493):
        first=windows[(286,evaluation)];second=windows[(287,evaluation)]
        if not torch.equal(first['motion'],second['motion']) or not torch.equal(first['arm'],second['arm']):
            raise ValueError('shared seed/assignment block mismatch')
        for motion in range(3):
            a=np.array([i for i,row in enumerate(rows) if row['training_seed']==286 and row['evaluation_seed']==evaluation and row['motion']==motion])
            b=np.array([i for i,row in enumerate(rows) if row['training_seed']==287 and row['evaluation_seed']==evaluation and row['motion']==motion])
            if [rows[i]['environment'] for i in a]!=[rows[i]['environment'] for i in b]:raise ValueError('paired environment IDs drift')
            strata.append((a,b))
    rng=np.random.default_rng(12026);bootstrap=np.zeros((2000,3),dtype=np.float64)
    for a,b in strata:
        draw=rng.integers(0,len(a),size=(2000,len(a)))
        bootstrap+=(risk[a][draw].sum(1)+risk[b][draw].sum(1))/len(risk)
    estimated={key:dict(point_mm2=float(risk[:,j].mean()),upper95_mm2=float(np.quantile(bootstrap[:,j],.95)),
                        central95_mm2=np.quantile(bootstrap[:,j],[.025,.975]).tolist(),
                        optimization_seed_points_mm2=seed_risk[:,:,j].mean(0).tolist()) for j,key in enumerate(comparisons)}
    gates=dict(nominal_better_command_upper95=estimated['nominal_minus_command']['upper95_mm2']<0,
               nominal_better_zero_upper95=estimated['nominal_minus_zero']['upper95_mm2']<0,
               nominal_better_command_each_seed=bool((seed_risk[:,:,0].mean(0)<0).all()))
    grouped={}
    for dimension in ('training_seed','evaluation_seed','motion'):
        grouped[dimension]={}
        for value in sorted({row[dimension] for row in rows}):
            mask=np.array([row[dimension]==value for row in rows])
            grouped[dimension][str(value)]={key:float(risk[mask,j].mean()) for j,key in enumerate(comparisons)}
    result=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',gates=gates,
        rows=len(rows),risk_differences=estimated,grouped_point_mm2=grouped,
        axis_point_mm2={axis:{key:float(axis_risk[:,i,j].mean()) for j,key in enumerate(comparisons)} for i,axis in enumerate(('x','y','z'))},
        bootstrap=dict(replicates=2000,seed=12026,strata=6,shared_actor_environment_blocks_preserved=True),
        boundary='identified randomized conditional-effect risk differences; no absolute effect RMSE, policy or object generalization')
    torch.save(dict(pseudo_contrast=torch.cat(all_pseudo),predictions={k:torch.cat(v) for k,v in all_predictions.items()},
                    rows=rows,risk_mm2=torch.from_numpy(risk),seed_risk_mm2=torch.from_numpy(seed_risk),bootstrap_mm2=torch.from_numpy(bootstrap)),root/'predictions.pt')
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
