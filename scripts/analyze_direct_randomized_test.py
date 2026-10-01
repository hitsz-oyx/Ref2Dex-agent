#!/usr/bin/env python3
"""Identified held-cohort risk differences for fixed randomized effect learners."""
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
from scripts.analyze_fresh_causal_transfer import bridge_and_limits
from src.task.CmResidual.direct_randomized_response import SEEDS,context,iid_assignment,predict,factual_contrast
from src.task.CmResidual.randomized_effect_risk import pseudo_contrast,risk_difference

CONTROLS=('factual','factual_compute_matched','global_residualized','zero')


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args()
    gpu=admission(4)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu['uuid']:raise ValueError('admitted GPU mismatch')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    root=args.directory;manifest=json.loads((root/'run_manifest.json').read_text())
    if manifest['run_status']!='COLLECTION_COMPLETED':raise ValueError('collection not complete')
    if manifest['panels']!=[[286,494],[286,495],[287,494],[287,495]]:raise ValueError('test cohort drift')
    bridge,_=bridge_and_limits();models=Path(manifest['frozen_models'])
    fit_data=torch.load(models/'fit_data.pt',map_location='cpu',weights_only=False)
    if sorted(fit_data['acquisition_seed'].unique().tolist())!=[492,493]:raise ValueError('fit/test leakage')
    all_pseudo=[];rows=[];windows={};predictions={};risk_parts=[];axis_parts=[];contexts=[]
    for actor,evaluation in manifest['panels']:
        name=f't{actor}_s{evaluation}';directory=root/name
        record=json.loads((directory/'results.json').read_text());filename=directory/'windows.pt'
        if not record['all_windows_complete'] or sha(filename)!=record['windows_sha256']:raise ValueError('incomplete/drifted test panel')
        data=torch.load(filename,map_location='cpu',weights_only=False);windows[(actor,evaluation)]=data
        if not data['complete'].all() or len(data['state'])!=768:raise ValueError('partial panel forbidden')
        if (data['gap']>.02).any() or (data['trigger']<10).any() or data['trigger'].remainder(2).any():raise ValueError('trigger drift')
        assignment,propensity=iid_assignment(768,evaluation)
        if record['assignment_mode']!='iid' or not torch.equal(data['arm'],assignment) or not torch.equal(data['propensity'],propensity):
            raise ValueError('IID assignment or propensity drift')
        expected=data['reference_action'].clone()
        for arm in range(1,7):
            chosen=data['arm']==arm;expected[chosen,(arm-1)//2]+=.01 if arm%2 else -.01
        if not torch.equal(expected,data['action']):raise ValueError('pulse drift')
        x=context(data['state'],data['reference_action'],bridge);contexts.append(x.cpu())
        y=data['next_state'][:,36:39]-data['state'][:,36:39]-data['state'][:,43:46]/30
        z=pseudo_contrast(y.double(),data['arm'],data['propensity'].double());all_pseudo.append(z)
        panel_risk=[];panel_axis=[]
        for seed in SEEDS:
            direct=predict(x,torch.load(models/f'direct_contrast_s{seed}.pt',map_location='cpu',weights_only=False)).reshape(-1,3,3).cpu().double()
            contrasts=dict(direct_contrast=direct)
            for method in ('factual','factual_compute_matched'):
                checkpoint=torch.load(models/f'{method}_s{seed}.pt',map_location='cpu',weights_only=False)
                contrasts[method]=factual_contrast(x,checkpoint).cpu().double()
            contrasts['global_residualized']=fit_data[f'global_s{seed}'].double().expand(len(x),-1,-1)
            contrasts['global_unadjusted']=fit_data['unadjusted_pseudo'].mean(0).double().expand(len(x),-1,-1)
            contrasts['zero']=torch.zeros_like(direct)
            for method,value in contrasts.items():
                if not torch.isfinite(value).all():raise FloatingPointError('nonfinite prediction')
                predictions.setdefault(f'{method}_s{seed}',[]).append(value)
            panel_risk.append(torch.stack([risk_difference(direct,contrasts[method],z)*1e6 for method in CONTROLS],-1))
            panel_axis.append(torch.stack([(direct.square()-contrasts[method].square()-2*(direct-contrasts[method])*z).sum(1)*1e6
                for method in CONTROLS],-1))
        risk_parts.append(torch.stack(panel_risk,1));axis_parts.append(torch.stack(panel_axis,1).mean(1))
        rows.extend(dict(training_seed=actor,evaluation_seed=evaluation,environment=int(e),motion=int(m))
            for e,m in zip(data['environment'],data['motion']))
        print(json.dumps(dict(panel=name,rows=768)),flush=True)
    seed_risk=torch.cat(risk_parts).numpy();risk=seed_risk.mean(1);axis_risk=torch.cat(axis_parts).numpy()
    strata=[]
    for evaluation in (494,495):
        first=windows[(286,evaluation)];second=windows[(287,evaluation)]
        if not torch.equal(first['motion'],second['motion']) or not torch.equal(first['arm'],second['arm']):raise ValueError('shared block mismatch')
        for motion in range(3):
            a=np.array([i for i,row in enumerate(rows) if row['training_seed']==286 and row['evaluation_seed']==evaluation and row['motion']==motion])
            b=np.array([i for i,row in enumerate(rows) if row['training_seed']==287 and row['evaluation_seed']==evaluation and row['motion']==motion])
            if [rows[i]['environment'] for i in a]!=[rows[i]['environment'] for i in b]:raise ValueError('environment block drift')
            strata.append((a,b))
    rng=np.random.default_rng(12026);bootstrap=np.zeros((2000,len(CONTROLS)))
    for a,b in strata:
        draw=rng.integers(0,len(a),size=(2000,len(a)))
        bootstrap+=(risk[a][draw].sum(1)+risk[b][draw].sum(1))/len(risk)
    estimates={method:dict(point_mm2=float(risk[:,j].mean()),upper95_mm2=float(np.quantile(bootstrap[:,j],.95)),
        central95_mm2=np.quantile(bootstrap[:,j],[.025,.975]).tolist(),optimization_seed_points_mm2=seed_risk[:,:,j].mean(0).tolist())
        for j,method in enumerate(CONTROLS)}
    gates={f'better_{method}_upper95':estimates[method]['upper95_mm2']<0 for method in CONTROLS}
    gates['better_factual_each_seed']=bool((seed_risk[:,:,0].mean(0)<0).all())
    grouped={}
    for dimension in ('training_seed','evaluation_seed','motion'):
        grouped[dimension]={}
        for value in sorted({row[dimension] for row in rows}):
            mask=np.array([row[dimension]==value for row in rows])
            grouped[dimension][str(value)]={method:float(risk[mask,j].mean()) for j,method in enumerate(CONTROLS)}
    result=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',gates=gates,
        rows=len(rows),risk_differences_direct_minus_control=estimates,grouped_point_mm2=grouped,
        axis_point_mm2={axis:{method:float(axis_risk[:,i,j].mean()) for j,method in enumerate(CONTROLS)} for i,axis in enumerate(('x','y','z'))},
        bootstrap=dict(replicates=2000,seed=12026,strata=6,paired_actor_environment_blocks=True),
        global_unadjusted_fit_matrix_mm=(fit_data['unadjusted_pseudo'].mean(0)*1000).tolist(),
        fit_wall_seconds=json.loads((models/'run_manifest.json').read_text())['wall_seconds'],
        boundary='exploratory conditional-effect risk differences on fixed actors/object; no absolute effect RMSE or policy/journal claim')
    torch.save(dict(pseudo_contrast=torch.cat(all_pseudo),context=torch.cat(contexts),
        predictions={key:torch.cat(value) for key,value in predictions.items()},rows=rows,
        seed_risk_mm2=torch.from_numpy(seed_risk),risk_mm2=torch.from_numpy(risk),bootstrap_mm2=torch.from_numpy(bootstrap)),root/'predictions.pt')
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
