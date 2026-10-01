#!/usr/bin/env python3
"""Independent CPU/NumPy reconstruction of fold scales and held risk statistics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha


def infer(x, checkpoint):
    # Separate implementation, CPU, physical units; no training helper import.
    model=torch.nn.Sequential(torch.nn.Linear(checkpoint['input_dim'],128),torch.nn.SiLU(),
        torch.nn.Linear(128,128),torch.nn.SiLU(),torch.nn.Linear(128,checkpoint['output_dim']))
    model.load_state_dict(checkpoint['model']);model.eval()
    return model((x-checkpoint['mean'])/checkpoint['std'])*checkpoint['target_std']+checkpoint['target_mean']


def check_close(a,b,atol=1e-8,rtol=1e-5):
    np.testing.assert_allclose(np.asarray(a),np.asarray(b),atol=atol,rtol=rtol)


def numpy_pseudo(y,arm,propensity):
    result=np.zeros((len(y),3,3))
    for axis in range(3):
        plus=2*axis+1;minus=plus+1
        result[:,:,axis]=y*((arm==plus)/propensity[:,plus]-(arm==minus)/propensity[:,minus])[:,None]
    return result


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fit',type=Path,required=True)
    parser.add_argument('--test',type=Path);args=parser.parse_args();torch.set_num_threads(2)
    fit=args.fit;manifest=json.loads((fit/'run_manifest.json').read_text())
    if manifest['run_status']!='COMPLETED':raise ValueError('nonterminal fit')
    for path,value in {**manifest['input_sha256'],**manifest['output_sha256']}.items():
        if sha(Path(path))!=value:raise ValueError('drift '+path)
    data=torch.load(fit/'fit_data.pt',map_location='cpu',weights_only=False)
    x,y=data['x'],data['y'];checks=[]
    for seed in (611,612,613):
        recovered=torch.zeros_like(y)
        for heldout in (492,493):
            cp=torch.load(fit/f'nuisance_s{seed}_held{heldout}.pt',map_location='cpu',weights_only=False)
            train=data['acquisition_seed']!=heldout;held=~train
            if cp['heldout_acquisition_seed']!=heldout or cp['fit_rows']!=int(train.sum()):raise ValueError('fold drift')
            a=x[train].numpy();b=y[train].numpy()
            check_close(cp['mean'].numpy(),a.mean(0),atol=2e-6,rtol=1e-4)
            check_close(cp['std'].numpy(),np.maximum(a.std(0),1e-5),atol=2e-6,rtol=1e-4)
            check_close(cp['target_mean'].numpy(),b.mean(0),atol=2e-8)
            check_close(cp['target_std'].numpy(),np.maximum(b.std(0),1e-6),atol=2e-8)
            recovered[held]=infer(x[held],cp)
            checks.append(dict(seed=seed,heldout=heldout,fit_rows=int(train.sum()),held_rows=int(held.sum()),fit_only_scales=True))
        check_close(recovered.numpy(),data[f'nuisance_s{seed}'].numpy(),atol=2e-8,rtol=2e-4)
        z=numpy_pseudo((y-data[f'nuisance_s{seed}']).numpy().astype(np.float64),data['arm'].numpy(),data['propensity'].double().numpy())
        check_close(z,data[f'pseudo_s{seed}'].numpy(),atol=1e-8,rtol=2e-6)
        check_close(z.mean(0),data[f'global_s{seed}'].numpy(),atol=1e-9,rtol=2e-5)
    result=dict(status='PASS',stage='FIT',fold_checks=checks,rows=len(x),input_hashes_verified=len(manifest['input_sha256']),
        checkpoint_hashes_verified=len(manifest['output_sha256']),independent_cpu_predictions=True,independent_numpy_pseudo=True)
    (fit/'closeout_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if args.test is None:return
    test=args.test;tm=json.loads((test/'run_manifest.json').read_text())
    if tm['run_status']!='COMPLETED':raise ValueError('nonterminal test')
    for path,value in tm['input_sha256'].items():
        if sha(Path(path))!=value:raise ValueError('test input drift '+path)
    saved=torch.load(test/'predictions.pt',map_location='cpu',weights_only=False)
    parts=[]
    for actor,evaluation in tm['panels']:
        directory=test/f't{actor}_s{evaluation}';r=json.loads((directory/'results.json').read_text())
        if sha(directory/'windows.pt')!=r['windows_sha256']:raise ValueError('test window drift')
        w=torch.load(directory/'windows.pt',map_location='cpu',weights_only=False)
        y=(w['next_state'][:,36:39]-w['state'][:,36:39]-w['state'][:,43:46]/30).double().numpy()
        parts.append(numpy_pseudo(y,w['arm'].numpy(),w['propensity'].double().numpy()))
    z=np.concatenate(parts);check_close(z,saved['pseudo_contrast'].numpy(),atol=1e-12,rtol=1e-12)
    controls=('factual','factual_compute_matched','global_residualized','zero');risks=[]
    for seed in (611,612,613):
        xx=saved['context'];cp=torch.load(fit/f'direct_contrast_s{seed}.pt',map_location='cpu',weights_only=False)
        check_close(infer(xx,cp).reshape(-1,3,3).numpy(),saved['predictions'][f'direct_contrast_s{seed}'].numpy(),atol=2e-8,rtol=2e-4)
        for method in ('factual','factual_compute_matched'):
            cp=torch.load(fit/f'{method}_s{seed}.pt',map_location='cpu',weights_only=False);candidates=[]
            for arm in range(1,7):
                indicator=torch.zeros(len(xx),6);indicator[:,arm-1]=1
                candidates.append(infer(torch.cat((xx,indicator),-1),cp))
            contrast=torch.stack([candidates[2*j]-candidates[2*j+1] for j in range(3)],-1)
            check_close(contrast.numpy(),saved['predictions'][f'{method}_s{seed}'].numpy(),atol=3e-8,rtol=3e-4)
        direct=saved['predictions'][f'direct_contrast_s{seed}'].numpy();scores=[]
        for control in controls:
            other=saved['predictions'][f'{control}_s{seed}'].numpy()
            scores.append((direct**2-other**2-2*(direct-other)*z).sum((1,2))*1e6)
        risks.append(np.stack(scores,-1))
    seed_risk=np.stack(risks,1);risk=seed_risk.mean(1)
    check_close(seed_risk,saved['seed_risk_mm2'].numpy(),atol=1e-10,rtol=1e-12)
    check_close(risk,saved['risk_mm2'].numpy(),atol=1e-10,rtol=1e-12)
    rng=np.random.default_rng(12026);boot=np.zeros((2000,4));rows=saved['rows']
    for evaluation in (494,495):
        for motion in range(3):
            a=np.array([i for i,r in enumerate(rows) if r['training_seed']==286 and r['evaluation_seed']==evaluation and r['motion']==motion])
            b=np.array([i for i,r in enumerate(rows) if r['training_seed']==287 and r['evaluation_seed']==evaluation and r['motion']==motion])
            draw=rng.integers(len(a),size=(2000,len(a)))
            boot+=(risk[a][draw].sum(1)+risk[b][draw].sum(1))/len(risk)
    check_close(boot,saved['bootstrap_mm2'].numpy(),atol=1e-10,rtol=1e-12)
    results=json.loads((test/'results.json').read_text());metrics=[]
    for j,control in enumerate(controls):
        observed=results['risk_differences_direct_minus_control'][control]
        check_close(observed['point_mm2'],risk[:,j].mean())
        check_close(observed['upper95_mm2'],np.quantile(boot[:,j],.95))
        check_close(observed['central95_mm2'],np.quantile(boot[:,j],[.025,.975]))
        check_close(observed['optimization_seed_points_mm2'],seed_risk[:,:,j].mean(0))
        metrics.append(dict(control=control,**observed))
    result=dict(status='PASS',stage='TEST',windows_verified=len(risk),input_hashes_verified=len(tm['input_sha256']),
        independent_numpy_risk_and_bootstrap=True,independent_cpu_predictions=True,comparisons=metrics)
    (test/'closeout_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
