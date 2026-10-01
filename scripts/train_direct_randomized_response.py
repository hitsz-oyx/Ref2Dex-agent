#!/usr/bin/env python3
"""Crossfit and fit fixed learners on the previous randomized acquisition only."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import torch
from scripts.run_contact_response_probe import admission, sha
from scripts.analyze_fresh_causal_transfer import bridge_and_limits
from src.task.CmResidual.direct_randomized_response import SEEDS, context, fit, predict, indicators
from src.task.CmResidual.randomized_effect_risk import pseudo_contrast

PRIOR = ROOT/'src/task/CmResidual/research/contact_response/output/P-20261001-randomized-effect-risk-r2'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=5); args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or ROOT not in output.parents: raise ValueError('unique independent output required')
    gpu = admission(args.gpu)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu['uuid']: raise ValueError('GPU mismatch')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    prior = json.loads((PRIOR/'run_manifest.json').read_text())
    audit = json.loads((PRIOR/'closeout_audit.json').read_text())
    if prior['run_status'] != 'COMPLETED' or audit.get('status') != 'PASS':
        raise ValueError('previous acquisition incomplete or unaudited')
    hashes = dict(prior['input_sha256'])
    sources = [Path(__file__), ROOT/'src/task/CmResidual/direct_randomized_response.py',
        ROOT/'scripts/run_direct_randomized_test.py', ROOT/'scripts/analyze_direct_randomized_test.py',
        ROOT/'docs/experiments/probes/P-20261001-direct-randomized-response.md',
        PRIOR/'run_manifest.json', PRIOR/'closeout_audit.json']
    hashes.update({str(p.resolve()): sha(p) for p in sources})
    # Historical native sources may change for the new explicit IID assignment mode;
    # retain prior input identities, but verify current physical assets separately.
    historical_sources = {str((ROOT/'scripts/run_randomized_effect_environment.py').resolve())}
    for path, value in hashes.items():
        if path not in historical_sources and sha(Path(path)) != value: raise ValueError('input drift: '+path)
    hashes[str((ROOT/'scripts/run_randomized_effect_environment.py').resolve())] = sha(ROOT/'scripts/run_randomized_effect_environment.py')
    output.mkdir(parents=True); begin = time.monotonic()
    manifest = dict(experiment_id='P-20261001-direct-randomized-response', stage='FIT', run_id=output.name,
        run_status='RUNNING', pid=os.getpid(), gpu=gpu, input_sha256=hashes, seeds=list(SEEDS), phases=[],
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        wall_limit_seconds=900, storage_limit_bytes=100*(1<<20), fit_only_acquisition_seeds=[492,493])
    manifest['historical_native_source_sha256'] = {p: prior['input_sha256'][p] for p in historical_sources}
    def save():
        manifest['wall_seconds'] = time.monotonic()-begin
        (output/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    def check():
        if time.monotonic()-begin > 870: raise TimeoutError('fit budget')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > 100*(1<<20): raise RuntimeError('storage budget')
    save()
    try:
        bridge, _ = bridge_and_limits(); parts = []; rows = []
        for actor, evaluation in prior['panels']:
            folder = PRIOR/f't{actor}_s{evaluation}'; filename = folder/'windows.pt'
            record = json.loads((folder/'results.json').read_text())
            if not record['all_windows_complete'] or sha(filename) != record['windows_sha256']:
                raise ValueError('incomplete/drifted fit panel')
            hashes[str(filename.resolve())] = sha(filename)
            data = torch.load(filename, weights_only=False, map_location='cpu')
            if not data['complete'].all() or len(data['state']) != 768: raise ValueError('fit coverage')
            x = context(data['state'], data['reference_action'], bridge)
            y = (data['next_state'][:,36:39]-data['state'][:,36:39]-data['state'][:,43:46]/30).cuda()
            parts.append(dict(x=x, y=y, arm=data['arm'].cuda(), propensity=data['propensity'].cuda(),
                              acquisition_seed=torch.full((768,), evaluation, device=x.device)))
            rows.extend(dict(actor=actor, acquisition_seed=evaluation, environment=int(e), motion=int(m))
                        for e,m in zip(data['environment'],data['motion']))
        data = {key: torch.cat([part[key] for part in parts]) for key in parts[0]}
        unadjusted = pseudo_contrast(data['y'], data['arm'], data['propensity'])
        saved = {key: value.cpu() for key,value in data.items()}; saved.update(rows=rows, unadjusted_pseudo=unadjusted.cpu())
        for seed in SEEDS:
            nuisance = torch.zeros_like(data['y']); fold_records = []
            for heldout in (492,493):
                training = data['acquisition_seed'] != heldout; held = ~training; start = time.monotonic()
                checkpoint = fit(data['x'][training],data['y'][training],seed,1000,check)
                checkpoint.update(method='nuisance', heldout_acquisition_seed=heldout,
                    fit_acquisition_seed=493 if heldout==492 else 492)
                torch.save(checkpoint,output/f'nuisance_s{seed}_held{heldout}.pt')
                nuisance[held] = predict(data['x'][held],checkpoint)
                fold_records.append(dict(heldout=heldout,fit_rows=int(training.sum()),held_rows=int(held.sum()),wall_seconds=time.monotonic()-start))
            target = pseudo_contrast(data['y']-nuisance,data['arm'],data['propensity'])
            saved[f'nuisance_s{seed}'] = nuisance.cpu(); saved[f'pseudo_s{seed}'] = target.cpu()
            saved[f'global_s{seed}'] = target.mean(0).cpu()
            manifest['phases'].append(dict(seed=seed,method='crossfit_nuisance',folds=fold_records))
            for method, x, y, updates in (
                ('direct_contrast',data['x'],target.flatten(1),1000),
                ('factual',torch.cat((data['x'],indicators(data['arm'])),-1),data['y'],1000),
                ('factual_compute_matched',torch.cat((data['x'],indicators(data['arm'])),-1),data['y'],3000)):
                start = time.monotonic(); checkpoint = fit(x,y,seed,updates,check); checkpoint['method'] = method
                torch.save(checkpoint,output/f'{method}_s{seed}.pt')
                phase = dict(seed=seed,method=method,updates=updates,wall_seconds=time.monotonic()-start)
                manifest['phases'].append(phase); save(); print(json.dumps(phase),flush=True)
        torch.save(saved,output/'fit_data.pt'); check()
        if any(sha(Path(p)) != h for p,h in hashes.items()): raise ValueError('fit input/source drift')
        manifest.update(run_status='COMPLETED', rows=len(rows), inputs_unchanged=True,
            output_sha256={str(p.resolve()):sha(p) for p in output.glob('*.pt')})
    except BaseException as error:
        manifest.update(run_status='FAILED',error=repr(error)); raise
    finally: save()


if __name__ == '__main__': main()
