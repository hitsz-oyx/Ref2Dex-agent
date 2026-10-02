#!/usr/bin/env python3
"""GPU fit/cal qualification of a centered geometry-dependent physical response."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import gpu_admission,sha


def metrics(pred,target,cv_error,torch):
    from src.task.CmResidual.randomized_motor_response import TARGET_SCALE
    scale=torch.tensor(TARGET_SCALE,dtype=torch.float64,device=pred.device)
    error=(pred-target)*scale
    return dict(velocity_vector_rmse_mps=float(error[:,:3].square().sum(-1).mean().sqrt()),
        displacement_vector_rmse_mm=float(error[:,3:6].square().sum(-1).mean().sqrt()*1000),
        clearance_rmse_mm=float(error[:,6].square().mean().sqrt()*1000),
        cv_velocity_vector_rmse_mps=float(cv_error.square().sum(-1).mean().sqrt()))


def run(args):
    begin=time.monotonic();base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    output=args.output.resolve()
    if output.parent!=base or output.exists() or args.output.is_symlink():raise ValueError('owned unique qualification output')
    source=base/'P-20261002-contact-geometry-source-r1'
    source_manifest=source/'run_manifest.json';m=json.loads(source_manifest.read_text())
    audit_path=ROOT/'docs/experiments/probes/P-20261002-contact-geometry-source-audit-r1.json'
    audit=json.loads(audit_path.read_text())
    if m['run_status']!='COMPLETED' or m['smoke_only'] or audit['run_status']!='COMPLETED' or len(audit['phases'])!=12:
        raise ValueError('terminal audited original randomized source')
    paths=[Path(__file__),ROOT/'scripts/audit_randomized_motor_response.py',
        ROOT/'src/task/CmResidual/randomized_motor_response.py',ROOT/'src/task/CmResidual/native_pd_selector.py',
        ROOT/'src/task/CmResidual/executable_contact_options.py',source_manifest,audit_path,
        ROOT/'docs/decisions/D-20261002-randomized-motor-effects.md',ROOT/'docs/experiments/probes/P-20261002-randomized-motor-response.md']
    phases=m['phases']
    if [p['seed'] for p in phases]!=list(range(571,583)):raise ValueError('fixed old12 source seeds')
    for p in phases:
        path=Path(p['directory'])/'records.pt'
        if p['run_status']!='COMPLETED' or p['exit_code'] or sha(path)!=p['result']['record_sha256']:
            raise ValueError('terminal source records')
        paths.append(path)
    hashes={str(p.resolve()):sha(p) for p in paths};admission=gpu_admission(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    import torch
    from src.task.CmResidual.randomized_motor_response import rows,normalization,fit,predict,design,TARGET_SCALE
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    output.mkdir(exist_ok=False)
    manifest=dict(experiment_id='P-20261002-randomized-motor-response',family='HD04',run_status='RUNNING',pid=os.getpid(),
        input_sha256=hashes,admission=admission,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        old_source_wall_seconds=m['cumulative_seconds'],source_reused=True,no_new_simulation=True,held_targets_used=False,
        wall_limit_seconds=300,output_limit_bytes=64<<20,preparation_budget_seconds=30.)
    def save():
        manifest['elapsed_seconds']=time.monotonic()-begin;manifest['cumulative_seconds']=manifest['elapsed_seconds']+30.
        (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if time.monotonic()-begin+30>300:raise TimeoutError('whole linear-response qualification budget')
        if any(sha(Path(k))!=v for k,v in hashes.items()):raise ValueError('fixed fit inputs drift')
    save()
    try:
        batches=[]
        for phase in phases:
            batches.append(rows(torch.load(Path(phase['directory'])/'records.pt',map_location='cpu',weights_only=False)))
        keys=('state','geometry','centered_candidates','centered_action','target','probability','chosen','bucket','dynamic','cv_velocity_error','gravity_prior')
        data={k:torch.cat([b[k] for b in batches]).cuda() for k in keys}
        episode=np.array(sum([b['episode'] for b in batches],[]));group=np.array(sum([b['group'] for b in batches],[]))
        fit_mask=data['bucket']<50;cal=~fit_mask;norm=normalization(data,fit_mask)
        coefficient=fit(data,fit_mask,norm);pred,candidates=predict(data,norm,coefficient)
        expectation=(candidates*data['probability'][:,:,None]).sum(1)
        centering_error=float((expectation-pred['state_only']).abs().max())
        if centering_error>1e-10:raise ValueError('candidate effects violate conditional centering')
        effects=(data['centered_candidates']*data['probability'][:,:,None]).sum(1)
        if effects.abs().max()>1e-12:raise ValueError('centered native action expectation not zero')
        zero=norm['action_rms']*0
        unchanged={k:p.detach().clone() for k,p in pred.items()}
        # A zero centered action must predict exactly the shared state response.
        zero_data=dict(data,centered_action=zero.expand_as(data['centered_action']))
        neutral,_=predict(zero_data,norm,coefficient)
        if not torch.equal(neutral['cm'],neutral['state_only']):raise ValueError('zero action must have zero model effect')
        scores={};dynamic=cal&data['dynamic']
        for subset,mask in (('all_cal',cal),('dynamic_cal',dynamic)):
            scores[subset]={mode:metrics(value[mask],data['target'][mask],data['cv_velocity_error'][mask],torch)
                           for mode,value in pred.items()} if mask.any() else None
        cal_cpu=cal.cpu().numpy();dynamic_cpu=dynamic.cpu().numpy()
        support=dict(fit_rows=int(fit_mask.sum()),cal_rows=int(cal.sum()),cal_episodes=len(set(episode[cal_cpu])),
            cal_groups=len(set(group[cal_cpu])),dynamic_cal_rows=int(dynamic.sum()),
            total_source_rows=sum(b['total_rows_including_excluded_held'] for b in batches),excluded_held_rows=sum(b['total_rows_including_excluded_held'] for b in batches)-len(episode))
        adequate=support['cal_rows']>=256 and support['cal_episodes']>=24 and support['cal_groups']>=8 and support['dynamic_cal_rows']>=64
        gates=dict(support=adequate)
        for subset in ('all_cal','dynamic_cal'):
            local=scores[subset]
            gates[subset+'_velocity_action_information']=local is not None and all(local['cm']['velocity_vector_rmse_mps']<=.9*local[control]['velocity_vector_rmse_mps'] for control in ('state_only','shuffled'))
        local=scores['all_cal']
        for key in ('displacement_vector_rmse_mm','clearance_rmse_mm'):
            gates[key+'_action_information']=all(local['cm'][key]<=.95*local[control][key] for control in ('state_only','shuffled'))
        gates['velocity_cv_nonregression']=local['cm']['velocity_vector_rmse_mps']<=local['cm']['cv_velocity_vector_rmse_mps']
        label='PROMISING' if all(gates.values()) else ('UNPROMISING' if adequate else 'UNCLEAR')
        normalized=data['centered_action'][fit_mask]/norm['action_rms']
        eigen=torch.linalg.eigvalsh(normalized.T@normalized/len(normalized))
        # Descriptive paired group bootstrap, retaining actual support distribution.
        error=((pred['cm']-data['target'])*torch.tensor(TARGET_SCALE,device='cuda')).cpu().numpy()
        control=((pred['state_only']-data['target'])*torch.tensor(TARGET_SCALE,device='cuda')).cpu().numpy()
        squared=(error[:,:3]**2).sum(-1);control_squared=(control[:,:3]**2).sum(-1)
        labels=np.unique(group[cal_cpu]);samples=[];rng=np.random.default_rng(16672)
        indices=[np.flatnonzero(cal_cpu&(group==g)) for g in labels]
        for _ in range(500):
            ix=np.concatenate([indices[k] for k in rng.integers(0,len(labels),len(labels))])
            samples.append(float(np.sqrt(squared[ix].mean())-np.sqrt(control_squared[ix].mean())))
        bootstrap=np.quantile(samples,[.05,.95]).tolist()
        check()
        bundle=dict(schema='ref2dex.randomized_motor_response.v1',normalization={k:v.cpu() for k,v in norm.items()},
            coefficients={k:v.cpu() for k,v in coefficient.items()},data={k:v.cpu() for k,v in data.items()},episode=episode.tolist(),group=group.tolist(),
            predictions={k:v.cpu() for k,v in pred.items()},candidate_predictions=candidates.cpu(),
            input_sha256=hashes,held_targets_used=False,scope='experimental linear physical response; not a controller')
        torch.save(bundle,output/'response.pt')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>64<<20:raise ValueError('qualification disk budget')
        result=dict(experiment_id=manifest['experiment_id'],run_status='COMPLETED',label=label,support=support,metrics=scores,gates=gates,
            centering_error=centering_error,neutral_exact_state=True,initial_native_pd_max_error=max(float(b['initial_native_pd_error']) for b in batches),
            normalized_action_covariance_eigenvalues=eigen.cpu().tolist(),descriptive_cal_group90_cm_minus_state_velocity_rmse=bootstrap,
            bundle_sha256=sha(output/'response.pt'),held_targets_used=False,model_device='GPU',ridge=.01,
            elapsed_seconds=time.monotonic()-begin,scope='reused fit/cal first-cycle physical response information only; no new actual candidate advantage, no task-risk/utility or actor claim')
        (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');manifest.update(run_status='COMPLETED',child_exit_code=0,result=result)
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as e:manifest.update(run_status='FAILED',error=repr(e));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=5);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
