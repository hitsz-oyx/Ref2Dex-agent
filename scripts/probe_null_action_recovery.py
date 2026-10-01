#!/usr/bin/env python3
"""Test predictor inverse recovery against exact native null interventions."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.run_contact_response_probe import admission,sha
from src.task.CmResidual.dexplore_cm_geometry import native_joint_limits,dexplore_action_to_native_targets

NULL=(7,9,11,13,16,17)
EFFECTIVE=tuple(i for i in range(18) if i not in NULL)
METHODS=('factual','full_inverse','effective_inverse','quotient_inverse')
SEEDS=(511,512,513)
DATA=ROOT/'src/task/CmResidual/research/contact_response/output/P-20261001-actuation-effect-factorization-r1/features.pt'


def net(dim,out,seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(dim,128),torch.nn.SiLU(),
        torch.nn.Linear(128,128),torch.nn.SiLU(),torch.nn.Linear(128,out)).cuda()


def randomized(command,generator):
    action=command.clone()
    action[:,NULL]=torch.rand((len(action),6),device='cuda',generator=generator)*2-1
    return action


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu',type=int,default=5)
    args=parser.parse_args();output=args.output.resolve()
    if output.exists() or ROOT not in output.parents:raise ValueError('unique independent output required')
    gpu=admission(args.gpu)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu['uuid']:raise ValueError('admitted GPU visibility mismatch')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    hand=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    sources=[Path(__file__),DATA,hand,ROOT/'src/task/CmResidual/dexplore_cm_geometry.py',
             ROOT/'docs/experiments/probes/P-20261001-null-action-recovery.md']
    hashes={str(p.resolve()):sha(p) for p in sources};output.mkdir(parents=True);begin=time.monotonic()
    manifest=dict(experiment_id='P-20261001-null-action-recovery',run_id=output.name,run_status='RUNNING',pid=os.getpid(),
        gpu=gpu,input_sha256=hashes,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        seeds=SEEDS,phases=[],wall_limit_seconds=900,storage_limit_bytes=100*(1<<20),no_physics_collection=True)
    def save():
        manifest['wall_seconds']=time.monotonic()-begin
        (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if time.monotonic()-begin>870:raise TimeoutError('null-recovery budget')
    save()
    try:
        saved=torch.load(DATA,map_location='cpu',weights_only=False)
        fit={k:saved['fit'][k].cuda() for k in ('state_features','raw_command','target','near')}
        test={k:saved['test'][k].cuda() for k in ('state_features','raw_command','target','near','q')}
        episode=saved['test']['episode'];del saved
        sm=fit['state_features'].mean(0);ss=fit['state_features'].std(0).clamp_min(.001)
        am=fit['raw_command'].mean(0);ass=fit['raw_command'].std(0).clamp_min(.001)
        am[list(NULL)]=0;ass[list(NULL)]=1/math.sqrt(3)
        ym=fit['target'].mean(0);ys=fit['target'].std(0).clamp_min(1e-4)
        fs=(fit['state_features']-sm)/ss;ts=(test['state_features']-sm)/ss;fy=(fit['target']-ym)/ys
        nearfit=fit['near'].nonzero().flatten();near=test['near']
        ta=randomized(test['raw_command'],torch.Generator(device='cuda').manual_seed(999))
        tb=randomized(test['raw_command'],torch.Generator(device='cuda').manual_seed(1000))
        limits=native_joint_limits(hand,'cuda:0')
        for start in range(0,len(ta),2048):
            a,b=ta[start:start+2048],tb[start:start+2048];q=test['q'][start:start+2048]
            if not torch.equal(dexplore_action_to_native_targets(a,q,*limits),dexplore_action_to_native_targets(b,q,*limits)):
                raise ValueError('claimed null coordinates change PD targets')
        manifest['exact_pd_null_invariance_verified_rows']=len(ta);save()
        results=[];predictions={};recoveries={}
        for seed in SEEDS:
            for method in METHODS:
                check();start=time.monotonic()
                model=net(81,3,seed);head=net(66,18,seed+1000)
                parameters=list(model.parameters())+(list(head.parameters()) if method!='factual' else [])
                optimizer=torch.optim.Adam(parameters,lr=.001)
                sampler=torch.Generator(device='cuda').manual_seed(seed+30000)
                inverse_mask=torch.ones(18,device='cuda')
                if method in ('effective_inverse','quotient_inverse'):inverse_mask[list(NULL)]=0
                losses=[]
                for update in range(1000):
                    if update%100==0:check()
                    a=nearfit[torch.randint(len(nearfit),(128,),device='cuda',generator=sampler)]
                    b=torch.randint(len(fs),(128,),device='cuda',generator=sampler);ids=torch.cat((a,b))
                    raw=randomized(fit['raw_command'][ids],sampler);an=(raw-am)/ass
                    ax=an.clone()
                    if method=='quotient_inverse':ax[:,NULL]=0
                    predicted=model(torch.cat((fs[ids],ax),-1))
                    factual_loss=(predicted-fy[ids]).square().mean()
                    inverse_loss=torch.zeros((),device='cuda')
                    if method!='factual':
                        restored=head(torch.cat((fs[ids],predicted),-1))
                        inverse_loss=((restored-an).square()*inverse_mask).mean()
                    loss=factual_loss+inverse_loss
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite null recovery fit')
                    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                    if update in (0,499,999):losses.append(dict(update=update+1,factual=float(factual_loss.detach()),inverse=float(inverse_loss.detach())))
                model.eval();head.eval()
                with torch.no_grad():
                    ysides=[];heads=[]
                    for raw in (ta,tb):
                        an=(raw-am)/ass;ax=an.clone()
                        if method=='quotient_inverse':ax[:,NULL]=0
                        pn=torch.cat([model(block) for block in torch.cat((ts,ax),-1).split(2048)])
                        ysides.append(pn*ys+ym)
                        if method!='factual':heads.append(torch.cat([head(block) for block in torch.cat((ts,pn),-1).split(2048)]))
                    null_delta=ysides[0]-ysides[1]
                    def scope(mask):
                        count=int(mask.sum())
                        result=dict(rows=count,factual_rmse_mm=float(torch.cat([p[mask]-test['target'][mask] for p in ysides]).square().sum(-1).mean().sqrt()*1000),
                                    null_prediction_rms_mm=float(null_delta[mask].square().sum(-1).mean().sqrt()*1000))
                        if heads:
                            error=torch.cat([(head_output[mask]-(raw[mask]-am)/ass).square() for head_output,raw in zip(heads,(ta,tb))])
                            result.update(inverse_full_mse=float(error.mean()),inverse_null_mse=float(error[:,NULL].mean()),
                                          inverse_effective_mse=float(error[:,EFFECTIVE].mean()))
                        return result
                    metrics=dict(all=scope(torch.ones(len(ta),dtype=torch.bool,device='cuda')),near=scope(near))
                row=dict(seed=seed,method=method,metrics=metrics,losses=losses,wall_seconds=time.monotonic()-start)
                results.append(row);predictions[f'{method}_s{seed}']=torch.stack(ysides).cpu()
                if heads:recoveries[f'{method}_s{seed}']=torch.stack(heads).cpu()
                torch.save(dict(model={k:v.cpu() for k,v in model.state_dict().items()},head={k:v.cpu() for k,v in head.state_dict().items()},
                    sm=sm.cpu(),ss=ss.cpu(),am=am.cpu(),action_std=ass.cpu(),ym=ym.cpu(),ys=ys.cpu(),seed=seed,method=method),output/f'{method}_s{seed}.pt')
                manifest['phases'].append(dict(seed=seed,method=method,run_status='COMPLETED',wall_seconds=row['wall_seconds']));save()
                print(json.dumps(row),flush=True)
                del model,head,optimizer
        average={method:{scope:{key:sum(row['metrics'][scope][key] for row in results if row['method']==method)/3
                         for key in next(row for row in results if row['method']==method)['metrics'][scope] if key!='rows'}
                         for scope in ('all','near')} for method in METHODS}
        full=average['full_inverse']['near'];factual=average['factual']['near']
        gates=dict(null_recovery_below_floor=full['inverse_null_mse']<=.8,
                   null_response_at_least1mm=full['null_prediction_rms_mm']>=1,
                   null_response_at_least_twice_factual=full['null_prediction_rms_mm']>=2*factual['null_prediction_rms_mm'],
                   greater_null_response_each_seed=all(next(row for row in results if row['method']=='full_inverse' and row['seed']==s)['metrics']['near']['null_prediction_rms_mm']>
                                                      next(row for row in results if row['method']=='factual' and row['seed']==s)['metrics']['near']['null_prediction_rms_mm'] for s in SEEDS),
                   retains_factual_fit=full['factual_rmse_mm']<=1.25*factual['factual_rmse_mm'])
        report=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',gates=gates,
                    averaged=average,results=results,exact_pd_null_invariance_verified_rows=len(ta),
                    population_null_inverse_mse_floor=1.,population_full_inverse_mse_floor=1/3,
                    input_sha256=hashes,wall_seconds=time.monotonic()-begin,
                    boundary='stylized physical-residual inverse objective; no AD-WM reproduction or policy claim')
        torch.save(dict(target=test['target'].cpu(),near=near.cpu(),episode=episode,predictions=predictions,recoveries=recoveries,
                        normalized_action_a=((ta-am)/ass).cpu(),normalized_action_b=((tb-am)/ass).cpu()),output/'predictions.pt')
        if any(sha(Path(p))!=h for p,h in hashes.items()):raise ValueError('input/source drift')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>100*(1<<20):raise ValueError('null recovery storage budget')
        (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        manifest.update(run_status='COMPLETED',label=report['label'],inputs_unchanged=True)
        print(json.dumps(dict(label=report['label'],gates=gates,averaged=average)),flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
