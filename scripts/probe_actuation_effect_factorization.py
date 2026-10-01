#!/usr/bin/env python3
"""Fit explicit command/motion/effect controls on episode-disjoint archived data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scripts.run_contact_response_probe import admission,sha
from src.task.CmResidual.actuation_effect import (
    DT,geometric_transport,motion_features,position_residual_target,state_features,
)
from src.task.CmResidual.dexplore_cm_geometry import (
    DExploreCmv2GeometryBridge,dexplore_action_to_native_targets,native_joint_limits,
)
from src.task.CmResidual.physical_value_contract import validate_rows
from src.task.CmResidual.v118_planner import QUERY_LINKS,TIP_LINKS

SEEDS=(411,412,413)
METHODS=('state_only','raw_command','nominal_motion','learned_motion','oracle_motion')
COLLECTION=ROOT.parent/'Ref2Dex-agent/src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7'


def load_rows(path,inputs):
    result=json.loads((path/'results.json').read_text())
    if result['run_status']!='COMPLETED' or result['mode']!='collect':
        raise ValueError('requires complete physical collection')
    inputs[str((path/'results.json').resolve())]=sha(path/'results.json')
    parts={key:[] for key in ('state','next_state','action','episode_id','motion_id')}
    accepted={int(e['episode_id']) for e in result['per_episode']}
    for shard in result['shards']:
        filename=Path(shard['path'])
        if sha(filename)!=shard['sha256']:raise ValueError('collection shard drift')
        inputs[str(filename.resolve())]=shard['sha256']
        data=torch.load(filename,map_location='cpu',weights_only=False)
        validate_rows(data)
        chosen=(data['step'].remainder(16)==0)&~data['done'].bool()
        for key in parts:parts[key].append(data[key][chosen])
        del data
    rows={key:torch.cat(values) for key,values in parts.items()}
    if not set(rows['episode_id'].tolist())<=accepted:raise ValueError('incomplete source episode')
    return rows


def network(dim,output,seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(dim,128),torch.nn.SiLU(),
                               torch.nn.Linear(128,128),torch.nn.SiLU(),
                               torch.nn.Linear(128,output)).cuda()


def score(prediction,target,near,episodes):
    errors=prediction-target
    def part(mask):
        e=errors[mask]
        return dict(rows=len(e),episodes=int(episodes[mask].unique().numel()),
                    rmse_mm=float(e.square().sum(-1).mean().sqrt()*1000),
                    epe_mm=float(e.norm(dim=-1).mean()*1000),
                    q95_mm=float(torch.quantile(e.norm(dim=-1),.95)*1000))
    return dict(all=part(torch.ones(len(target),dtype=torch.bool,device=target.device)),near=part(near))


def features(rows,bridge,limits,indices,check):
    out={key:[] for key in ('state_features','raw_command','nominal_motion','oracle_motion',
                            'actuator_x','actuator_y','q','dq','rotation','current_tips','target',
                            'gap','center_distances','episode','motion')}
    for start in range(0,len(rows['state']),256):
        check()
        state=rows['state'][start:start+256].cuda()
        future=rows['next_state'][start:start+256].cuda()
        action=rows['action'][start:start+256].cuda()
        geometry=bridge.current(state[:,:18],state[:,36:49])
        rotation=geometry.object_pose[:,:3,:3]
        tips=geometry.link_poses[:,indices,:3,3]
        base=state_features(state,tips,rotation)
        targets=dexplore_action_to_native_targets(action,state[:,:18],*limits)
        target_tips=bridge.kinematics.forward(targets[:,None])[:,0,indices,:3,3]
        actual_tips=bridge.kinematics.forward(future[:,None,:18])[:,0,indices,:3,3]
        # Stride8 geometry follows the existing approach-gap convention.
        gap=torch.cdist(geometry.hand_points[:,::8],geometry.object_points[:,::8]).amin(dim=(1,2))
        center_distances=torch.cdist(tips,geometry.object_points).amin(dim=-1)
        values=dict(state_features=base,raw_command=action,
                    nominal_motion=motion_features(tips,target_tips,rotation),
                    oracle_motion=motion_features(tips,actual_tips,rotation),
                    actuator_x=torch.cat((state[:,:36],targets-state[:,:18],base),-1),
                    actuator_y=future[:,:18]-state[:,:18]-state[:,18:36]*DT,
                    q=state[:,:18],dq=state[:,18:36],rotation=rotation,current_tips=tips,
                    target=position_residual_target(state,future,rotation),gap=gap,
                    center_distances=center_distances,
                    episode=rows['episode_id'][start:start+256].cuda(),motion=rows['motion_id'][start:start+256].cuda())
        for key,value in values.items():
            if value.dtype.is_floating_point and not torch.isfinite(value).all():
                raise FloatingPointError('nonfinite '+key)
            out[key].append(value)
    result={key:torch.cat(values) for key,values in out.items()}
    result['near']=result['gap']<=.02
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=4)
    args=parser.parse_args()
    if ROOT not in args.output.resolve().parents or args.output.exists():
        raise ValueError('unique output in isolated worktree required')
    gpu=admission(args.gpu)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu['uuid']:
        raise ValueError('CUDA visibility must match admitted GPU')
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True)
    begin=time.monotonic()
    asset=ROOT/'third_party/DExplore/dexplore/data/assets'
    hand=asset/'inspire_hand_new/inspire_hand_right.urdf';obj=asset/'mjcf/airplane.urdf'
    source_paths=[Path(__file__).resolve(),ROOT/'src/task/CmResidual/actuation_effect.py',
                  ROOT/'src/task/CmResidual/dexplore_cm_geometry.py',ROOT/'src/task/CmResidual/v118_planner.py',
                  ROOT/'third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/cm_geometry.py',
                  ROOT/'docs/experiments/probes/P-20261001-actuation-effect-factorization.md',hand,obj]
    inputs={str(p.resolve()):sha(p) for p in source_paths}
    manifest=dict(experiment_id='P-20261001-actuation-effect-factorization',run_id=args.output.name,
                  run_status='RUNNING',pid=os.getpid(),command=sys.argv,gpu=gpu,input_sha256=inputs,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  fixed_updates=1000,seeds=list(SEEDS),phases=[],wall_limit_seconds=900,
                  reused_data=True,no_policy_training=True)
    def save():
        manifest['wall_seconds']=time.monotonic()-begin
        (args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if time.monotonic()-begin>870:raise TimeoutError('whole factorization budget')
    save()
    try:
        fit_raw=load_rows(COLLECTION/'collect_s283',inputs)
        test_raw=load_rows(COLLECTION/'collect_s284',inputs)
        if set(fit_raw['episode_id'].tolist())&set(test_raw['episode_id'].tolist()):
            raise ValueError('episode overlap')
        bridge=DExploreCmv2GeometryBridge(hand_urdf=hand,object_urdf=obj,device='cuda:0',seed=42)
        limits=native_joint_limits(hand,'cuda:0')
        indices=[QUERY_LINKS.index(name) for name in ('hand_base_link',*TIP_LINKS)]
        with torch.no_grad():
            fit=features(fit_raw,bridge,limits,indices,check)
            test=features(test_raw,bridge,limits,indices,check)
        del fit_raw,test_raw
        save()
        coverage=dict(fit_rows=len(fit['target']),test_rows=len(test['target']),
                      fit_near_rows=int(fit['near'].sum()),test_near_rows=int(test['near'].sum()),
                      test_near_episodes=int(test['episode'][test['near']].unique().numel()))
        print(json.dumps(dict(phase='features_complete',coverage=coverage,wall_seconds=time.monotonic()-begin)),flush=True)
        if coverage['test_near_rows']<1024 or coverage['test_near_episodes']<64 or not fit['near'].any():
            report=dict(run_status='COMPLETED',label='UNCLEAR',reason='INSUFFICIENT_COVERAGE',coverage=coverage)
            (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
            manifest.update(run_status='COMPLETED',label='UNCLEAR');save();return
        torch.save({name:{k:v.cpu() for k,v in values.items()} for name,values in (('fit',fit),('test',test))},args.output/'features.pt')
        target_mean=fit['target'].mean(0);target_std=fit['target'].std(0).clamp_min(1e-4)
        target=(fit['target']-target_mean)/target_std
        passive=score(torch.zeros_like(test['target']),test['target'],test['near'],test['episode'])
        near_fit=fit['near'].nonzero().flatten()
        results=[];actuator_results=[];predictions={};geometric={}
        for kind in ('nominal_motion','oracle_motion'):
            predicted=geometric_transport(test[kind],test['center_distances'],test['state_features'][:,33:36])
            geometric[kind]=score(predicted,test['target'],test['near'],test['episode'])
        for seed in SEEDS:
            check();stage_begin=time.monotonic()
            xmean=fit['actuator_x'].mean(0);xstd=fit['actuator_x'].std(0).clamp_min(.01)
            ymean=fit['actuator_y'].mean(0);ystd=fit['actuator_y'].std(0).clamp_min(1e-4)
            actuator=network(len(xmean),18,seed)
            optimizer=torch.optim.Adam(actuator.parameters(),lr=.001)
            sampler=torch.Generator(device='cuda').manual_seed(seed+10000)
            x=(fit['actuator_x']-xmean)/xstd;y=(fit['actuator_y']-ymean)/ystd
            for update in range(1000):
                if update%100==0:check()
                ids=torch.randint(len(x),(256,),device='cuda',generator=sampler)
                loss=(actuator(x[ids])-y[ids]).square().mean()
                if not torch.isfinite(loss):raise FloatingPointError('actuator loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            actuator.eval()
            with torch.no_grad():
                for data in (fit,test):
                    all_delta=torch.cat([actuator(block)*ystd+ymean for block in ((data['actuator_x']-xmean)/xstd).split(1024)])
                    q_next=data['q']+data['dq']*DT+all_delta
                    displacements=[]
                    for start in range(0,len(q_next),512):
                        tip_next=bridge.kinematics.forward(q_next[start:start+512,None])[:,0,indices,:3,3]
                        displacements.append(motion_features(data['current_tips'][start:start+512],tip_next,data['rotation'][start:start+512]))
                    data['learned_motion']=torch.cat(displacements)
                    if data is test:
                        joint_error=all_delta-data['actuator_y']
                        near=data['near']
                        actuator_results.append(dict(seed=seed,wall_seconds=time.monotonic()-stage_begin,
                             near_wrist_translation_rmse_mm=float(joint_error[near,:3].square().sum(-1).mean().sqrt()*1000),
                             near_angular_rmse_rad=float(joint_error[near,3:].square().mean().sqrt())))
            geometric[str(seed)]=score(geometric_transport(test['learned_motion'],test['center_distances'],
                                      test['state_features'][:,33:36]),test['target'],test['near'],test['episode'])
            torch.save(dict(model={k:v.cpu() for k,v in actuator.state_dict().items()},xmean=xmean.cpu(),xstd=xstd.cpu(),
                            ymean=ymean.cpu(),ystd=ystd.cpu()),args.output/f'actuator_s{seed}.pt')
            manifest['phases'].append(dict(kind='actuator',seed=seed,run_status='COMPLETED',wall_seconds=time.monotonic()-stage_begin))
            del actuator,optimizer
            for method in METHODS:
                check();stage_begin=time.monotonic()
                def effect_x(data):
                    command=torch.zeros_like(data['raw_command']) if method=='state_only' else data[method]
                    return torch.cat((data['state_features'],command),-1)
                fx,tx=effect_x(fit),effect_x(test)
                mean=fx.mean(0);std=fx.std(0).clamp_min(.001)
                fx=(fx-mean)/std;tx=(tx-mean)/std
                model=network(fx.shape[1],3,seed)
                optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                sampler=torch.Generator(device='cuda').manual_seed(seed+20000)
                losses=[]
                for update in range(1000):
                    if update%100==0:check()
                    a=near_fit[torch.randint(len(near_fit),(128,),device='cuda',generator=sampler)]
                    b=torch.randint(len(fx),(128,),device='cuda',generator=sampler)
                    ids=torch.cat((a,b))
                    loss=(model(fx[ids])-target[ids]).square().mean()
                    if not torch.isfinite(loss):raise FloatingPointError('effect loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                    if update in (0,499,999):losses.append(dict(update=update+1,loss=float(loss.detach())))
                model.eval()
                with torch.no_grad():prediction=torch.cat([model(block)*target_std+target_mean for block in tx.split(2048)])
                result=dict(seed=seed,method=method,metrics=score(prediction,test['target'],test['near'],test['episode']),
                            losses=losses,wall_seconds=time.monotonic()-stage_begin)
                results.append(result);predictions[f'{method}_s{seed}']=prediction.cpu()
                torch.save(dict(model={k:v.cpu() for k,v in model.state_dict().items()},mean=mean.cpu(),std=std.cpu(),
                                target_mean=target_mean.cpu(),target_std=target_std.cpu(),seed=seed,method=method),
                           args.output/f'effect_{method}_s{seed}.pt')
                manifest['phases'].append(dict(kind='effect',seed=seed,method=method,run_status='COMPLETED',
                                              wall_seconds=time.monotonic()-stage_begin));save()
                print(json.dumps(result),flush=True)
                del model,optimizer
        averaged={method:{scope:{metric:sum(r['metrics'][scope][metric] for r in results if r['method']==method)/3
                                for metric in ('rmse_mm','epe_mm','q95_mm')}
                          for scope in ('all','near')} for method in METHODS}
        learned_geometric=sum(geometric[str(seed)]['near']['rmse_mm'] for seed in SEEDS)/3
        learned=averaged['learned_motion']['near']['rmse_mm']
        controls=dict(state_only=averaged['state_only']['near']['rmse_mm'],
                      raw_command=averaged['raw_command']['near']['rmse_mm'],
                      passive=passive['near']['rmse_mm'],learned_geometric=learned_geometric)
        gates={f'beat_{kind}10pct':learned<=.9*value for kind,value in controls.items()}
        gates['nonnegative_vs_raw_each_seed']=all(
            next(r for r in results if r['seed']==seed and r['method']=='learned_motion')['metrics']['near']['rmse_mm']<=
            next(r for r in results if r['seed']==seed and r['method']=='raw_command')['metrics']['near']['rmse_mm']
            for seed in SEEDS)
        report=dict(experiment_id=manifest['experiment_id'],run_id=manifest['run_id'],run_status='COMPLETED',
                    label='PROMISING' if all(gates.values()) else 'UNPROMISING',coverage=coverage,gates=gates,
                    averaged=averaged,passive=passive,geometric=geometric,actuator_results=actuator_results,
                    results=results,input_sha256=inputs,wall_seconds=time.monotonic()-begin,
                    boundary='reused-data one-step physical predictive Probe; oracle uses future hand state; no policy utility')
        torch.save(dict(predictions=predictions,target=test['target'].cpu(),near=test['near'].cpu(),
                        episode=test['episode'].cpu(),motion=test['motion'].cpu()),args.output/'predictions.pt')
        if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())>500*(1<<20):
            raise RuntimeError('storage budget')
        if any(sha(Path(p))!=h for p,h in inputs.items()):raise ValueError('input/source drift')
        (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        manifest.update(run_status='COMPLETED',label=report['label'],input_sha256=inputs,inputs_unchanged=True)
        print(json.dumps(dict(label=report['label'],gates=gates,averaged=averaged,passive=passive)),flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
