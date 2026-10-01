#!/usr/bin/env python3
"""Current-only acquisition and frozen-checkpoint physical contrast scoring."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scripts.run_contact_response_probe import admission,sha
from src.task.CmResidual.causal_acquisition import eligible_steps,first_geometry_trigger
from src.task.CmResidual.actuation_effect import DT,state_features,motion_features
from src.task.CmResidual.dexplore_cm_geometry import DExploreCmv2GeometryBridge,native_joint_limits,dexplore_action_to_native_targets
from src.task.CmResidual.v118_planner import QUERY_LINKS,TIP_LINKS


def bridge_and_limits():
    asset=ROOT/'third_party/DExplore/dexplore/data/assets'
    hand=asset/'inspire_hand_new/inspire_hand_right.urdf';obj=asset/'mjcf/airplane.urdf'
    return DExploreCmv2GeometryBridge(hand_urdf=hand,object_urdf=obj,device='cuda:0',seed=42),native_joint_limits(hand,'cuda:0')


@torch.no_grad()
def schedule(directory):
    filename=directory/'trace.pt'
    data=torch.load(filename,map_location='cpu',weights_only=False)
    eligible=eligible_steps(data['state_before'],data['done'])
    gap=torch.full(eligible.shape,float('inf'))
    ids=eligible.nonzero();bridge,_=bridge_and_limits()
    for selection in ids.split(64):
        state=data['state_before'][selection[:,0],selection[:,1]].cuda()
        geo=bridge.current(state[:,:18],state[:,36:49])
        distances=torch.cdist(geo.hand_points,geo.object_points).amin((1,2)).cpu()
        gap[selection[:,0],selection[:,1]]=distances
    triggers=first_geometry_trigger(eligible,gap)
    env=(triggers>=0).nonzero().flatten()
    current=data['state_before'][triggers[env],env]
    selected_gap=gap[triggers[env],env]
    result=dict(triggers=triggers,gaps=selected_gap,current_state=current,environment=env,
                trace_sha256=sha(filename),geometry='full1538x1024 seed42 current state',
                coverage=dict(valid_windows=len(env),by_motion={str(i):int((data['motion_id'][env]==i).sum()) for i in range(3)},
                              candidate_force_states=len(ids),max_gap_mm=float(selected_gap.max()*1000) if len(env) else None))
    torch.save(result,directory/'schedule.pt')
    (directory/'schedule.json').write_text(json.dumps(result['coverage'],indent=2)+'\n')
    print(json.dumps(result['coverage']),flush=True)


def take(data,env,horizon):
    ticks=data['triggers'][env]
    before=data['state_before'][ticks,env]
    after=data['state_after'][ticks+horizon-1,env]
    return before,after,data['action'][ticks,env]


@torch.no_grad()
def predict(before,action,bridge,limits,checkpoint):
    state=before.cuda();action=action.cuda()
    geo=bridge.current(state[:,:18],state[:,36:49]);rotation=geo.object_pose[:,:3,:3]
    indices=[QUERY_LINKS.index(name) for name in ('hand_base_link',*TIP_LINKS)]
    tips=geo.link_poses[:,indices,:3,3]
    if checkpoint['method']=='nominal_motion':
        targets=dexplore_action_to_native_targets(action,state[:,:18],*limits)
        future_tips=bridge.kinematics.forward(targets[:,None])[:,0,indices,:3,3]
        ax=motion_features(tips,future_tips,rotation)
    elif checkpoint['method']=='raw_command':ax=action
    elif checkpoint['method']=='state_only':ax=torch.zeros_like(action)
    else:raise ValueError('unregistered checkpoint')
    x=torch.cat((state_features(state,tips,rotation),ax),-1)
    # One architecture shared with fitting; load only frozen factual checkpoints.
    model=torch.nn.Sequential(torch.nn.Linear(81,128),torch.nn.SiLU(),torch.nn.Linear(128,128),
                              torch.nn.SiLU(),torch.nn.Linear(128,3)).cuda().eval()
    model.load_state_dict(checkpoint['model'])
    local=model((x-checkpoint['mean'].cuda())/checkpoint['std'].cuda())*checkpoint['target_std'].cuda()+checkpoint['target_mean'].cuda()
    return (state[:,43:46]*DT+torch.einsum('bi,bji->bj',local,rotation)).cpu()


def error_score(prediction,target):
    err=prediction-target
    valid=target.norm(dim=-1)>1e-6
    cosine=torch.nn.functional.cosine_similarity(prediction[valid],target[valid],dim=-1,eps=1e-8)
    return dict(rows=len(target),rmse_mm=float(err.square().sum(-1).mean().sqrt()*1000),
                epe_mm=float(err.norm(dim=-1).mean()*1000),mean_cosine=float(cosine.mean()) if valid.any() else None,
                nonzero_target_rows=int(valid.sum()))


@torch.no_grad()
def analyze(output):
    manifest=json.loads((output/'run_manifest.json').read_text())
    if manifest['run_status']!='COLLECTION_COMPLETED':raise ValueError('collection is not terminal')
    frozen=Path(manifest['frozen_models']);bridge,limits=bridge_and_limits()
    measured=[];repeat=[];five=[];group=[];predictions={f'{method}_s{seed}':[] for method in ('nominal_motion','raw_command','state_only') for seed in (411,412,413)}
    agreement=[]
    for training,seed in manifest['panels']:
        prefix=f't{training}_s{seed}'
        ref=output/(prefix+'_reference');acq=torch.load(ref/'schedule.pt',weights_only=False,map_location='cpu')
        env=acq['environment'];env_motions=torch.load(ref/'trace.pt',weights_only=False,map_location='cpu')['motion_id'][env]
        runs={name:torch.load(output/(prefix+'_'+name)/'response.pt',weights_only=False,map_location='cpu') for name in
              ('zero_a','zero_b','x_plus','x_minus','y_plus','y_minus','z_plus','z_minus')}
        a,b,_=take(runs['zero_a'],env,1);base_a=a
        c,d,_=take(runs['zero_b'],env,1)
        repeat.append((b[:,36:39]-a[:,36:39])-(d[:,36:39]-c[:,36:39]))
        for name,data in runs.items():
            before,_,_=take(data,env,1)
            agreement.append(dict(panel=prefix,arm=name,
                object_position_rms_mm=float((before[:,36:39]-base_a[:,36:39]).square().sum(-1).mean().sqrt()*1000),
                max_joint_abs_error=float((before[:,:18]-base_a[:,:18]).abs().max()),
                max_object_orientation_abs_error=float((before[:,39:43]-base_a[:,39:43]).abs().max())))
            check=agreement[-1]
            if check['object_position_rms_mm']>.05 or check['max_joint_abs_error']>1e-4 or check['max_object_orientation_abs_error']>1e-4:
                raise ValueError('pre-intervention state agreement failed: '+json.dumps(check))
        for axis in ('x','y','z'):
            plus=runs[axis+'_plus'];minus=runs[axis+'_minus']
            ps,pe,pa=take(plus,env,1);ms,me,ma=take(minus,env,1)
            truth=(pe[:,36:39]-ps[:,36:39])-(me[:,36:39]-ms[:,36:39]);measured.append(truth)
            fs,fe,_=take(plus,env,5);gs,ge,_=take(minus,env,5)
            five.append((fe[:,36:39]-fs[:,36:39])-(ge[:,36:39]-gs[:,36:39]))
            group.extend([dict(panel=prefix,axis=axis,environment=int(e),motion=int(m)) for e,m in zip(env,env_motions)])
            for method in ('nominal_motion','raw_command','state_only'):
                for model_seed in (411,412,413):
                    checkpoint=torch.load(frozen/f'effect_{method}_s{model_seed}.pt',weights_only=False,map_location='cpu')
                    value=predict(ps,pa,bridge,limits,checkpoint)-predict(ms,ma,bridge,limits,checkpoint)
                    predictions[f'{method}_s{model_seed}'].append(value)
        print(json.dumps(dict(panel=prefix,analyzed_windows=len(env))),flush=True)
    truth=torch.cat(measured);rep=torch.cat(repeat);five=torch.cat(five)
    predictions={k:torch.cat(v) for k,v in predictions.items()}
    metrics={key:error_score(value,truth) for key,value in predictions.items()}
    averaged={method:{metric:(sum(metrics[f'{method}_s{s}'][metric] for s in (411,412,413))/3
                             if all(metrics[f'{method}_s{s}'][metric] is not None for s in (411,412,413)) else None)
                      for metric in ('rmse_mm','epe_mm','mean_cosine')}
              for method in ('nominal_motion','raw_command','state_only')}
    pulse_rms=float(truth.square().sum(-1).mean().sqrt()*1000);repeat_rms=float(rep.square().sum(-1).mean().sqrt()*1000)
    zero=error_score(torch.zeros_like(truth),truth)
    nominal=averaged['nominal_motion']
    gates=dict(physical_resolution=pulse_rms>0 and pulse_rms>=2*repeat_rms,
               beat_raw10pct=nominal['rmse_mm']<=.9*averaged['raw_command']['rmse_mm'],
               beat_zero10pct=nominal['rmse_mm']<=.9*zero['rmse_mm'],cosine_at_least_half=nominal['mean_cosine'] is not None and nominal['mean_cosine']>=.5,
               nonnegative_vs_raw_each_seed=all(metrics[f'nominal_motion_s{s}']['rmse_mm']<=metrics[f'raw_command_s{s}']['rmse_mm'] for s in (411,412,413)))
    by_group={}
    for dimension in ('panel','axis','motion'):
        by_group[dimension]={}
        for value in sorted({row[dimension] for row in group},key=str):
            mask=torch.tensor([row[dimension]==value for row in group])
            by_group[dimension][str(value)]={key:error_score(pred[mask],truth[mask]) for key,pred in predictions.items()}
    result=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',gates=gates,
                averaged=averaged,metrics=metrics,zero_control=zero,pulse_rms_mm=pulse_rms,repeat_rms_mm=repeat_rms,
                pulse_repeat_ratio=pulse_rms/repeat_rms if repeat_rms>0 else None,
                five_step_pulse_rms_mm=float(five.square().sum(-1).mean().sqrt()*1000),
                pre_intervention_agreement=agreement,by_group=by_group,
                boundary='fresh-initialization causal Probe, one object, no training or policy utility')
    torch.save(dict(target=truth,repeat=rep,five_step=five,predictions=predictions,groups=group),output/'predictions.pt')
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('label','gates','averaged','pulse_rms_mm','repeat_rms_mm')}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('schedule','analyze'));parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();gpu=admission(4)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu['uuid']:raise ValueError('admitted GPU visibility')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    if args.mode=='schedule':schedule(args.directory)
    else:analyze(args.directory)


if __name__=='__main__':main()
