#!/usr/bin/env python3
"""Independent native-actuator, scalar-physics and GPU normal-equation audit."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import gpu_admission,sha


def run(args):
    admission=gpu_admission(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;begin=time.monotonic()
    manifest=json.loads((args.run/'run_manifest.json').read_text());result=json.loads((args.run/'results.json').read_text())
    if manifest['run_status']!='COMPLETED' or result['bundle_sha256']!=sha(args.run/'response.pt'):
        raise ValueError('terminal pinned fit bundle')
    if any(sha(Path(k))!=v for k,v in manifest['input_sha256'].items()):raise ValueError('fixed input drift')
    bundle=torch.load(args.run/'response.pt',map_location='cpu',weights_only=False)
    if bundle['held_targets_used'] or (bundle['data']['bucket']>=70).any():raise ValueError('held data entered fit/cal')
    source=ROOT/'src/task/CmResidual/research/contact_consequence/output/P-20261002-contact-geometry-source-r1'
    m=json.loads((source/'run_manifest.json').read_text());all_features=[];all_geometry=[];all_z=[];all_p=[];all_targets=[]
    all_actual_z=[];buckets=[];all_dynamic=[];all_cv=[];episodes=[];groups=[];excluded=0
    scale=np.array([.1,.1,.1,.005,.005,.005,.002]);independent=[0,1,2,3,4,5,6,8,10,12,14,15]
    for phase in m['phases']:
        path=Path(phase['directory'])/'records.pt'
        if sha(path)!=phase['result']['record_sha256']:raise ValueError('original source drift')
        b=torch.load(path,map_location='cpu',weights_only=False)
        for i in range(len(b['state'])):
            bucket=int(b['split_group_bucket'][i])
            if bucket>=70:excluded+=1;continue
            s=b['state'][i].double().numpy();post=b['future_state'][i,0].double().numpy()
            rel=s[36:39]-b['initial_key_positions'][i,0].double().numpy()
            rv=s[43:46]-b['initial_key_velocities'][i,0].double().numpy()
            mass=float(b['mass_kg'][i]);g=b['gravity_magnitude'];dt=b['control_dt_seconds']
            force=max(float(np.linalg.norm(v)) for v in b['initial_hand_force'][i].double().numpy())
            obj=float(np.linalg.norm(b['initial_object_force'][i].double().numpy()))
            logforce=np.log1p(np.array([force,obj])/(mass*g));clr=float(b['initial_clearance'][i]);rest=float(b['rest_z'][i]);height=max(s[38]-rest,0.)
            all_features.append(np.concatenate([s[:36],s[39:49],rel,logforce,[clr,height]]))
            all_geometry.append(np.concatenate([rel,rv,logforce,[clr,height]]))
            native=b['candidate_actions'][i].double().numpy().copy();native[:,6:]=(1+native[:,6:])/2
            pd=b['pd_offset'].double().numpy()+b['pd_scale'].double().numpy()*native;pd[:,:6]+=s[None,:6]
            for dst,src,ratio in [(7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)]:pd[:,dst]=pd[:,src]*ratio
            probability=np.array([.2,.2,.1,.1,.1,.1,.1,.1]);probability/=probability.sum()
            centered=pd[:,independent]-sum(probability[a]*pd[a,independent] for a in range(8))
            a=int(b['assignment'][i]);all_z.append(centered);all_p.append(probability);all_actual_z.append(centered[a])
            dv=post[43:46]-s[43:46];all_cv.append(dv);all_dynamic.append(np.linalg.norm(dv)>.05)
            dp=post[36:39]-s[36:39]-s[43:46]*dt
            delta_clr=float(b['future_clearance'][i,0])-clr-s[45]*dt
            all_targets.append(np.concatenate([dv-np.array([0.,0.,-g*dt]),dp,[delta_clr]])/scale)
            buckets.append(bucket);episodes.append(b['episode_id'][i]);groups.append(f"{int(b['motion_id'][i])}/{int(b['start_frame'][i])}")
    expected=dict(state=np.array(all_features),geometry=np.array(all_geometry),centered_candidates=np.array(all_z),
        centered_action=np.array(all_actual_z),probability=np.array(all_p),target=np.array(all_targets),cv_velocity_error=np.array(all_cv))
    errors={}
    def compare(key,x,y,tol=2e-8):
        error=float(np.max(np.abs(np.array(x)-np.array(y))));errors[key]=error
        if error>tol:raise ValueError(key+' independent mismatch '+str(error))
    for key,x in expected.items():compare('raw_'+key,x,bundle['data'][key].numpy())
    fit=np.array(buckets)<50;cal=~fit;dynamic=cal&np.array(all_dynamic)
    norm={}
    for key in ('state','geometry'):
        x=expected[key][fit];norm[key+'_mean']=x.mean(0);norm[key+'_std']=np.maximum(x.std(0),.001)
    norm['action_rms']=np.maximum(np.sqrt((expected['centered_action'][fit]**2).mean(0)),1e-4)
    for key,value in norm.items():compare('fit_normalization_'+key,value,bundle['normalization'][key].numpy())
    normalized_state=np.clip((expected['state']-norm['state_mean'])/norm['state_std'],-8,8)
    normalized_geometry=np.clip((expected['geometry']-norm['geometry_mean'])/norm['geometry_std'],-8,8)
    state=np.column_stack([np.ones(len(fit)),normalized_state]);phi=np.column_stack([np.ones(len(fit)),normalized_geometry])
    z=expected['centered_action']/norm['action_rms'];response=(phi[:,:,None]*z[:,None,:]).reshape(len(fit),-1)
    def tensor(x):return torch.tensor(x,dtype=torch.float64,device='cuda')
    def solve(x,y):
        a,b=tensor(x),tensor(y);n=len(x)
        return torch.linalg.solve((a.T@a)/n+.01*torch.eye(a.shape[1],dtype=torch.float64,device='cuda'),a.T@b/n).cpu().numpy()
    beta=solve(state[fit],expected['target'][fit]);residual=expected['target'][fit]-state[fit]@beta
    cm=solve(response[fit],residual)
    rng=torch.Generator(device='cuda').manual_seed(16671);permutation=torch.randperm(int(fit.sum()),generator=rng,device='cuda').cpu().numpy()
    shuffled_z=z.copy();shuffled_z[fit]=shuffled_z[fit][permutation]
    shuffled_response=(phi[:,:,None]*shuffled_z[:,None,:]).reshape(len(fit),-1)
    shuffled=solve(shuffled_response[fit],residual)
    for key,value in [('state',beta),('cm',cm),('shuffled',shuffled)]:compare('coefficients_'+key,value,bundle['coefficients'][key].numpy(),1e-7)
    baseline=state@beta
    predicted=dict(state_only=baseline,cm=baseline+response@cm,shuffled=baseline+response@shuffled)
    for key,value in predicted.items():compare('prediction_'+key,value,bundle['predictions'][key].numpy(),1e-7)
    all_design=(phi[:,None,:,None]*(expected['centered_candidates']/norm['action_rms'])[:,:,None,:]).reshape(len(fit),8,-1)
    candidates=baseline[:,None]+all_design@cm
    compare('all_candidate_predictions',candidates,bundle['candidate_predictions'].numpy(),1e-7)
    compare('conditional_zero_mean',sum(expected['probability'][:,a,None]*candidates[:,a] for a in range(8)),baseline,1e-10)
    scores={}
    for subset,mask in [('all_cal',cal),('dynamic_cal',dynamic)]:
        scores[subset]={}
        for mode,values in predicted.items():
            e=(values[mask]-expected['target'][mask])*scale
            local=dict(velocity_vector_rmse_mps=float(np.sqrt(np.mean((e[:,:3]**2).sum(-1)))),
                displacement_vector_rmse_mm=float(np.sqrt(np.mean((e[:,3:6]**2).sum(-1)))*1000),
                clearance_rmse_mm=float(np.sqrt(np.mean(e[:,6]**2)))*1000,
                cv_velocity_vector_rmse_mps=float(np.sqrt(np.mean((expected['cv_velocity_error'][mask]**2).sum(-1)))))
            scores[subset][mode]=local
            for key,value in local.items():compare(subset+'_'+mode+'_'+key,value,result['metrics'][subset][mode][key],1e-7)
    support=result['support'];cal_episode=len(set(np.array(episodes)[cal]));cal_group=len(set(np.array(groups)[cal]))
    if (support['fit_rows']!=int(fit.sum()) or support['cal_rows']!=int(cal.sum()) or support['cal_episodes']!=cal_episode
            or support['cal_groups']!=cal_group or support['dynamic_cal_rows']!=int(dynamic.sum()) or support['excluded_held_rows']!=excluded):
        raise ValueError('support/held exclusion drift')
    gates=dict(support=int(cal.sum())>=256 and cal_episode>=24 and cal_group>=8 and int(dynamic.sum())>=64)
    for subset in ('all_cal','dynamic_cal'):
        local=scores[subset];gates[subset+'_velocity_action_information']=all(local['cm']['velocity_vector_rmse_mps']<=.9*local[c]['velocity_vector_rmse_mps'] for c in ('state_only','shuffled'))
    local=scores['all_cal']
    for key in ('displacement_vector_rmse_mm','clearance_rmse_mm'):
        gates[key+'_action_information']=all(local['cm'][key]<=.95*local[c][key] for c in ('state_only','shuffled'))
    gates['velocity_cv_nonregression']=local['cm']['velocity_vector_rmse_mps']<=local['cm']['cv_velocity_vector_rmse_mps']
    label='PROMISING' if all(gates.values()) else ('UNPROMISING' if gates['support'] else 'UNCLEAR')
    if gates!=result['gates'] or label!=result['label']:raise ValueError('original gates/classification drift')
    output=dict(run_status='COMPLETED',passed=True,label=label,rows=len(fit),excluded_held_rows=excluded,errors=errors,
        raw_pre_and_first_post_labels_verified=True,fit_only_normalization=True,independent_gpu_solutions_verified=True,
        full_candidate_and_mathematical_centering_verified=True,original_gates_verified=True,
        elapsed_seconds=time.monotonic()-begin,bundle_sha256=sha(args.run/'response.pt'),
        scope='physical response qualification only; unexecuted predictions not treated as truth')
    if args.output.exists():raise ValueError('unique qualification audit')
    args.output.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=5);run(p.parse_args())
