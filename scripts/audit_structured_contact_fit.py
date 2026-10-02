#!/usr/bin/env python3
"""Independent factual events/PD/physical labels/fit norms and full GPU replay."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission
from structured_contact_data import load
from check_structured_contact_engineering import independent_labels,independent_pd


def rotation(q):
    x,y,z,w=q.T
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(-1,3,3)


def independent_physical(b):
    state=b['state'].numpy().astype(np.float64);post=b['future_state'][:,0].numpy().astype(np.float64)
    def geometry(s,positions,velocities):
        r=rotation(s[:,39:43])
        p=np.einsum('nji,nkj->nki',r,positions-s[:,None,36:39])
        v=np.einsum('nji,nkj->nki',r,velocities-s[:,None,43:46])
        return p,v
    prepos,prevel=geometry(state,b['initial_key_positions'].numpy(),b['initial_key_velocities'].numpy())
    pos,vel=geometry(post,b['future_key_positions'][:,0].numpy(),b['future_key_velocities'][:,0].numpy())
    local=np.concatenate(((pos-prepos)/.005,(vel-prevel)/.1),-1).reshape(len(state),36)
    clr=(b['future_clearance'][:,-3:].numpy().min(-1)-b['initial_clearance'].numpy())[:,None]/.002
    return np.concatenate((local,(post[:,43:46]-state[:,43:46])/.1,(post[:,36:39]-state[:,36:39])/.005,clr),-1)


def independent_actions(b,option):
    q=b['state'][:,:18].numpy();off=b['pd_offset'].numpy();scale=b['pd_scale'].numpy()
    n=len(q);ids=np.arange(n);chosen=b['assignment'].numpy() if option is None else np.full(n,option)
    cup=independent_pd(b['candidate_actions'][:,1].numpy(),q,off,scale)
    pd=independent_pd(b['candidate_actions'].numpy()[ids,chosen],q,off,scale)
    node=np.zeros((n,6,6),dtype=np.float32)
    for k,channels in enumerate(((0,1,2,3,4,5),(6,7),(8,9),(10,11),(12,13),(14,15,16,17))):
        node[:,k,:len(channels)]=(pd-cup)[:,channels]
    ref=np.zeros((n,6,6),dtype=np.float32);ref[:,:,1]=1
    return np.concatenate((node,b['candidate_weights'].numpy()[ids,chosen]-ref),-1),(chosen!=0).astype(np.float32)[:,None]


def independent_metrics(pred,target,take):
    p=pred['event_probability'][take];s=pred['support_probability'][take]
    bits=np.array([[int((i>>j)&1) for j in (2,1,0)] for i in range(8)])
    marginals=p@bits
    e=pred['physical'][take]-target['physical'][take]
    return dict(rows=int(take.sum()),joint_support_brier=float(np.mean((s-target['support'][take])**2)),
        height_mae_mm=float(np.mean(np.abs(pred['supported_height'][take]-target['supported_height'][take]))*10),
        height_bias_mm=float(np.mean(pred['supported_height'][take]-target['supported_height'][take])*10),
        lift_brier=float(np.mean((pred['lift_probability'][take]-target['lift'][take])**2)),
        loss_brier=float(np.mean((pred['loss_probability'][take]-target['loss'][take])**2)),
        object_dv_rmse_mps=float(np.sqrt(np.mean(e[:,36:39]**2))*.1),
        relative_position_rmse_mm=float(np.sqrt(np.mean(e[:,:36].reshape(-1,6,6)[...,:3]**2))*5),
        subset_max_excess=float(np.max(pred['lift_probability'][take,None]-marginals)),
        support_subset_max_excess=float(np.max(s[:,None]-marginals)))


def run(args):
    start=time.monotonic()
    if args.output.exists():raise ValueError('unique audit artifact required')
    manifest_path=args.run/'run_manifest.json';m=json.loads(manifest_path.read_text())
    result=json.loads((args.run/'results.json').read_text())
    if m['run_status']!='COMPLETED' or sha(args.run/'results.json')!=m['result_sha256']:
        raise ValueError('terminal fit required')
    if len(m['models'])!=12 or any(p['updates']!=1000 for p in m['models']):raise ValueError('matched fixed model panel')
    if any(sha(Path(k))!=v for k,v in m['input_sha256'].items()):raise ValueError('fit input drift')
    if sha(args.run/'structured_contact_consequence.pt')!=result['checkpoint_sha256'] or sha(args.run/'forecasts.pt')!=result['forecast_sha256']:
        raise ValueError('fit artifacts drift')
    admission=gpu_admission(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    import torch
    from src.task.CmResidual.structured_contact_consequence import record_features,normalize,FEATURES,StructuredContactConsequence
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    records,origins,hashes=load()
    p=torch.load(args.run/'structured_contact_consequence.pt',map_location='cpu',weights_only=False)
    saved=torch.load(args.run/'forecasts.pt',map_location='cpu',weights_only=False)
    parts=[record_features(b) for b in records];raw={k:torch.cat([b[k] for b in parts]) for k in FEATURES}
    expected={k:np.concatenate([independent_labels(b)[k] for b in records]) for k in ('event','height','support','supported_height','loss','lift')}
    expected['physical']=np.concatenate([independent_physical(b) for b in records])
    errors={}
    for key,value in expected.items():
        errors['label_'+key]=float(np.max(np.abs(value.astype(float)-saved['target'][key].numpy().astype(float))))
        if errors['label_'+key]>(.002 if key=='physical' else 2e-4):raise ValueError('independent labels '+key)
    # Confirm hand/node geometry against world poses, in addition to old full-source audit.
    for b,part in zip(records,parts):
        state=b['state'].numpy().astype(np.float64);r=rotation(state[:,39:43])
        rel=np.einsum('nji,nkj->nki',r,b['initial_key_positions'].numpy()-state[:,None,36:39])
        err=float(np.max(np.abs(rel-part['node_context'].numpy()[...,:3])))
        errors['current_relative_geometry']=max(errors.get('current_relative_geometry',0.),err)
        if err>2e-5:raise ValueError('current geometry contract')
        force=np.concatenate((b['initial_hand_force'].numpy().reshape(len(state),15),b['initial_object_force'].numpy()),-1)/(b['mass_kg'].numpy()[:,None]*b['gravity_magnitude'])
        phys=np.concatenate((np.sign(force)*np.log1p(np.abs(force)),b['initial_clearance'].numpy()[:,None],
            np.maximum(0,b['state'][:,38].numpy()-b['rest_z'].numpy())[:,None]),-1)
        err=float(np.max(np.abs(phys-part['physical'].numpy())))
        errors['current_force_geometry']=max(errors.get('current_force_geometry',0.),err)
        if err>2e-5:raise ValueError('current physical features')
        action,law=independent_actions(b,None)
        err=float(np.max(np.abs(action-part['node_action'].numpy())))
        errors['current_factual_action']=max(errors.get('current_factual_action',0.),err)
        if err>2e-5 or not np.array_equal(law,part['law'].numpy()):raise ValueError('factual pre-only action')
    buckets=np.concatenate([b['split_group_bucket'].numpy() for b in records])
    origin=np.concatenate([np.full(len(b['state']),o) for o,b in zip(origins,records)])
    arm=np.concatenate([b['assignment'].numpy() for b in records]);fit=buckets<50
    if not np.array_equal(buckets,saved['bucket'].numpy()) or not np.array_equal(origin,saved['origin'].numpy()) or not np.array_equal(arm,saved['assignment'].numpy()):
        raise ValueError('source row order')
    episodes=[e for b in records for e in b['episode_id']]
    groups=[(int(i),int(j)) for b in records for i,j in zip(b['motion_id'],b['start_frame'])]
    counts={}
    for name,take in [('fit',fit),('cal',(buckets>=50)&(buckets<70)),('reused_held',buckets>=70)]:
        counts[name]=dict(rows=int(take.sum()),episodes=len({e for e,t in zip(episodes,take) if t}),
            initial_groups=len({g for g,t in zip(groups,take) if t}),support=int((take&expected['support']).sum()),
            unsupported=int((take&~expected['support']).sum()),
            generated_distribution_rows=int((take&(origin==20)&(arm>=2)&(arm<=4)).sum()))
    if counts!=result['counts']:raise ValueError('independent source support counts')
    adequate=all(v['rows']>=150 and v['episodes']>=32 and v['initial_groups']>=8 for v in counts.values())
    adequate &= counts['reused_held']['support']>=32 and counts['reused_held']['unsupported']>=32 and counts['reused_held']['generated_distribution_rows']>=60
    for key in FEATURES:
        x=raw[key].numpy()[fit].astype(np.float64);axes=(0,1) if key in ('history','node_context','node_action') else 0
        mean=x.mean(axis=axes);std=np.sqrt(((x-mean)**2).mean(axis=axes)).clip(.001)
        for stat,actual in [('mean',mean),('std',std)]:
            stored=p['normalization'][key+'_'+stat].numpy()
            errors['norm_'+key+'_'+stat]=float(np.max(np.abs(actual-stored)))
            if not np.allclose(actual,stored,atol=2e-5,rtol=2e-5):raise ValueError('fit-only normalization '+key+stat)
    norm={k:v.cuda() for k,v in p['normalization'].items()};f=normalize({k:v.cuda() for k,v in raw.items()},norm)
    proposal=[]
    for option in range(8):
        action=np.concatenate([independent_actions(b,option)[0] for b in records]);law=(arm*0+(option!=0)).astype(np.float32)[:,None]
        c=dict(f)
        for key,value in [('node_action',action),('law',law)]:
            x=torch.from_numpy(value).cuda();c[key]=((x-norm[key+'_mean'])/norm[key+'_std']).clamp(-8,8)
        proposal.append(c)
    max_forecast=0.
    with torch.no_grad():
        for mode,states in p['models'].items():
            for member,weights in enumerate(states):
                model=StructuredContactConsequence(p['physical_dim'],mode).cuda().eval().requires_grad_(False)
                model.load_state_dict(weights,strict=True)
                actual_chunks=[model(**{k:f[k][s:s+256] for k in FEATURES}) for s in range(0,len(arm),256)]
                for key,values in saved['actual'][mode].items():
                    actual=torch.cat([v[key] for v in actual_chunks]).cpu();reference=values[member]
                    err=float((actual-reference).abs().max());max_forecast=max(max_forecast,err)
                    if not torch.allclose(actual,reference,atol=2e-5,rtol=2e-6):raise ValueError('full factual NN replay '+key)
                for option in range(8):
                    c=proposal[option];chunks=[]
                    for s in range(0,len(arm),256):
                        out=model(**{k:c[k][s:s+256] for k in FEATURES})
                        chunks.append(torch.cat((out['event_probability'],out['supported_height'][:,None],out['loss_probability'][:,None],out['lift_probability'][:,None],out['conditional_height'][:,None]),-1).cpu())
                    actual=torch.cat(chunks);reference=saved['candidates'][mode][member,:,option]
                    err=float((actual-reference).abs().max());max_forecast=max(max_forecast,err)
                    if not torch.allclose(actual,reference,atol=2e-5,rtol=2e-6):raise ValueError('independent candidate PD full NN replay')
                if time.monotonic()-start>600:raise TimeoutError('audit budget')
                del model
    max_metric=0.;target={k:v.numpy().astype(np.float64) for k,v in saved['target'].items()}
    splits=dict(cal=(buckets>=50)&(buckets<70),reused_held=buckets>=70,
        generated_distribution=(buckets>=70)&(origin==20)&(arm>=2)&(arm<=4))
    recomputed={}
    for mode,values in saved['actual'].items():
        prediction={k:v.numpy().astype(np.float64).mean(0) for k,v in values.items()}
        recomputed[mode]={}
        for name,take in splits.items():
            actual=independent_metrics(prediction,target,take);reference=result['metrics'][mode][name]
            recomputed[mode][name]=actual
            for key,value in actual.items():
                err=abs(value-reference[key]);max_metric=max(max_metric,err)
                if err>2e-5:raise ValueError('independent metric '+mode+'/'+name+'/'+key)
    gates={'supervision':bool(adequate)};cm=recomputed['cm']['reused_held']
    for control in ('state_only','shuffled'):
        other=recomputed[control]['reused_held']
        for metric in ('height_mae_mm','joint_support_brier'):gates[control+'_'+metric+'_5pct']=cm[metric]<=.95*other[metric]
        for metric in ('lift_brier','loss_brier','object_dv_rmse_mps','relative_position_rmse_mm'):gates[control+'_'+metric+'_nonworse10pct']=cm[metric]<=1.10*other[metric]
        gates[control+'_generated_height_nonworse2pct']=recomputed['cm']['generated_distribution']['height_mae_mm']<=1.02*recomputed[control]['generated_distribution']['height_mae_mm']
    gates['event_containment']=cm['subset_max_excess']<=1e-6 and cm['support_subset_max_excess']<=1e-6;gates['passed']=all(gates.values())
    label='UNCLEAR' if not gates['supervision'] else ('PROMISING' if gates['passed'] else 'UNPROMISING')
    if gates!=result['gate'] or label!=result['label']:raise ValueError('fixed gate/label mismatch')
    if any(sha(Path(k))!=v for k,v in m['input_sha256'].items()):raise ValueError('end input drift')
    torch.cuda.synchronize()
    out=dict(run_status='COMPLETED',audit_passed=True,rows=len(arm),independent_errors=errors,
        full_nn_forecast_max_error=max_forecast,independent_metric_max_error=max_metric,label=label,
        elapsed_seconds=time.monotonic()-start,cumulative_seconds=result['cumulative_seconds']+time.monotonic()-start,
        run_manifest_sha256=sha(manifest_path),result_sha256=sha(args.run/'results.json'),
        checkpoint_sha256=sha(args.run/'structured_contact_consequence.pt'),auditor_sha256=sha(Path(__file__)),
        gpu=admission,scope='full factual label/normalization/known-candidate NN/gate audit; no new utility claim')
    args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='gpu'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu',type=int,default=1)
    run(parser.parse_args())
