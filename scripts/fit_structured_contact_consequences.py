#!/usr/bin/env python3
"""Fixed matched GPU fit of coherent H10 consequences on audited actual data."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission
from structured_contact_data import load,BASE
CARD=ROOT/'docs/experiments/probes/P-20261002-structured-contact-effects.md'
ENGINEERING=ROOT/'docs/experiments/probes/P-20261002-structured-contact-engineering-r2.json'


def metrics(pred,target,take):
    import torch
    from src.task.CmResidual.structured_contact_consequence import event_marginals
    probs=pred['event_probability'][take];support=pred['support_probability'][take]
    y=target['support'][take].float();height=pred['supported_height'][take]
    physical=pred['physical'][take]-target['physical'][take]
    return dict(rows=int(take.sum()),joint_support_brier=float((support-y).square().mean()),
        height_mae_mm=float((height-target['supported_height'][take]).abs().mean()*10),
        height_bias_mm=float((height-target['supported_height'][take]).mean()*10),
        lift_brier=float((pred['lift_probability'][take]-target['lift'][take].float()).square().mean()),
        loss_brier=float((pred['loss_probability'][take]-target['loss'][take].float()).square().mean()),
        object_dv_rmse_mps=float(physical[:,36:39].square().mean().sqrt()*.1),
        relative_position_rmse_mm=float(physical[:,:36].reshape(-1,6,6)[...,:3].square().mean().sqrt()*5),
        subset_max_excess=float((pred['lift_probability'][take,None]-event_marginals(probs)).max()),
        support_subset_max_excess=float((support[:,None]-event_marginals(probs)).max()))


def run(args):
    start=time.monotonic();prior=65.24201305889148
    if args.output.parent.resolve()!=BASE.resolve() or args.output.exists() or args.output.is_symlink():
        raise ValueError('unique owned output required')
    eng=json.loads(ENGINEERING.read_text())
    if not eng['engineering_passed'] or eng['run_status']!='COMPLETED':raise ValueError('engineering required')
    if any(sha(Path(k))!=v for k,v in eng['input_sha256'].items()):raise ValueError('engineering input drift')
    admission=gpu_admission(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from src.task.CmResidual.structured_contact_consequence import record_features,factual_labels,normalization,normalize,FEATURES,MODES,StructuredContactConsequence,objective
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    records,origins,hashes=load()
    for p in (Path(__file__),CARD,ENGINEERING,ROOT/'src/task/CmResidual/structured_contact_consequence.py'):
        hashes[str(p.resolve())]=sha(p)
    args.output.mkdir()
    manifest=dict(experiment_id='P-20261002-structured-contact-effects',family='HF21',probe_index_in_family=1,
        run_status='RUNNING',pid=os.getpid(),command=sys.argv,gpu=admission,models=[],input_sha256=hashes,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        prior_engineering_preparation_seconds=prior,wall_limit_seconds=1800,output_limit_bytes=512<<20,
        no_new_rollouts=True,scope='reused-source model information Probe; no independent utility Validation')
    def save():(args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if prior+time.monotonic()-start>1740:raise TimeoutError('whole Probe wall bound')
        if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())>512<<20:raise ValueError('Probe output bound')
    save()
    try:
        parts=[record_features(b) for b in records];target_parts=[factual_labels(b) for b in records]
        raw={k:torch.cat([p[k] for p in parts]).cuda() for k in FEATURES}
        target={k:torch.cat([p[k] for p in target_parts]).cuda() for k in target_parts[0]}
        bucket=torch.cat([b['split_group_bucket'] for b in records]).cuda()
        origin=torch.cat([torch.full_like(b['assignment'],o) for o,b in zip(origins,records)]).cuda()
        arm=torch.cat([b['assignment'] for b in records]).cuda()
        early=torch.cat([~b['outcome']['initially_clear'] for b in records]).cuda()
        split=dict(fit=bucket<50,cal=(bucket>=50)&(bucket<70),reused_held=bucket>=70)
        episodes=[e for b in records for e in b['episode_id']]
        groups=[f'{int(m)}/{int(s)}' for b in records for m,s in zip(b['motion_id'],b['start_frame'])]
        n=len(bucket)
        if n!=5018:raise ValueError('fixed union rows required')
        counts={}
        for name,take in split.items():
            mask=take.cpu().tolist()
            counts[name]=dict(rows=int(take.sum()),episodes=len({e for e,t in zip(episodes,mask) if t}),
                initial_groups=len({g for g,t in zip(groups,mask) if t}),support=int((take&target['support']).sum()),
                unsupported=int((take&~target['support']).sum()),
                generated_distribution_rows=int((take&(origin==20)&(arm>=2)&(arm<=4)).sum()))
        adequate=all(v['rows']>=150 and v['episodes']>=32 and v['initial_groups']>=8 for v in counts.values())
        adequate &= counts['reused_held']['support']>=32 and counts['reused_held']['unsupported']>=32 and counts['reused_held']['generated_distribution_rows']>=60
        norm=normalization(raw,split['fit']);f=normalize(raw,norm);fit=torch.where(split['fit'])[0]
        proposal_features=[]
        for option in range(8):
            option_parts=[record_features(b,option) for b in records]
            c=dict(f)
            for key in ('node_action','law'):
                value=torch.cat([p[key] for p in option_parts]).cuda()
                c[key]=((value-norm[key+'_mean'])/norm[key+'_std']).clamp(-8,8)
            proposal_features.append(c)
        manifest.update(counts=counts,normalization_fit_only=True);save()
        states={};predictions={};candidate_predictions={};reports={};parameter_counts={}
        prediction_fields=('event_probability','support_probability','lift_probability','loss_probability','supported_height','conditional_height','physical')
        for mode in MODES:
            states[mode]=[];members=[];candidates=[]
            for seed in (12811,12812,12813):
                check();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                model=StructuredContactConsequence(raw['physical'].shape[-1],mode).cuda()
                parameter_counts[mode]=sum(p.numel() for p in model.parameters())
                optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=1e-4)
                sampler=torch.Generator(device='cuda').manual_seed(seed+30000)
                action=f['node_action'].clone();law=f['law'].clone()
                if mode=='shuffled':
                    shuffler=torch.Generator(device='cuda').manual_seed(seed+40000)
                    for source_id in (19,20):
                        for is_early in (False,True):
                            rows=fit[(origin[fit]==source_id)&(early[fit]==is_early)]
                            permutation=rows[torch.randperm(len(rows),device='cuda',generator=shuffler)]
                            action[rows]=f['node_action'][permutation];law[rows]=f['law'][permutation]
                model.train()
                for update in range(1000):
                    if update%100==0:check()
                    ids=fit[torch.randint(len(fit),(256,),device='cuda',generator=sampler)]
                    inputs={k:f[k][ids] for k in FEATURES};inputs.update(node_action=action[ids],law=law[ids])
                    optimizer.zero_grad(set_to_none=True);p=model(**inputs)
                    value=objective(p,{k:v[ids] for k,v in target.items()},mode)
                    if not torch.isfinite(value):raise ValueError('nonfinite fit loss')
                    value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                model.eval().requires_grad_(False)
                with torch.no_grad():
                    chunks=[model(**{k:f[k][s:s+256] for k in FEATURES}) for s in range(0,n,256)]
                    actual={k:torch.cat([c[k] for c in chunks]).cpu() for k in prediction_fields}
                    if any(not torch.isfinite(v).all() for v in actual.values()):raise ValueError('nonfinite actual forecast')
                    members.append(actual)
                    programme=[]
                    for option in range(8):
                        c=proposal_features[option];values=[]
                        for s in range(0,n,256):
                            out=model(**{k:c[k][s:s+256] for k in FEATURES})
                            values.append(torch.cat((out['event_probability'],out['supported_height'][:,None],
                                out['loss_probability'][:,None],out['lift_probability'][:,None],out['conditional_height'][:,None]),-1).cpu())
                        programme.append(torch.cat(values))
                    candidates.append(torch.stack(programme,1))
                states[mode].append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
                manifest['models'].append(dict(mode=mode,seed=seed,updates=1000,final_loss=float(value)));save()
                print(json.dumps(dict(mode=mode,seed=seed,updates=1000,run_status='COMPLETED')),flush=True)
                del model,optimizer
            predictions[mode]={k:torch.stack([m[k] for m in members]) for k in prediction_fields}
            mean={k:v.mean(0).cuda() for k,v in predictions[mode].items()}
            candidate_predictions[mode]=torch.stack(candidates)
            reports[mode]={name:metrics(mean,target,take) for name,take in split.items() if name!='fit'}
            generated=split['reused_held']&(origin==20)&(arm>=2)&(arm<=4)
            reports[mode]['generated_distribution']=metrics(mean,target,generated)
        if len(set(parameter_counts.values()))!=1:raise ValueError('matched parameter capacity required')
        cm=reports['cm']['reused_held'];gates=dict(supervision=bool(adequate))
        for control in ('state_only','shuffled'):
            other=reports[control]['reused_held']
            for metric in ('height_mae_mm','joint_support_brier'):
                gates[control+'_'+metric+'_5pct']=cm[metric]<=.95*other[metric]
            for metric in ('lift_brier','loss_brier','object_dv_rmse_mps','relative_position_rmse_mm'):
                gates[control+'_'+metric+'_nonworse10pct']=cm[metric]<=1.10*other[metric]
            gates[control+'_generated_height_nonworse2pct']=reports['cm']['generated_distribution']['height_mae_mm']<=1.02*reports[control]['generated_distribution']['height_mae_mm']
        gates['event_containment']=cm['subset_max_excess']<=1e-6 and cm['support_subset_max_excess']<=1e-6
        gates['passed']=all(gates.values())
        label='UNCLEAR' if not adequate else ('PROMISING' if gates['passed'] else 'UNPROMISING')
        checkpoint=dict(schema='ref2dex.structured_contact_consequence.v1',physical_dim=raw['physical'].shape[-1],
            models=states,normalization={k:v.cpu() for k,v in norm.items()},features=FEATURES,input_sha256=hashes,
            scope=manifest['scope'],event_semantics='hand_force_proxy/object_force_proxy/end3_mesh_clearance')
        torch.save(checkpoint,args.output/'structured_contact_consequence.pt')
        torch.save(dict(schema='ref2dex.structured_contact_forecasts.v1',actual=predictions,candidates=candidate_predictions,
            candidate_columns=['event0','event1','event2','event3','event4','event5','event6','event7',
                'expected_supported_height_10mm','geometric_loss_probability','supported_lift_probability','conditional_height_10mm'],
            target={k:v.cpu() for k,v in target.items()},bucket=bucket.cpu(),origin=origin.cpu(),assignment=arm.cpu(),
            normalization_fit_only=True),args.output/'forecasts.pt')
        check()
        if any(sha(Path(k))!=v for k,v in hashes.items()):raise ValueError('training source/input drift')
        result=dict(run_status='COMPLETED',label=label,counts=counts,metrics=reports,gate=gates,
            parameter_counts=parameter_counts,elapsed_seconds=time.monotonic()-start,
            cumulative_seconds=prior+time.monotonic()-start,shared_old_source_cost_separate=True,
            checkpoint_sha256=sha(args.output/'structured_contact_consequence.pt'),forecast_sha256=sha(args.output/'forecasts.pt'),
            scope=manifest['scope'])
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        manifest.update(run_status='COMPLETED',input_hashes_unchanged=True,result_sha256=sha(args.output/'results.json'),
            elapsed_seconds=time.monotonic()-start,cumulative_seconds=result['cumulative_seconds']);save();check()
        print(json.dumps(dict(run_status='COMPLETED',label=label,gate=gates,metrics=reports['cm'],
            cumulative_seconds=result['cumulative_seconds'])),flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',error=repr(error),elapsed_seconds=time.monotonic()-start);save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu',type=int,default=1)
    run(parser.parse_args())
