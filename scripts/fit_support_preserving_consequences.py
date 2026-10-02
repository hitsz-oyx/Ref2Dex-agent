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
from support_preserving_contact_data import load,BASE
from support_preserving_information import fixed_gates
CARD=ROOT/'docs/experiments/probes/P-20261002-support-preserving-contact-effects.md'
ENGINEERING=BASE/'P-20261002-support-preserving-engineering-r1.json'
OLD_CHECKPOINT=BASE/'P-20261002-structured-contact-fit-r1/structured_contact_consequence.pt'


def metrics(pred,target,take):
    from fit_structured_contact_consequences import metrics as original_metrics
    result=original_metrics(pred,target,take)
    joint=pred['event_probability'][take,6:8].sum(-1)
    result['joint_presence_brier']=float((joint-target['joint'][take].float()).square().mean())
    if 'contact_loss_probability' in pred:
        contact=pred['contact_loss_probability'][take]
        result['contact_loss_brier']=float((contact-target['contact_loss'][take].float()).square().mean())
        result['contact_loss_necessary_max_excess']=float(((1-joint)-contact).max())
    return result


def run(args):
    start=time.monotonic();prior=60.+json.loads(ENGINEERING.read_text())['elapsed_seconds']
    previous_path=None
    if args.premature_attempt is not None:
        previous_path=args.premature_attempt.resolve()/'run_manifest.json'
        previous=json.loads(previous_path.read_text())
        if previous['run_status']!='STOPPED' or previous['child_exit_code']!=143 or not previous['invalid_for_scientific_classification']:
            raise ValueError('verified terminal premature attempt required')
        if (args.premature_attempt/'results.json').exists():raise ValueError('cannot retry a classified scientific result')
        prior+=previous['elapsed_upper_bound_seconds']
    if args.output.parent.resolve()!=BASE.resolve() or args.output.exists() or args.output.is_symlink():
        raise ValueError('unique owned output required')
    eng=json.loads(ENGINEERING.read_text())
    if not eng['engineering_passed'] or eng['run_status']!='COMPLETED':raise ValueError('engineering required')
    if any(sha(Path(k))!=v for k,v in eng['input_sha256'].items()):raise ValueError('engineering input drift')
    admission=gpu_admission(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from src.task.CmResidual.support_preserving_consequence import record_features,factual_labels,normalize,FEATURES,MODES,SupportPreservingContactConsequence,objective,warm_start
    from src.task.CmResidual.structured_contact_consequence import StructuredContactConsequence
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    records,origins,hashes=load()
    for p in (Path(__file__),CARD,ENGINEERING,ROOT/'src/task/CmResidual/support_preserving_consequence.py',OLD_CHECKPOINT,ROOT/'scripts/support_preserving_information.py',
        ROOT/'scripts/fit_structured_contact_consequences.py',ROOT/'scripts/audit_support_preserving_fit.py'):
        hashes[str(p.resolve())]=sha(p)
    if previous_path is not None:hashes[str(previous_path)]=sha(previous_path)
    args.output.mkdir()
    manifest=dict(experiment_id='P-20261002-support-preserving-contact-effects',family='HF22',probe_index_in_family=1,
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
        split=dict(new_fit=(origin==21)&(bucket<50),new_cal=(origin==21)&(bucket>=50)&(bucket<70),
            new_reused_held=(origin==21)&(bucket>=70),old_reused_held=(origin!=21)&(bucket>=70))
        episodes=[e for b in records for e in b['episode_id']]
        groups=[f'{int(m)}/{int(s)}' for b in records for m,s in zip(b['motion_id'],b['start_frame'])]
        n=len(bucket)
        if n!=6263:raise ValueError('fixed union rows required')
        counts={}
        for name,take in split.items():
            mask=take.cpu().tolist()
            counts[name]=dict(rows=int(take.sum()),episodes=len({e for e,t in zip(episodes,mask) if t}),
                initial_groups=len({g for g,t in zip(groups,mask) if t}),support=int((take&target['support']).sum()),
                unsupported=int((take&~target['support']).sum()),
                generated_distribution_rows=int((take&(arm>=2)&(arm<=4)).sum()),
                contact_loss=int((take&target['contact_loss']).sum()),contact_retained=int((take&~target['contact_loss']).sum()))
        adequate=all(counts[k]['rows']>=150 and counts[k]['episodes']>=32 and counts[k]['initial_groups']>=8
                     for k in ('new_fit','new_cal','new_reused_held'))
        adequate &= counts['new_reused_held']['contact_loss']>=32 and counts['new_reused_held']['contact_retained']>=32
        adequate &= counts['new_reused_held']['generated_distribution_rows']>=60
        bundle=torch.load(OLD_CHECKPOINT,map_location='cpu',weights_only=False)
        if bundle['schema']!='ref2dex.structured_contact_consequence.v1':raise ValueError('exact old bundle required')
        norm={k:v.cuda() for k,v in bundle['normalization'].items()};f=normalize(raw,norm)
        fit=torch.where(bucket<50)[0];old_fit=fit[origin[fit]!=21];new_fit=fit[origin[fit]==21]
        if len(old_fit)!=2532 or len(new_fit)!=624:raise ValueError('fixed fit replay/new support')
        frozen_predictions=[]
        with torch.no_grad():
            for weights in bundle['models']['cm']:
                old_model=StructuredContactConsequence(raw['physical'].shape[-1],'cm').cuda().eval().requires_grad_(False)
                old_model.load_state_dict(weights,strict=True)
                values=[old_model(**{k:f[k][s:s+256] for k in FEATURES}) for s in range(0,n,256)]
                frozen_predictions.append({k:torch.cat([v[k] for v in values]).cpu() for k in
                    ('event_probability','support_probability','lift_probability','loss_probability','supported_height','conditional_height','physical')})
                del old_model
        frozen={k:torch.stack([v[k] for v in frozen_predictions]) for k in frozen_predictions[0]}
        mean={k:v.mean(0).cuda() for k,v in frozen.items()}
        old_reports={name:metrics(mean,target,take) for name,take in split.items() if name!='new_fit'}
        generated=split['new_reused_held']&(arm>=2)&(arm<=4)
        old_reports['new_generated_distribution']=metrics(mean,target,generated)
        proposal_features=[]
        for option in range(8):
            option_parts=[record_features(b,option) for b in records]
            c=dict(f)
            for key in ('node_action','law'):
                value=torch.cat([p[key] for p in option_parts]).cuda()
                c[key]=((value-norm[key+'_mean'])/norm[key+'_std']).clamp(-8,8)
            proposal_features.append(c)
        manifest.update(counts=counts,normalization_inherited_old_fit_only=True);save()
        states={};predictions={};candidate_predictions={};reports={};parameter_counts={}
        prediction_fields=('event_probability','support_probability','lift_probability','loss_probability','supported_height','conditional_height','physical','joint_probability','contact_loss_probability')
        for mode in MODES:
            states[mode]=[];members=[];candidates=[]
            for member,seed in enumerate((13811,13812,13813)):
                check();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                model=SupportPreservingContactConsequence(raw['physical'].shape[-1],mode).cuda()
                warm_start(model,bundle['models'][mode][member])
                parameter_counts[mode]=sum(p.numel() for p in model.parameters())
                optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-4)
                sampler=torch.Generator(device='cuda').manual_seed(seed+30000)
                action=f['node_action'].clone();law=f['law'].clone()
                if mode=='shuffled':
                    shuffler=torch.Generator(device='cuda').manual_seed(seed+40000)
                    for source_id in (19,20,21):
                        for is_early in (False,True):
                            rows=fit[(origin[fit]==source_id)&(early[fit]==is_early)]
                            permutation=rows[torch.randperm(len(rows),device='cuda',generator=shuffler)]
                            action[rows]=f['node_action'][permutation];law[rows]=f['law'][permutation]
                model.train()
                for update in range(500):
                    if update%100==0:check()
                    ids=torch.cat((old_fit[torch.randint(len(old_fit),(128,),device='cuda',generator=sampler)],
                        new_fit[torch.randint(len(new_fit),(128,),device='cuda',generator=sampler)]))
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
                                out['loss_probability'][:,None],out['lift_probability'][:,None],out['conditional_height'][:,None],out['joint_probability'][:,None],
                                out['contact_loss_probability'][:,None]),-1).cpu())
                        programme.append(torch.cat(values))
                    candidates.append(torch.stack(programme,1))
                states[mode].append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
                manifest['models'].append(dict(mode=mode,seed=seed,updates=500,final_loss=float(value)));save()
                print(json.dumps(dict(mode=mode,seed=seed,updates=500,run_status='COMPLETED')),flush=True)
                del model,optimizer
            predictions[mode]={k:torch.stack([m[k] for m in members]) for k in prediction_fields}
            mean={k:v.mean(0).cuda() for k,v in predictions[mode].items()}
            candidate_predictions[mode]=torch.stack(candidates)
            reports[mode]={name:metrics(mean,target,take) for name,take in split.items() if name!='new_fit'}
            generated=split['new_reused_held']&(arm>=2)&(arm<=4)
            reports[mode]['new_generated_distribution']=metrics(mean,target,generated)
        if len(set(parameter_counts.values()))!=1:raise ValueError('matched parameter capacity required')
        gates=fixed_gates(reports,old_reports,adequate)
        label='UNCLEAR' if not adequate else ('PROMISING' if gates['passed'] else 'UNPROMISING')
        checkpoint=dict(schema='ref2dex.support_preserving_contact_consequence.v1',physical_dim=raw['physical'].shape[-1],
            models=states,normalization={k:v.cpu() for k,v in norm.items()},features=FEATURES,input_sha256=hashes,
            scope=manifest['scope'],event_semantics='hand_force_proxy/object_force_proxy/end3_mesh_clearance',
            contact_loss_semantics='any H10 hand/object presence proxy lost',old_checkpoint_sha256=sha(OLD_CHECKPOINT))
        torch.save(checkpoint,args.output/'support_preserving_contact_consequence.pt')
        torch.save(dict(schema='ref2dex.support_preserving_contact_forecasts.v1',actual=predictions,candidates=candidate_predictions,frozen_old_cm=frozen,
            candidate_columns=['event0','event1','event2','event3','event4','event5','event6','event7',
                'expected_supported_height_10mm','geometric_loss_probability','supported_lift_probability','conditional_height_10mm','terminal_joint_probability','any_contact_loss_probability'],
            target={k:v.cpu() for k,v in target.items()},bucket=bucket.cpu(),origin=origin.cpu(),assignment=arm.cpu(),
            normalization_inherited_old_fit_only=True),args.output/'forecasts.pt')
        check()
        if any(sha(Path(k))!=v for k,v in hashes.items()):raise ValueError('training source/input drift')
        result=dict(run_status='COMPLETED',label=label,counts=counts,metrics=reports,frozen_old_cm_metrics=old_reports,gate=gates,
            parameter_counts=parameter_counts,elapsed_seconds=time.monotonic()-start,
            cumulative_seconds=prior+time.monotonic()-start,shared_old_source_cost_separate=True,
            checkpoint_sha256=sha(args.output/'support_preserving_contact_consequence.pt'),forecast_sha256=sha(args.output/'forecasts.pt'),
            scope=manifest['scope'])
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        manifest.update(run_status='COMPLETED',input_hashes_unchanged=True,result_sha256=sha(args.output/'results.json'),
            elapsed_seconds=time.monotonic()-start,cumulative_seconds=result['cumulative_seconds']);save();check()
        print(json.dumps(dict(run_status='COMPLETED',label=label,gate=gates,metrics=reports['cm'],
            cumulative_seconds=result['cumulative_seconds'])),flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',error=repr(error),elapsed_seconds=time.monotonic()-start);save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu',type=int,default=5);parser.add_argument('--premature-attempt',type=Path)
    run(parser.parse_args())
