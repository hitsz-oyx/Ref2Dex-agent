#!/usr/bin/env python3
"""Excluded current states: optimized coefficients, full NN and feasible fallback."""
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
BASE=ROOT/'src/task/CmResidual/research/contact_consequence/output'


def run(args):
    start=time.monotonic()
    if args.output.parent.resolve()!=BASE.resolve() or args.output.exists():raise ValueError('unique owned engineering output')
    source=BASE/'P-20261002-optimized-contact-native-engineering-r2/seed590/records.pt'
    checkpoint=BASE/'P-20261002-structured-contact-fit-r1/structured_contact_consequence.pt'
    audit_path=ROOT/'docs/experiments/probes/P-20261002-structured-contact-fit-audit-r1.json'
    audit=json.loads(audit_path.read_text())
    if not audit['audit_passed'] or audit['label']!='PROMISING' or audit['checkpoint_sha256']!=sha(checkpoint):
        raise ValueError('audited fixed information signal required')
    paths=[source,checkpoint,audit_path,Path(__file__),ROOT/'src/task/CmResidual/structured_contact_actions.py',
        ROOT/'src/task/CmResidual/structured_contact_consequence.py',ROOT/'src/task/CmResidual/optimized_contact_actions.py',
        ROOT/'src/task/CmResidual/native_pd_selector.py',ROOT/'src/task/CmResidual/paired_evaluation.py']
    hashes={str(p.resolve()):sha(p) for p in paths}
    admission=gpu_admission(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid'];os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from src.task.CmResidual.structured_contact_actions import FrozenStructuredActionGenerator,live_inputs
    from src.task.CmResidual.structured_contact_consequence import current_features,normalize
    from src.task.CmResidual.paired_evaluation import fingerprint
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    b=torch.load(source,map_location='cpu',weights_only=False)
    if b['seed']!=590 or len(b['state'])!=96:raise ValueError('excluded source')
    n=len(b['state']);position=b['state'][:,:18].cuda();bank=b['expert_bank'][:,0].cuda();anchor=b['hold_target'].cuda()
    offset=b['pd_offset'].cuda();scale=b['pd_scale'].cuda()
    inputs=live_inputs(b['history'].cuda(),b['native_observation'][:,0].cuda(),b['initial_hand_force'].cuda(),
        b['initial_object_force'].cuda(),b['mass_kg'].cuda(),b['gravity_magnitude'],b['initial_clearance'].cuda(),b['rest_z'].cuda())
    generator=FrozenStructuredActionGenerator(checkpoint,'cuda:0')
    before=fingerprint({k:[m.state_dict() for m in v] for k,v in generator.models.items()})
    reference=bank.new_zeros(n,6,6);reference[:,:,1]=1
    reports={};weights_by_mode={};metrics={}
    for mode in ('cm','direct_score','shuffled'):
        weights,report=generator.optimize(mode,bank,anchor,position,offset,scale,inputs)
        context=generator.context(mode,bank,anchor,position,offset,scale,inputs)
        errors={}
        def compare(key,actual,expected):
            error=float((actual-expected).abs().max());errors[key]=max(errors.get(key,0.),error)
            if not torch.allclose(actual,expected,atol=2e-4,rtol=2e-6):raise ValueError('cached/full NN '+key)
        with torch.no_grad():
            score,loss,support,command,pd,ood=generator.predict(mode,weights,bank,anchor,position,offset,scale,context)
            _,_,_,cup,cup_pd,_=generator.predict(mode,reference,bank,anchor,position,offset,scale,context)
            data=current_features(**inputs,bank=bank,cup_action=cup,candidate_action=command,
                weights=weights,law=bank.new_ones(n),offset=offset,scale=scale)
            f=normalize(data,generator.norm)
            for member,model in enumerate(generator.models[mode]):
                full=model(**f)
                compare('score_mm',score[member],full['supported_height']*10)
                if mode=='direct_score':full=generator.models['state_only'][member](**f)
                compare('loss',loss[member],full['loss_probability']);compare('support',support[member],full['support_probability'])
            compare('saved_score_mm',score,report['predicted_score_mm']);compare('saved_loss',loss,report['predicted_loss'])
            compare('saved_support',support,report['predicted_support']);compare('saved_command',command,report['initial_command'])
            compare('saved_pd',pd,report['initial_pd_targets'])
            if (weights<0).any() or not torch.allclose(weights.sum(-1),torch.ones_like(weights.sum(-1)),atol=2e-6):
                raise ValueError('convex expert domain')
            for start_channel,stop in ((0,3),(6,18)):
                lower=bank.min(1).values[:,start_channel:stop];upper=bank.max(1).values[:,start_channel:stop]
                if (command[:,start_channel:stop]<lower-2e-6).any() or (command[:,start_channel:stop]>upper+2e-6).any():raise ValueError('command expert domain')
            if float((pd[:,3:6]-anchor[:,3:6]).abs().max())>2e-5:raise ValueError('fixed current rotation')
            changed=report['changed_reference'];gain=score-report['reference_score_mm']
            mean=gain.mean(0);std=((gain-mean[None]).square().mean(0)+1e-6).sqrt()
            risk=(loss-report['reference_loss']).mean(0);retention=(support-report['reference_support']).mean(0)
            if changed.any() and ((mean[changed]-1.645*std[changed]<=0).any() or (risk[changed]>.02+1e-6).any()
                or (retention[changed]<-.05-1e-6).any() or report['context_ood'][changed].any() or ood[changed].any()):
                raise ValueError('selected programme hard feasibility')
            if report['context_ood'].any() and not torch.equal(weights[report['context_ood']],reference[report['context_ood']]):
                raise ValueError('OOD must abstain to cup')
            pd_changed=(pd-cup_pd).abs().amax(-1)>1e-5
            metrics[mode]=dict(rows=n,changed_weights=int(changed.sum()),changed_pd=int(pd_changed.sum()),
                context_ood=int(report['context_ood'].sum()),mean_predicted_gain_mm=float(mean.mean()),
                changed_predicted_gain_mm=float(mean[changed].mean()) if changed.any() else None,
                changed_predicted_support_difference=float(retention[changed].mean()) if changed.any() else None,
                max_errors=errors,trace_shape=list(report['trace'].shape))
        reports[mode]={k:v.cpu() if torch.is_tensor(v) else v for k,v in report.items()};weights_by_mode[mode]=weights.cpu()
        if time.monotonic()-start>300:raise TimeoutError('engineering bound')
    after=fingerprint({k:[m.state_dict() for m in v] for k,v in generator.models.items()})
    if before!=after or any(p.grad is not None for models in generator.models.values() for m in models for p in m.parameters()):
        raise ValueError('frozen parameters/no model gradients')
    if any(sha(Path(k))!=v for k,v in hashes.items()):raise ValueError('engineering input drift')
    torch.cuda.synchronize();args.output.mkdir()
    torch.save(dict(schema='ref2dex.structured_generation_engineering.v1',weights=weights_by_mode,reports=reports,
        inputs={k:v.cpu() if torch.is_tensor(v) else v for k,v in inputs.items()},bank=bank.cpu(),position=position.cpu(),
        anchor=anchor.cpu(),offset=offset.cpu(),scale=scale.cpu()),args.output/'planning.pt')
    result=dict(run_status='COMPLETED',engineering_passed=True,excluded_seed=590,metrics=metrics,
        input_sha256=hashes,parameters_frozen=True,parameter_fingerprint=before,gpu=admission,
        elapsed_seconds=time.monotonic()-start,planning_sha256=sha(args.output/'planning.pt'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        scope='frozen generation/executable domain/NN/abstain engineering; no new actual outcome or control benefit')
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('gpu','input_sha256')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu',type=int,default=1)
    run(parser.parse_args())
