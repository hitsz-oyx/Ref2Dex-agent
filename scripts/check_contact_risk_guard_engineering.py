#!/usr/bin/env python3
"""Excluded current-state risk guard: full-NN replay and real PD differences."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission


def run(args):
    begin = time.monotonic()
    if args.output.exists(): raise ValueError('unique engineering directory')
    source = ROOT/'src/task/CmResidual/research/contact_consequence/output/P-20261002-structured-opportunity-engineering-r1/seed623/records.pt'
    audit_path = ROOT/'docs/experiments/probes/P-20261002-support-preserving-contact-fit-audit-r2.json'
    checkpoint = ROOT/'src/task/CmResidual/research/contact_consequence/output/P-20261002-support-preserving-contact-fit-r2/support_preserving_contact_consequence.pt'
    accepted = json.loads(audit_path.read_text())
    if not accepted['audit_passed'] or accepted['checkpoint_sha256'] != sha(checkpoint):
        raise ValueError('accepted frozen contact consequence heads required')
    admission = gpu_admission(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = admission['uuid']
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from src.task.CmResidual.contact_risk_guard import ContactRiskGuard
    from src.task.CmResidual.structured_contact_actions import live_inputs
    from src.task.CmResidual.support_preserving_consequence import current_features,normalize
    from src.task.CmResidual.optimized_contact_actions import mixed_command
    from src.task.CmResidual.paired_evaluation import fingerprint
    from check_support_preserving_engineering import independent_pd
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    b = torch.load(source,map_location='cpu',weights_only=False)
    if b['seed'] != 623 or len(b['state']) != 33: raise ValueError('excluded input set')
    def current(record):
        return live_inputs(record['history'].cuda(),record['native_observation'][:,0].cuda(),
            record['initial_hand_force'].cuda(),record['initial_object_force'].cuda(),record['mass_kg'].cuda(),
            record['gravity_magnitude'],record['initial_clearance'].cuda(),record['rest_z'].cuda())
    inputs = current(b)
    poisoned = dict(b)
    for key in ('future_state','future_native_observation','future_hand_force','future_object_force','future_clearance'):
        poisoned[key] = torch.full_like(b[key],float('nan'))
    other = current(poisoned)
    if any(not torch.equal(value,other[key]) if torch.is_tensor(value) else value!=other[key] for key,value in inputs.items()):
        raise ValueError('future input leakage')
    bank = b['expert_bank'][:,0].cuda();anchor=b['hold_target'].cuda();q=b['state'][:,:18].cuda()
    offset=b['pd_offset'].cuda();scale=b['pd_scale'].cuda()
    generator = ContactRiskGuard(checkpoint,'cuda')
    before = fingerprint({k:[m.state_dict() for m in models] for k,models in generator.models.items()})
    reference = bank.new_zeros(len(bank),6,6);reference[:,:,1]=1
    cup = mixed_command(bank,reference,anchor,q,offset,scale)
    def full(weights,command,risk_mode):
        data=current_features(**inputs,bank=bank,cup_action=cup,candidate_action=command,weights=weights,
            law=bank.new_ones(len(bank)),offset=offset,scale=scale)
        features=normalize(data,generator.norm)
        with torch.no_grad():
            height=torch.stack([model(**features)['supported_height']*10 for model in generator.models['direct_score']])
            risks=[]
            for model in generator.models[risk_mode]:
                output=model(**features)
                risks.append(torch.stack((output['joint_probability'],output['contact_loss_probability'],
                    output['loss_probability'],output['support_probability']),-1))
        return height,torch.stack(risks)
    reports={};weights_by_mode={};errors={}
    for mode in ('unguarded','state_only','shuffled','cm'):
        if time.monotonic()-begin>240: raise TimeoutError('bounded guard engineering')
        weights,report=generator.optimize(mode,bank,anchor,q,offset,scale,inputs)
        height,risk=full(weights,report['initial_command'],report['risk_source'])
        ref_height,ref_risk=full(reference,cup,report['risk_source'])
        for key,actual,expected in (('height',height,report['predicted_score_mm']),('risk',risk,report['predicted_risk']),
                ('reference_height',ref_height,report['reference_score_mm']),('reference_risk',ref_risk,report['reference_risk'])):
            error=float((actual-expected).abs().max());errors[mode+'_'+key]=error
            if not torch.allclose(actual,expected,atol=2e-5,rtol=2e-6):raise ValueError('full NN '+mode+' '+key)
        pd=independent_pd(report['initial_command'].cpu().numpy(),q.cpu().numpy(),offset.cpu().numpy(),scale.cpu().numpy())
        error=float(abs(pd-report['initial_pd_targets'].cpu().numpy()).max());errors[mode+'_pd']=error
        if error>2e-5:raise ValueError('independent native PD')
        if (weights<0).any() or not torch.allclose(weights.sum(-1),torch.ones_like(weights[...,0]),atol=2e-6):
            raise ValueError('convex executable coefficients')
        changed=report['changed_reference'];diff=(risk-ref_risk).mean(0)
        if mode!='unguarded' and changed.any() and ((diff[changed,0]<-.02-1e-6).any() or
                (diff[changed,1:3]>.02+1e-6).any() or (diff[changed,3]<-.05-1e-6).any()):
            raise ValueError('hard current risk constraints')
        if not torch.equal(weights[report['context_ood']],reference[report['context_ood']]):
            raise ValueError('OOD fallback')
        weights_by_mode[mode]=weights
        reports[mode]={k:(v.detach().cpu() if torch.is_tensor(v) else v) for k,v in report.items()}
    if not torch.equal(weights_by_mode['state_only'],weights_by_mode['unguarded']):
        raise ValueError('candidate-independent state risk must equal unguarded policy')
    unguarded=reports['unguarded'];cm=reports['cm']
    _,direct_risk=full(weights_by_mode['unguarded'],unguarded['initial_command'].cuda(),'cm')
    _,cup_risk=full(reference,cup,'cm')
    difference=(direct_risk-cup_risk).mean(0)
    unsafe=(difference[:,0]<-.02)|(difference[:,1]>.02)|(difference[:,2]>.02)|(difference[:,3]<-.05)
    changed_pd=(cm['initial_pd_targets']-unguarded['initial_pd_targets']).abs().amax(-1)>1e-5
    binding=unsafe.cpu()&changed_pd
    after=fingerprint({k:[m.state_dict() for m in models] for k,models in generator.models.items()})
    if before!=after or any(p.grad is not None or p.requires_grad for models in generator.models.values() for m in models for p in m.parameters()):
        raise ValueError('frozen NN parameters')
    paths=[Path(__file__),source,checkpoint,audit_path,ROOT/'src/task/CmResidual/contact_risk_guard.py',
        ROOT/'src/task/CmResidual/support_preserving_consequence.py',ROOT/'src/task/CmResidual/structured_contact_actions.py',
        ROOT/'docs/decisions/D-20261002-contact-risk-constrained-scoring.md']
    hashes={str(p.resolve()):sha(p) for p in paths}
    args.output.mkdir()
    torch.save(dict(weights={k:v.cpu() for k,v in weights_by_mode.items()},reports=reports,
        current_inputs={k:v.cpu() if torch.is_tensor(v) else v for k,v in inputs.items()},
        future_used=False),args.output/'planning.pt')
    result=dict(run_status='COMPLETED',engineering_passed=True,excluded_seed=623,rows=33,
        full_nn_and_pd_errors=errors,model_parameters_frozen=True,future_poison_unchanged=True,
        state_only_equals_unguarded=True,unguarded_cm_predicted_violations=int(unsafe.sum()),
        cm_changes_unguarded_pd=int(changed_pd.sum()),binding_guard_changes=int(binding.sum()),
        cm_changes_cup=int(cm['changed_reference'].sum()),context_ood=int(cm['context_ood'].sum()),
        practical_binding_coverage=bool(binding.any()),input_sha256=hashes,
        checkpoint_sha256=sha(checkpoint),planning_sha256=sha(args.output/'planning.pt'),
        elapsed_seconds=time.monotonic()-begin,gpu=admission,
        scope='excluded pre-only engineering; predicted risk and PD changes, not actual outcomes or utility')
    if result['elapsed_seconds']>300 or sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())>64<<20:
        raise ValueError('fixed engineering bound')
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','gpu')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=5)
    run(p.parse_args())
