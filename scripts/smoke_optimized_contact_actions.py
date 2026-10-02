#!/usr/bin/env python3
"""GPU engineering of frozen consequence gradient proposals on real banks."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission


def run(args):
    begin=time.monotonic()
    if args.output.exists():raise ValueError('unique engineering artifact required')
    admission=gpu_admission(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    import torch
    from src.task.CmResidual.optimized_contact_actions import FrozenConsequenceActionGenerator,live_inputs
    from src.task.CmResidual.contact_geometry_consequence import transitions,FEATURES,normalize,node_inputs
    from src.task.CmResidual.executable_contact_options import hold_target
    from src.task.CmResidual.paired_evaluation import fingerprint
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    b=torch.load(args.record,map_location='cpu',weights_only=False)
    b={k:(v[:24] if torch.is_tensor(v) and len(v.shape)>0 and len(v)==len(b['state']) else v) for k,v in b.items()}
    b['outcome']={k:v[:24] for k,v in b['outcome'].items()}
    # Nested outcome tensors and episode metadata are not needed by live_inputs.
    n=len(b['state']);raw=transitions(b)
    c={k:v.to('cuda') for k,v in b.items() if torch.is_tensor(v)}
    x=live_inputs(c['history'],c['native_observation'][:,0],c['initial_hand_force'],c['initial_object_force'],
                  c['mass_kg'],b['gravity_magnitude'],c['initial_clearance'],c['rest_z'])
    live_errors={k:float((value.cpu()-raw[k][::10]).abs().max()) for k,value in x.items()}
    if max(live_errors.values())>2e-5:raise ValueError('online pre-only input/skip mismatch')
    generator=FrozenConsequenceActionGenerator(args.checkpoint,'cuda')
    before=fingerprint([{mode:[m.state_dict() for m in models]} for mode,models in generator.models.items()])
    bank=c['expert_bank'][:,0];position=c['state'][:,:18];offset=c['pd_offset'];scale=c['pd_scale']
    anchor=hold_target(position,offset,scale);reference=torch.zeros(n,6,6,device='cuda');reference[:,:,1]=1
    reports={}
    for mode in ('cm','direct_score','shuffled'):
        context=generator.context(mode,x)
        with torch.no_grad():
            cached_score,cached_risk,_,targets=generator.predict(mode,reference,bank,anchor,position,offset,scale,context)
            _,node_action=node_inputs(c['state'],c['native_observation'][:,0],targets,reference)
            original=dict(x,node_action=node_action,law=torch.ones(n,1,device='cuda'))
            normalized=normalize(original,generator.norm)
            members=[]
            for model in generator.models[mode]:
                pred=model(*(normalized[k] for k in FEATURES),original['prior'])
                members.append(torch.stack((pred[:,51]*10,torch.sigmoid(pred[:,49])),-1))
            expected=torch.stack(members)
            cache_error=float((torch.stack((cached_score,cached_risk),-1)-expected).abs().max())
        if cache_error>2e-5:raise ValueError('cached physical head differs from frozen model')
        # Test the real no_grad collector call site: planning must enable only input gradients.
        with torch.no_grad():
            weights,info=generator.optimize(mode,bank,anchor,position,offset,scale,x)
        if not torch.isfinite(weights).all() or (weights<0).any() or not torch.allclose(weights.sum(-1),torch.ones(n,6,device='cuda'),atol=2e-6):
            raise ValueError('planned simplex domain')
        action=info['initial_command'];targets=info['initial_pd_targets']
        if not torch.allclose(targets[:,3:6],anchor[:,3:6],atol=2e-5):raise ValueError('planned rotation target')
        for channels in (slice(0,3),slice(6,18)):
            lower=bank[...,channels].amin(1);upper=bank[...,channels].amax(1)
            if (action[:,channels]<lower-2e-6).any() or (action[:,channels]>upper+2e-6).any():raise ValueError('planned expert command envelope')
        if info['trace'].shape!=(33,n,4) or not torch.isfinite(info['trace']).all():raise ValueError('bounded planning trace')
        reports[mode]=dict(cache_max_error=cache_error,states=n,changed_reference=int(info['changed_reference'].sum()),
                           weight_sum_max_error=float((weights.sum(-1)-1).abs().max()),
                           rotation_target_max_error=float((targets[:,3:6]-anchor[:,3:6]).abs().max()),
                           best_relative_score_mm_mean=float((info['predicted_score_mm']-cached_score).mean()),
                           steps=info['optimize_steps'])
        if time.monotonic()-begin>300:raise TimeoutError('5min gradient engineering budget')
    after=fingerprint([{mode:[m.state_dict() for m in models]} for mode,models in generator.models.items()])
    if before!=after or any(p.requires_grad for models in generator.models.values() for m in models for p in m.parameters()):
        raise ValueError('frozen consequence parameters changed')
    paths=[Path(__file__),ROOT/'src/task/CmResidual/optimized_contact_actions.py',args.record,args.checkpoint,
           ROOT/'docs/decisions/D-20261002-cm-optimized-action-generation.md']
    r=dict(run_status='COMPLETED',engineering_passed=True,gpu=admission,pre_only_live_input_errors=live_errors,
           modes=reports,model_parameters_frozen=True,input_sha256={str(p.resolve()):sha(p) for p in paths},
           elapsed_seconds=time.monotonic()-begin,scope='GPU input-gradient and executable coefficient engineering; generated actions NOT simulated, no candidate opportunity/utility claim')
    args.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('gpu','input_sha256')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('record','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',type=int,default=1);run(p.parse_args())
