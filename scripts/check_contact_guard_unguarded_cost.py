#!/usr/bin/env python3
"""Actual GPU optimizer call-count diagnosis and equivalent-vectorization check."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import gpu_admission,sha


def run(args):
    os.environ['CUDA_VISIBLE_DEVICES']=gpu_admission(args.gpu)['uuid']
    import torch
    from src.task.CmResidual.contact_risk_guard import ContactRiskGuard
    from src.task.CmResidual.structured_contact_actions import live_inputs
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'src/task/CmResidual/research/contact_consequence/output'
    record=torch.load(base/'P-20261002-optimized-contact-native-engineering-r2/seed590/records.pt',map_location='cpu',weights_only=False)
    checkpoint=base/'P-20261002-support-preserving-contact-fit-r2/support_preserving_contact_consequence.pt'
    inputs=live_inputs(record['history'].cuda(),record['native_observation'][:,0].cuda(),
        record['initial_hand_force'].cuda(),record['initial_object_force'].cuda(),record['mass_kg'].cuda(),
        record['gravity_magnitude'],record['initial_clearance'].cuda(),record['rest_z'].cuda())
    parameters=(record['expert_bank'][:,0].cuda(),record['hold_target'].cuda(),record['state'][:,:18].cuda(),
        record['pd_offset'].cuda(),record['pd_scale'].cuda(),inputs)
    old=ContactRiskGuard(checkpoint,'cuda');count=[0];original=old.risk
    def counted(*a,**kw):
        count[0]+=1
        return original(*a,**kw)
    old.risk=counted
    torch.cuda.synchronize();begin=time.monotonic();weights,report=old.optimize('unguarded',*parameters)
    torch.cuda.synchronize();old_seconds=time.monotonic()-begin
    if args.legacy:
        print(json.dumps(dict(legacy_risk_calls=count[0],legacy_seconds=old_seconds)))
        if count[0]>4:raise AssertionError('unguarded objective repeats non-objective risk every input update')
        return
    from src.task.CmResidual.efficient_contact_risk_guard import EfficientContactRiskGuard
    new=EfficientContactRiskGuard(checkpoint,'cuda');counter=[0];method=new.risk
    def efficient_count(*a,**kw):
        counter[0]+=1
        return method(*a,**kw)
    new.risk=efficient_count
    torch.cuda.synchronize();begin=time.monotonic();fresh,new_report=new.optimize('unguarded',*parameters)
    torch.cuda.synchronize();new_seconds=time.monotonic()-begin
    errors={}
    for key in ('trace','predicted_score_mm','predicted_risk','reference_score_mm','reference_risk','initial_command','initial_pd_targets'):
        errors[key]=float((report[key]-new_report[key]).abs().max())
        if not torch.allclose(report[key],new_report[key],atol=2e-5,rtol=2e-6):raise AssertionError('equivalence '+key)
    if not torch.equal(weights,fresh):raise AssertionError('unguarded optimization weights changed')
    if counter[0]>4:raise AssertionError('batched diagnostic risk still repeated')
    result=dict(run_status='COMPLETED',engineering_passed=True,excluded_seed=590,rows=len(weights),
        weights_exact=True,report_errors=errors,legacy_risk_calls=count[0],efficient_risk_calls=counter[0],
        legacy_seconds=old_seconds,efficient_seconds=new_seconds,checkpoint_sha256=sha(checkpoint),
        scope='same objective, candidate, PD and risk trace; runtime engineering only')
    if args.output.exists():raise ValueError('unique equivalence output')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=5);p.add_argument('--legacy',action='store_true');p.add_argument('--output',type=Path)
    run(p.parse_args())
