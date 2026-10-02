#!/usr/bin/env python3
"""Accept only independently reconstructed labels, fit statistics and NN outputs."""
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
    if args.output.exists():raise ValueError('unique fit audit output required')
    manifest=json.loads((args.fit/'run_manifest.json').read_text())
    result=json.loads((args.fit/'results.json').read_text())
    if manifest['run_status']!='COMPLETED' or result['run_status']!='COMPLETED':raise ValueError('terminal fit required')
    if any(sha(Path(k))!=v for k,v in manifest['input_sha256'].items()):raise ValueError('fit source hash drift')
    checkpoint_path=args.fit/'contact_geometry_consequence.pt';candidate_path=args.fit/'candidate_predictions.pt'
    if sha(checkpoint_path)!=result['checkpoint_sha256'] or sha(candidate_path)!=result['candidate_predictions_sha256']:
        raise ValueError('model/candidate artifact drift')
    expected_runs=[(mode,seed,1000) for mode in ('cm','state_only','shuffled','direct_score') for seed in (12601,12602,12603)]
    if [(r['mode'],r['seed'],r['updates']) for r in manifest['models']]!=expected_runs:
        raise ValueError('matched model/update contract')
    admission=gpu_admission(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    import torch
    from qualify_contact_geometry_source import load
    from audit_contact_geometry_model_inputs import independent
    from fit_contact_geometry_consequence import metrics
    from src.task.CmResidual.contact_geometry_consequence import transitions,normalize,candidate_features,FEATURES,GeometryConsequenceModel
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    records,counts,adequate,source_hashes=load(args.source,args.source_audit)
    p=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if p['schema']!='ref2dex.contact_geometry_consequence.v1' or p['input_sha256']!=manifest['input_sha256']:
        raise ValueError('checkpoint/source contract')
    candidate=torch.load(candidate_path,map_location='cpu',weights_only=False)
    parts=[];expected_parts=[];input_errors={};actual_candidate_errors={}
    for b in records:
        d=transitions(b);expected,errors=independent(b,d);parts.append(d);expected_parts.append(expected)
        for k,v in errors.items():input_errors[k]=max(input_errors.get(k,0),v)
        c=candidate_features(b,d);idx=torch.arange(len(b['state']))*8+b['assignment']
        for k in list(FEATURES)+['prior']:
            error=float((c[k][idx]-d[k][::10]).abs().max())
            actual_candidate_errors[k]=max(actual_candidate_errors.get(k,0),error)
    if max(actual_candidate_errors.values())>2e-6:raise ValueError('actual candidate input reconstruction')
    raw={k:torch.cat([d[k] for d in parts]) for k in parts[0]}
    independently_assembled={k:torch.cat([d[k] for d in expected_parts]) for k in FEATURES}
    buckets=torch.cat([b['split_group_bucket'].repeat_interleave(10) for b in records])
    fit=buckets<50;normal_errors={}
    for k in FEATURES:
        x=independently_assembled[k][fit];dims=(0,1) if k in ('history','node_state','node_action') else 0
        for suffix,v in [('mean',x.mean(dims)),('std',x.std(dims,unbiased=False).clamp_min(.001))]:
            key=k+'_'+suffix;normal_errors[key]=float((v-p['normalization'][key]).abs().max())
    if max(normal_errors.values())>2e-5:raise ValueError('fit-only independent statistics reconstruction')
    data={k:v.cuda() for k,v in raw.items()};norm={k:v.cuda() for k,v in p['normalization'].items()}
    features=normalize(data,norm);splits={'cal':((buckets>=50)&(buckets<70)).cuda(),'held':(buckets>=70).cuda()}
    metric_errors={};candidate_errors={}
    for mode,states in p['models'].items():
        if len(states)!=3:raise ValueError('three matched members required')
        predictions=[];candidate_predictions=[]
        for state in states:
            model=GeometryConsequenceModel(p['physical_dim'],mode).cuda()
            model.load_state_dict(state,strict=True);model.eval().requires_grad_(False)
            with torch.no_grad():
                outputs=[]
                for offset in range(0,len(buckets),512):
                    s=slice(offset,offset+512)
                    outputs.append(model(*(features[k][s] for k in FEATURES),data['prior'][s]))
                predictions.append(torch.cat(outputs))
                candidates=[]
                for b,d in zip(records,parts):
                    c={k:v.cuda() for k,v in candidate_features(b,d).items()};f=normalize(c,norm);out=[]
                    for offset in range(0,len(c['prior']),512):
                        s=slice(offset,offset+512)
                        out.append(model(*(f[k][s] for k in FEATURES),c['prior'][s])[:,[49,51]])
                    candidates.append(torch.cat(out).reshape(len(b['state']),8,2).cpu())
                candidate_predictions.append(torch.cat(candidates))
            del model
        pred=torch.stack(predictions).mean(0)
        for split,mask in splits.items():
            actual=metrics(pred,data['target'],data['prior'],data['first'],mask)
            for k,v in actual.items():metric_errors[f'{mode}/{split}/{k}']=abs(v-result['metrics'][mode][split][k])
        candidate_errors[mode]=float((torch.stack(candidate_predictions)-candidate['predictions'][mode]).abs().max())
        if result['cumulative_seconds']+time.monotonic()-begin>3600:raise TimeoutError('whole slot audit budget')
    if max(metric_errors.values())>2e-5 or max(candidate_errors.values())>2e-5:
        raise ValueError('actual metric/NN candidate replay drift')
    r=dict(run_status='COMPLETED',audit_passed=True,gpu=admission,counts=counts,
           all52labels_and_pre_inputs_errors=input_errors,actual_candidate_input_errors=actual_candidate_errors,
           fit_only_normalizer_errors=normal_errors,metric_replay_max_error=max(metric_errors.values()),
           candidate_NN_replay_errors=candidate_errors,checkpoint_sha256=sha(checkpoint_path),
           candidate_predictions_sha256=sha(candidate_path),fit_manifest_sha256=sha(args.fit/'run_manifest.json'),
           auditor_sha256=sha(Path(__file__)),elapsed_seconds=time.monotonic()-begin,
           cumulative_seconds=result['cumulative_seconds']+time.monotonic()-begin,
           scope='all actual physical input/labels, matched12models, fit-only statistics and NN replay; no actual controller utility')
    args.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='gpu'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','source-audit','fit','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',type=int,default=1);run(p.parse_args())
