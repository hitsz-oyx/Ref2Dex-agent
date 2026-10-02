#!/usr/bin/env python3
"""Fixed matched GPU geometry/action consequence and direct-score fitting."""
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


def metrics(pred,target,prior,first,mask):
    import torch
    step=mask;window=mask&first
    error=pred[step]-target[step];macro=pred[window]-target[window]
    return dict(transitions=int(step.sum()),windows=int(window.sum()),
                relative_position_rmse_mm=float(error[:,:36].reshape(-1,6,6)[...,:3].square().mean().sqrt()*5),
                relative_velocity_rmse_mps=float(error[:,:36].reshape(-1,6,6)[...,3:].square().mean().sqrt()*.1),
                object_dv_rmse_mps=float(error[:,36:39].square().mean().sqrt()*.1),
                object_dx_rmse_mm=float(error[:,39:42].square().mean().sqrt()*5),
                clearance_delta_rmse_mm=float(error[:,42].square().mean().sqrt()*2),
                presence_brier=float((torch.sigmoid(pred[step,43:45])-target[step,43:45]).square().mean()),
                h10_joint_height_mae_mm=float(macro[:,51].abs().mean()*10),
                h10_lift_brier=float((torch.sigmoid(pred[window,50])-target[window,50]).square().mean()),
                h10_loss_brier=float((torch.sigmoid(pred[window,49])-target[window,49]).square().mean()),
                prior_object_dv_rmse_mps=float((prior[step,36:39]-target[step,36:39]).square().mean().sqrt()*.1),
                prior_relative_position_rmse_mm=float((prior[step,:36]-target[step,:36]).reshape(-1,6,6)[...,:3].square().mean().sqrt()*5))


def run(args):
    import torch
    from qualify_contact_geometry_source import load
    from src.task.CmResidual.contact_geometry_consequence import transitions,FEATURES,GeometryConsequenceModel,loss,normalize,candidate_features
    begin=time.monotonic()
    base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    if args.output.resolve().parent!=base or args.output.exists() or args.output.is_symlink():
        raise ValueError('unique owned fit output required')
    q=json.loads(args.qualification.read_text())
    if q['run_status']!='COMPLETED' or not q['supervision_adequate']:
        raise ValueError('qualified supervised panel required')
    records,counts,adequate,hashes=load(args.source,args.audit)
    if not adequate or counts!=q['counts'] or hashes!=q['input_sha256']:
        raise ValueError('qualified input drift')
    engineering=json.loads(args.engineering.read_text())
    if not engineering['engineering_passed'] or engineering['run_status']!='COMPLETED':
        raise ValueError('independent input/label and gradient smoke required')
    if any(sha(Path(k))!=v for k,v in engineering['input_sha256'].items()):
        raise ValueError('model engineering drift')
    source=json.loads((args.source/'run_manifest.json').read_text());audit=json.loads(args.audit.read_text())
    prior_seconds=source['cumulative_seconds']+audit['elapsed_seconds']+q['elapsed_seconds']+engineering['elapsed_seconds']+engineering['additional_preparation_seconds']
    prior_bytes=sum(p.stat().st_size for p in args.source.rglob('*') if p.is_file())
    old=json.loads((ROOT/'docs/experiments/probes/P-20261002-contact-geometry-native-engineering-completion-r2.json').read_text())
    prior_bytes+=old['total_output_bytes']
    for p in [Path(__file__),ROOT/'scripts/qualify_contact_geometry_source.py',
              ROOT/'src/task/CmResidual/contact_geometry_consequence.py',
              ROOT/'docs/experiments/probes/P-20261002-contact-geometry-model-contract.md',
              args.qualification,args.engineering]:
        hashes[str(p.resolve())]=sha(p)
    admission=gpu_admission(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir()
    manifest=dict(experiment_id='P-20261002-contact-geometry-action-information',family='HF19',probe_index_in_family=1,
                  run_status='STARTED',pid=os.getpid(),command=sys.argv,gpu=admission,models=[],input_sha256=hashes,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  prior_seconds=prior_seconds,prior_bytes=prior_bytes,wall_limit_seconds=3600,output_limit_bytes=8<<30)
    def save():(args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if prior_seconds+time.monotonic()-begin>3540:raise TimeoutError('whole slot wall budget')
        if prior_bytes+sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())>8<<30:raise ValueError('whole slot storage budget')
        if any(sha(Path(k))!=v for k,v in hashes.items()):raise ValueError('fit/source input drift')
    save()
    try:
        parts=[transitions(b) for b in records]
        raw={k:torch.cat([d[k] for d in parts]) for k in parts[0]}
        buckets=torch.cat([b['split_group_bucket'].repeat_interleave(10) for b in records]).cuda()
        splits={'fit':buckets<50,'cal':(buckets>=50)&(buckets<70),'held':buckets>=70}
        data={k:v.cuda() for k,v in raw.items()};norm={}
        for k in FEATURES:
            x=data[k][splits['fit']];dims=(0,1) if k in ('history','node_state','node_action') else 0
            norm[k+'_mean']=x.mean(dims);norm[k+'_std']=x.std(dims,unbiased=False).clamp_min(.001)
        features=normalize(data,norm)
        fit=splits['fit'].nonzero().flatten();starts=(splits['fit']&data['first']).nonzero().flatten()
        models={};reports={};selectors={};manifest.update(run_status='RUNNING',counts=counts);save()
        for mode in ('cm','state_only','shuffled','direct_score'):
            trained=[];predictions=[];candidate_predictions=[]
            for seed in (12601,12602,12603):
                check();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                model=GeometryConsequenceModel(data['physical'].shape[-1],mode).cuda()
                optimizer=torch.optim.Adam(model.parameters(),lr=.0003,weight_decay=.0001)
                gen=torch.Generator(device='cuda').manual_seed(seed+30000)
                action=features['node_action'].clone();law=features['law'].clone()
                if mode=='shuffled':
                    shuffler=torch.Generator(device='cuda').manual_seed(seed+40000)
                    for step in range(10):
                        for early in (False,True):
                            rows=fit[(fit%10==step)&(data['early'][fit]==early)]
                            shuffled=rows[torch.randperm(len(rows),device='cuda',generator=shuffler)]
                            action[rows]=features['node_action'][shuffled];law[rows]=features['law'][shuffled]
                model.train()
                for update in range(1000):
                    if update%100==0 and prior_seconds+time.monotonic()-begin>3540:
                        raise TimeoutError('whole slot training budget')
                    ids=torch.cat((starts[torch.randint(len(starts),(128,),device='cuda',generator=gen)],
                                   fit[torch.randint(len(fit),(128,),device='cuda',generator=gen)]))
                    optimizer.zero_grad(set_to_none=True)
                    prediction=model(features['history'][ids],features['physical'][ids],features['node_state'][ids],
                                     action[ids],law[ids],data['prior'][ids])
                    value=loss(prediction,data['target'][ids],data['first'][ids],mode)
                    value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                    if not torch.isfinite(value):raise ValueError('nonfinite loss')
                model.eval().requires_grad_(False)
                with torch.no_grad():
                    outputs=[]
                    for offset in range(0,len(buckets),512):
                        s=slice(offset,offset+512)
                        outputs.append(model(*(features[k][s] for k in FEATURES),data['prior'][s]))
                    predictions.append(torch.cat(outputs))
                    candidate_outputs=[]
                    for record,part in zip(records,parts):
                        c=candidate_features(record,part)
                        c={k:v.cuda() for k,v in c.items()};normalized=normalize(c,norm)
                        values=[]
                        for offset in range(0,len(c['prior']),512):
                            s=slice(offset,offset+512)
                            values.append(model(*(normalized[k][s] for k in FEATURES),c['prior'][s])[:,[49,51]])
                        candidate_outputs.append(torch.cat(values).reshape(len(record['state']),8,2).cpu())
                    candidate_predictions.append(torch.cat(candidate_outputs))
                trained.append({k:v.cpu().clone() for k,v in model.state_dict().items()})
                manifest['models'].append(dict(mode=mode,seed=seed,updates=1000,final_loss=float(value)));save()
                print(json.dumps(dict(mode=mode,seed=seed,status='COMPLETED')),flush=True)
                del model,optimizer
            mean=torch.stack(predictions).mean(0)
            reports[mode]={name:metrics(mean,data['target'],data['prior'],data['first'],mask) for name,mask in splits.items() if name!='fit'}
            models[mode]=trained;selectors[mode]=torch.stack(candidate_predictions)
        cm=reports['cm']['held'];gates=dict(supervision=adequate)
        for control in ('state_only','shuffled'):
            other=reports[control]['held']
            for metric in ('object_dv_rmse_mps','relative_position_rmse_mm','h10_joint_height_mae_mm'):
                gates[control+'_'+metric+'_5pct']=cm[metric]<=.95*other[metric]
            for metric in ('clearance_delta_rmse_mm','presence_brier'):
                gates[control+'_'+metric+'_no_worse2pct']=cm[metric]<=1.02*other[metric]
            gates[control+'_h10_lift_no_worse5pct']=cm['h10_lift_brier']<=1.05*other['h10_lift_brier']
        gates['passed']=all(gates.values())
        label='PROMISING' if gates['passed'] else 'UNPROMISING'
        checkpoint=dict(schema='ref2dex.contact_geometry_consequence.v1',physical_dim=data['physical'].shape[-1],models=models,
                        normalization={k:v.cpu() for k,v in norm.items()},input_sha256=hashes,
                        features=list(FEATURES),scope='per-node geometry/native-PD and known H10 programme physics; no unknown future actions')
        torch.save(checkpoint,args.output/'contact_geometry_consequence.pt')
        torch.save(dict(schema='ref2dex.contact_geometry_offline_candidates.v1',predictions=selectors,
                        source_manifest_sha256=sha(args.source/'run_manifest.json')),
                   args.output/'candidate_predictions.pt')
        check()
        result=dict(run_status='COMPLETED',label=label,metrics=reports,gate=gates,counts=counts,
                    elapsed_seconds=time.monotonic()-begin,cumulative_seconds=prior_seconds+time.monotonic()-begin,
                    scoped_bytes=prior_bytes+sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()),
                    checkpoint_sha256=sha(args.output/'contact_geometry_consequence.pt'),
                    candidate_predictions_sha256=sha(args.output/'candidate_predictions.pt'),
                    scope='factual physical information with matched action controls and direct score; offline policy/actual utility still separate')
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        manifest.update(run_status='COMPLETED',input_hashes_unchanged=True,result=result);print(json.dumps(result,indent=2))
    except BaseException as exc:
        manifest.update(run_status='FAILED',error=repr(exc));raise
    finally:
        manifest['elapsed_seconds']=time.monotonic()-begin;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','audit','qualification','engineering','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',type=int,default=1);run(p.parse_args())
