#!/usr/bin/env python3
"""Diagnose scale shift and probabilistic contact reliability, without fitting."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission


def run(args):
    admission=gpu_admission(args.gpu)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (str(args.gpu),admission['uuid']):raise ValueError('explicit idleGPU required')
    import torch
    from src.task.CmResidual.contact_ranker import physical_history
    from src.task.CmResidual.contact_trajectory import TrajectoryNetwork,all_trajectories,decode_trajectories,trajectory_targets
    torch.set_num_threads(2);device=torch.device('cuda:0')
    saved=torch.load(args.fit/'trajectory.pt',map_location='cpu',weights_only=False)
    report=json.loads((args.fit/'results.json').read_text())
    if sha(args.fit/'trajectory.pt')!=report['trajectory_sha256']:raise ValueError('model changed')
    payload=[]
    for path,digest in saved['source_record_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError('source changed')
        payload.append(torch.load(path,map_location='cpu',weights_only=False))
    data={k:torch.cat([p[k] for p in payload]) for k in ['history','rest_z','candidate_actions','assignment','motion_id','start_frame','trigger','state','future_state','future_contact']}
    split=torch.tensor(json.loads((args.fit/'split.json').read_text())['row_split'])
    fit=(split==0).nonzero().flatten();cal=(split==1).nonzero().flatten()
    raw=physical_history(data['history'],data['rest_z'])
    h=(raw-saved['history_mean'])/saved['history_scale']
    fit_mean=raw[fit].mean((0,1));fit_scale=raw[fit].std((0,1),unbiased=False).clamp_min(.001)
    rebased=(raw-fit_mean)/fit_scale
    targets=trajectory_targets(data['future_state'],data['future_contact'].all(-1),data['state'][:,38],data['rest_z'])
    result=dict(scope='terminal diagnostic; no refit, model selection or change to failed primary gate',normalization={},contact={})
    for name,rows in [('fit',fit),('calibration',cal)]:
        row_max=h[rows].abs().amax((1,2));columns=h[rows].abs().amax((0,1));top=columns.argsort(descending=True)[:10]
        result['normalization'][name]=dict(rows=len(rows),old_norm_max=float(row_max.max()),
            rows_above10_fraction=float((row_max>10).float().mean()),rows_above100_fraction=float((row_max>100).float().mean()),
            fit_rebased_max=float(rebased[rows].abs().max()),
            top_columns=[dict(column=int(i),maximum_old_abs_norm=float(columns[i]),old_mean=float(saved['history_mean'][i]),
                old_scale=float(saved['history_scale'][i]),new_fit_mean=float(fit_mean[i]),new_fit_scale=float(fit_scale[i])) for i in top.tolist()])
    candidate=(data['candidate_actions'].to(device)-saved['action_mean'].to(device))/saved['action_scale'].to(device)
    context=torch.cat((data['candidate_actions'][:,4],torch.nn.functional.one_hot(data['motion_id'].long(),3).float(),
                       ((data['start_frame']+data['trigger']).float()/600)[:,None]),-1).to(device)
    context=(context-saved['context_mean'].to(device))/saved['context_scale'].to(device)
    h=h.to(device);truth=targets['joint_contact'][cal]
    prior=float(targets['joint_contact'][fit].mean());high=h[cal].abs().amax((1,2)).cpu()>10
    for variant,states in saved['models'].items():
        members=[]
        with torch.no_grad(),torch.random.fork_rng(devices=[]):
            for state in states:
                model=TrajectoryNetwork(state_only=variant=='state_only').to(device);model.load_state_dict(state);model.eval()
                chunks=[]
                for offset in range(0,len(cal),128):
                    rows=cal[offset:offset+128]
                    p=decode_trajectories(all_trajectories(model,h[rows],candidate[rows],context[rows]),saved['height_mean'].to(device),saved['height_scale'].to(device))
                    chunks.append(p['joint_contact'][torch.arange(len(rows),device=device),data['assignment'][rows].to(device)].cpu())
                members.append(torch.cat(chunks))
        probability=torch.stack(members).mean(0);bins=[];ece=0.
        for b in range(10):
            mask=(probability>=b/10)&(probability<(b+1)/10 if b<9 else probability<=1)
            if mask.any():
                q=float(probability[mask].mean());observed=float(truth[mask].mean())
                ece+=int(mask.sum())/len(mask)*abs(q-observed)
                bins.append(dict(rows=int(mask.sum()),prediction=q,observed=observed))
        result['contact'][variant]=dict(brier=float((probability-truth).square().mean()),
            constant_brier=float((prior-truth).square().mean()),mae=float((probability-truth).abs().mean()),
            empirical_positive_rate=float(truth.mean()),fit_constant_prediction=prior,descriptive_ece=ece,bins=bins,
            high_norm_mae=float((probability[high]-truth[high]).abs().mean()) if high.any() else None,
            low_norm_mae=float((probability[~high]-truth[~high]).abs().mean()) if (~high).any() else None)
    result['input_hashes_unchanged']=sha(args.fit/'trajectory.pt')==report['trajectory_sha256'] and all(sha(Path(p))==v for p,v in saved['source_record_sha256'].items())
    if args.output.exists():raise ValueError('audit exists')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=4)
    run(p.parse_args())
