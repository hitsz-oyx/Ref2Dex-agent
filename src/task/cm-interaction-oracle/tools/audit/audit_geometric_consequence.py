#!/usr/bin/env python3
"""Frozen-weight GPU replay and physical feature-support audit, no new fits."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5]
TASK=ROOT/'src/task/cm-interaction-oracle'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(TASK/'src'));sys.path.insert(0,str(TASK/'tools/run'))
from geometric_consequence import normalize,pca_apply,pad,action_design,ridge_predict,physics_baseline
from probe_conditional_consequence import preprocess
from consequence_sufficiency import readouts,PRIMARY


def gpu_tree(value,device):
    if isinstance(value,torch.Tensor): return value.to(device)
    if isinstance(value,dict): return {k:gpu_tree(v,device) for k,v in value.items()}
    return value


def summary(x,train,test):
    x=x.detach().cpu().numpy()
    return dict(train_abs_max=float(np.abs(x[train]).max()),test_abs_max=float(np.abs(x[test]).max()),
        train_row_norm95=float(np.quantile(np.linalg.norm(x[train],axis=-1),.95)),
        test_row_norm95=float(np.quantile(np.linalg.norm(x[test],axis=-1),.95)),
        test_row_norm_max=float(np.linalg.norm(x[test],axis=-1).max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    args=parser.parse_args()
    target=args.run_dir/'engineering_replay.json'
    if target.exists(): raise FileExistsError(target)
    torch.set_num_threads(2)
    device=torch.device('cuda:0')
    d=torch.load(args.run_dir/'diagnostic.pt',map_location='cpu',weights_only=False)
    p=torch.load(args.dataset,map_location='cpu',weights_only=False)
    train,test=d['train'],d['test']
    z=torch.cat(readouts(p)[:2],-1)
    assert torch.equal(z,d['z'])
    base=physics_baseline(p['before'].to(device),float(p['control_dt'])*8)
    assert torch.equal(base.cpu(),d['base'])
    raw=gpu_tree(d['geometry'],device)
    states=gpu_tree(d['predictor_states'],device)
    onehot=torch.nn.functional.one_hot(p['arm'].to(device),15)[:,1:].float()
    rows=torch.arange(len(p['arm']),device=device)
    arms=p['arm'].to(device)
    replay={};distributions={};error_concentration={}
    for prefix,n_cpu in d['preprocess'].items():
        n=gpu_tree(n_cpu,device)
        ids=torch.as_tensor(n_cpu['fit'],device=device)
        h,_,_=preprocess(p,ids,device,n['h'])
        h=torch.cat((h,pca_apply(raw['geometry'],n['geometry'])),-1)
        perm_arms=torch.as_tensor(d['arms'][d['permutations'][prefix]['permutation']],device=device)
        acts=dict(State=torch.zeros(len(arms),32,device=device),Arm=pad(normalize(onehot,n['arm'])),
            Joint=pad(normalize(raw['joint'][rows,arms],n['joint'])),
            Flow=pca_apply(raw['flow'][rows,arms],n['flow']),
            FlowShuffled=pca_apply(raw['flow'][rows,perm_arms],n['flow']))
        acts['RawState'],acts['RawArm']=acts['State'],acts['Arm']
        for name,a in acts.items():
            x=action_design(h,a)
            baseline=torch.zeros_like(base) if name.startswith('Raw') else base
            pred=ridge_predict(x,states[prefix+'_'+name])*n['target']['scale']+baseline
            expected=d['full_predictions'][name] if prefix=='full' else d['oof_predictions'][name].numpy()
            hold=n_cpu['hold']
            replay[prefix+'_'+name]=float(np.max(np.abs(pred.cpu().numpy()[hold]-expected[hold])))
            if prefix=='full':
                distributions[name]=dict(action=summary(a,train,test),state=summary(h,train,test),design=summary(x,train,test))
                err=(((pred.cpu()-z)/d['target_scale']).square()[:,12:].mean(1)).numpy()
                order=test[np.argsort(err[test])[::-1]]
                total=err[test].sum()
                error_concentration[name]=dict(top1_fraction=float(err[order[:1]].sum()/total),top5_fraction=float(err[order[:5]].sum()/total),
                    top10_fraction=float(err[order[:10]].sum()/total),
                    top_rows=[dict(row=int(r),environment=int(d['clusters'][r]),arm=int(d['arms'][r]),I_mse=float(err[r]),
                        action_norm=float(a[r].norm()),state_norm=float(h[r].norm())) for r in order[:10]])
        if prefix=='full':
            frozen=states['full_Flow']
            all_candidates=[]
            for arm in range(15):
                a=pca_apply(raw['flow'][test,arm],n['flow'])
                all_candidates.append((ridge_predict(action_design(h[test],a),frozen)*n['target']['scale']+base[test]).cpu().numpy())
            replay['Flow_candidates']=float(np.abs(np.stack(all_candidates,1)-d['candidate_arrays']['Flow']).max())
            # Downstream scorer can be replayed from serialized OOF/full values.
            task_h=torch.cat((h,(base-d['target_mean'].to(device))/d['target_scale'].to(device)),-1)
            blank=torch.zeros_like(base)
            zmean,zscale=d['target_mean'].to(device),d['target_scale'].to(device)
            specs=dict(H=(acts['State'],blank),Arm=(acts['Arm'],blank),Joint=(acts['Joint'],blank),Flow=(acts['Flow'],blank),
                GT=(acts['State'],(z.to(device)-zmean)/zscale),
                P_State=(acts['State'],(d['oof_predictions']['State'].to(device)-zmean)/zscale),
                P_Flow=(acts['State'],(d['oof_predictions']['Flow'].to(device)-zmean)/zscale),
                Flow_P_Flow=(acts['Flow'],(d['oof_predictions']['Flow'].to(device)-zmean)/zscale))
            for name,(a,phys) in specs.items():
                model=gpu_tree(d['downstream_states'][name],device)
                predicted=ridge_predict(torch.cat((action_design(task_h,a),phys),-1),model)*d['task_target_scale'].to(device)+d['task_target_mean'].to(device)
                replay['task_'+name]=float((predicted.cpu()-d['task_predictions'][name]).abs().max())
    assert max(replay.values())<1e-4
    # Pure statistical audit of per-component projection support, not fitting.
    n=gpu_tree(d['preprocess']['full'],device)
    scores=pca_apply(raw['flow'][rows,arms],n['flow']).cpu().numpy()
    shift=dict(train_std=scores[train].std(0).tolist(),test_std=scores[test].std(0).tolist(),
        test_abs_max=np.abs(scores[test]).max(0).tolist(),pca_score_scale_m=n['flow']['score']['scale'].cpu().tolist())
    report=dict(status='PASS',max_frozen_replay_error=max(replay.values()),replays=replay,
        feature_distributions=distributions,I_error_concentration=error_concentration,flow_projection_support=shift,
        limits='Shared-helper GPU replay establishes stored-weight consistency, not independent semantic validation; separate review required.')
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS',max_replay_error=max(replay.values()),Flow=error_concentration['Flow'],Flow_features=distributions['Flow']),indent=2))


if __name__=='__main__':main()
