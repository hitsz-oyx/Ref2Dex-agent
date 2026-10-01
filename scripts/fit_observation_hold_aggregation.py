#!/usr/bin/env python3
"""One fixed fresh-state corrective-label aggregation, never fitting old tests."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.audit_observation_hold_failure import expert_targets
from scripts.run_contact_response_probe import sha
from src.task.CmResidual.dexplore_bc_policy import DExploreBcPolicy,normalized_action
from src.task.CmResidual.observation_hold_policy import ACTION_SCALES,SCHEMA,canonical_action
from src.task.CmResidual.paired_evaluation import fingerprint

def corrective_labels(initial,data):
    goal=expert_targets(initial);x=data['context'].numpy();offset=initial['pd_offset'].numpy();scale=initial['pd_scale'].numpy()
    y=(goal-offset)/scale;y[...,:6]=(goal[...,:6]-offset[:6]-x[...,:6])/scale[:6];y[...,6:]=2*y[...,6:]-1;y[...,[7,9,11,13,16,17]]=0
    if not np.isfinite(y).all() or np.any(np.abs(y)>1+1e-6):raise ValueError('teacher query outside action bounds; do not clip/filter')
    return torch.from_numpy(y.astype(np.float32))

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);a=p.parse_args();root=a.directory;torch.set_num_threads(2)
    out=root/'fit';out.mkdir(exist_ok=False);begin=time.monotonic()
    if not torch.cuda.is_available():raise ValueError('GPU fit required')
    m=json.loads((root/'run_manifest.json').read_text());xs=[];ys=[];provenance={}
    for directory,is_teacher in ((a.baseline/'teacher',True),(root/'collect513',False),(root/'collect514',False)):
        r=json.loads((directory/'results.json').read_text())
        if r['run_status']!='COMPLETED' or r['mode']!=('teacher' if is_teacher else 'policy'):raise ValueError('collection mode/provenance')
        for name in ('initial','trace'):
            f=directory/(name+'.pt');h=sha(f)
            if h!=r[name+'_sha256']:raise ValueError('collection data drift')
            provenance[str(f)]=h
        initial=torch.load(directory/'initial.pt',map_location='cpu',weights_only=False);data=torch.load(directory/'trace.pt',map_location='cpu',weights_only=False)
        queried=corrective_labels(initial,data)
        if is_teacher:
            if not torch.allclose(queried,data['action'],atol=1e-5,rtol=0):raise ValueError('teacher independent reconstruction')
            queried=data['action']
        elif r['policy_checkpoint_sha256']!=m['baseline_policy_sha256'] or r['learned_policy_calls']!=202:raise ValueError('frozen behavior policy')
        xs.append(data['context'].reshape(-1,70));ys.append(queried.reshape(-1,18))
    x=torch.cat(xs);y=torch.cat(ys)
    if x.shape!=(58176,70) or not torch.isfinite(x).all() or y.shape!=(58176,18):raise ValueError('complete all-row aggregate')
    dataset=out/'aggregate.pt';torch.save(dict(context=x,action=y,source_sha256=provenance,old_test_rows_used=0),dataset)
    source=a.baseline/'fit/policy.pt'
    if sha(source)!=m['baseline_policy_sha256']:raise ValueError('warm start drift')
    payload=torch.load(source,map_location='cpu',weights_only=False)
    if payload['schema']!=SCHEMA or payload['source_actor_weights_used'] or payload['official_policy_checkpoint'] is not None:raise ValueError('scratch source provenance')
    device=torch.device('cuda:0');torch.manual_seed(735);torch.cuda.manual_seed_all(735)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    x=x.to(device);y=y.to(device);mean=payload['observation_mean'].to(device);std=payload['observation_std'].to(device)
    model=DExploreBcPolicy(70,18,tuple(payload['hidden_dims'])).to(device);model.load_state_dict(payload['model'])
    initial_fp=fingerprint(model.state_dict())
    if initial_fp!=payload['final_model_fingerprint']:raise ValueError('warm start fingerprint')
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);scales=torch.tensor(ACTION_SCALES,device=device);loss_trace=[]
    for update in range(2000):
        if time.monotonic()-begin>300:raise TimeoutError('fit budget')
        ids=torch.randint(len(x),(512,),device=device);pred=canonical_action(normalized_action(model,x[ids],mean,std));loss=((pred-y[ids])/scales).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise ValueError('nonfinite fit')
        if (update+1)%200==0:loss_trace.append(dict(update=update+1,loss=float(loss)));print(json.dumps(loss_trace[-1]),flush=True)
    model.eval();path=out/'policy.pt'
    payload.update(model={k:v.cpu() for k,v in model.state_dict().items()},seed=735,updates=2000,aggregation_rounds=1,
        baseline_policy_sha256=sha(source),aggregate_sha256=sha(dataset),initial_model_fingerprint=initial_fp,final_model_fingerprint=fingerprint(model.state_dict()))
    torch.save(payload,path)
    result=dict(run_status='COMPLETED',updates=2000,fit_rows=len(x),no_test_selection=True,old_test_rows_used=0,collection_expert_actions_executed=0,
        initial_model_fingerprint=initial_fp,final_model_fingerprint=payload['final_model_fingerprint'],normalization_preserved=True,checkpoint_sha256=sha(path),
        baseline_policy_sha256=sha(source),aggregate_sha256=sha(dataset),source_sha256=provenance,loss_trace=loss_trace,wall_seconds=time.monotonic()-begin,device=str(device))
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
