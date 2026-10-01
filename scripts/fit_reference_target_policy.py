#!/usr/bin/env python3
"""One fixed scratch absolute-target fit on retained teacher rows; no validation-selected endpoint."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.task.CmResidual.dexplore_bc_policy import DExploreBcPolicy,normalized_action
from src.task.CmResidual.observation_hold_policy import canonical_action
from src.task.CmResidual.reference_target_policy import SCHEMA,residual_labels
from scripts.run_contact_response_probe import sha
from src.task.CmResidual.paired_evaluation import fingerprint


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory;torch.set_num_threads(2)
    teacher=ROOT/'src/task/CmResidual/research/contact_response/output/P-20261002-observation-hold-baseline-r1/teacher';meta=json.loads((teacher/'results.json').read_text());datafile=teacher/'trace.pt'
    if meta['mode']!='teacher' or meta['run_status']!='COMPLETED' or sha(datafile)!=meta['trace_sha256']:raise ValueError('teacher provenance')
    out=root/'fit';out.mkdir(exist_ok=False);begin=time.monotonic()
    if not torch.cuda.is_available():raise ValueError('GPU fit required')
    device=torch.device('cuda:0');torch.manual_seed(736);torch.cuda.manual_seed_all(736)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=torch.load(datafile,map_location='cpu',weights_only=False)
    x=data['context'].reshape(-1,70).to(device);y=residual_labels(data['target'].reshape(-1,18).to(device),x[:,51:69])
    if x.shape!=(19392,70) or not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('all teacher rows required')
    mean=x.mean(0);std=x.std(0,unbiased=False).clamp_min(.001);model=DExploreBcPolicy(70,18,(512,256,128)).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    initial=fingerprint(model.state_dict());loss_trace=[]
    for update in range(2000):
        if time.monotonic()-begin>300:raise TimeoutError('fixed fit wall budget')
        ids=torch.randint(len(x),(512,),device=device);pred=canonical_action(normalized_action(model,x[ids],mean,std))
        loss=(pred-y[ids]).square().mean();optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise ValueError('nonfinite fit')
        if (update+1)%200==0:loss_trace.append(dict(update=update+1,loss=float(loss)));print(json.dumps(loss_trace[-1]),flush=True)
    model.eval();path=out/'policy.pt'
    payload=dict(model={k:v.cpu() for k,v in model.state_dict().items()},observation_mean=mean.cpu(),observation_std=std.cpu(),
        observation_dim=70,action_dim=18,hidden_dims=(512,256,128),official_policy_checkpoint=None,source_actor_weights_used=False,
        schema=SCHEMA,seed=736,updates=2000,output_coordinates='reference-relative absolute native PD target',teacher_trace_sha256=sha(datafile),initial_random_fingerprint=initial,final_model_fingerprint=fingerprint(model.state_dict()))
    torch.save(payload,path)
    result=dict(run_status='COMPLETED',updates=2000,fit_rows=len(x),no_test_selection=True,random_initialization=True,
        initial_model_fingerprint=initial,final_model_fingerprint=payload['final_model_fingerprint'],checkpoint_sha256=sha(path),
        teacher_trace_sha256=sha(datafile),loss_trace=loss_trace,wall_seconds=time.monotonic()-begin,device=str(device))
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
