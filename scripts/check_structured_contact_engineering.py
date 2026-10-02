#!/usr/bin/env python3
"""Excluded engineering seed: independent labels/PD, no-future and gradients."""
import argparse
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission
BASE=ROOT/'src/task/CmResidual/research/contact_consequence/output'


def independent_labels(b):
    h=b['future_hand_force'].numpy().astype(np.float64)
    obj=b['future_object_force'].numpy().astype(np.float64)
    weight=b['mass_kg'].numpy().astype(np.float64)*b['gravity_magnitude']
    hand=np.linalg.norm(h,axis=-1).max(-1)/weight[:,None]>.1
    object_present=np.linalg.norm(obj,axis=-1)/weight[:,None]>.1
    clear=b['future_clearance'].numpy()>=.002
    event=np.all(hand[:,-3:],-1)*4+np.all(object_present[:,-3:],-1)*2+np.all(clear[:,-3:],-1)
    height=np.maximum(0,b['future_state'][:,-3:,38].numpy().min(-1)-b['rest_z'].numpy())/.01
    loss=[]
    for initial,values in zip(b['initial_clearance'].numpy()>=.002,clear):
        seen=bool(initial);failed=False
        for value in values:
            failed=failed or (seen and not bool(value));seen=seen or bool(value)
        loss.append(failed)
    return dict(event=event,height=height,support=event==7,supported_height=height*(event==7),
                loss=np.array(loss),lift=(event==7)&(height>=3))


def independent_pd(a,q,offset,scale):
    raw=a.copy();raw[...,6:]=(1+raw[...,6:])*.5
    out=offset+scale*raw;out[...,:6]+=q[...,:6]
    for dst,src,ratio in ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)):
        out[...,dst]=out[...,src]*ratio
    return out


def run(args):
    started=time.monotonic()
    if args.output.exists():raise ValueError('unique engineering artifact required')
    source=BASE/'P-20261002-optimized-contact-native-engineering-r2/seed590/records.pt'
    paths=[source,Path(__file__),ROOT/'src/task/CmResidual/structured_contact_consequence.py',
        ROOT/'scripts/structured_contact_data.py',ROOT/'docs/decisions/D-20261002-structured-contact-consequences.md',
        ROOT/'src/task/CmResidual/contact_geometry_consequence.py',ROOT/'src/task/CmResidual/native_pd_selector.py']
    hashes={str(p.resolve()):sha(p) for p in paths}
    admission=gpu_admission(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    import torch
    from src.task.CmResidual.structured_contact_consequence import record_features,factual_labels,normalization,normalize,StructuredContactConsequence,event_marginals,objective,current_features
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    b=torch.load(source,map_location='cpu',weights_only=False)
    if b['seed']!=590 or len(b['state'])!=96:
        raise ValueError('excluded audited engineering source required')
    labels=factual_labels(b);expected=independent_labels(b);errors={}
    for k,v in expected.items():
        error=float(np.max(np.abs(labels[k].numpy().astype(float)-v.astype(float))))
        errors['label_'+k]=error
        if error>2e-4:raise ValueError('independent label '+k)
    d=record_features(b);n=len(b['state']);ids=np.arange(n);chosen=b['assignment'].numpy()
    channels=((0,1,2,3,4,5),(6,7),(8,9),(10,11),(12,13),(14,15,16,17))
    q=b['state'][:,:18].numpy();off=b['pd_offset'].numpy();scale=b['pd_scale'].numpy()
    cup=independent_pd(b['candidate_actions'][:,1].numpy(),q,off,scale)
    actual=independent_pd(b['candidate_actions'].numpy()[ids,chosen],q,off,scale)
    bank=independent_pd(b['expert_bank'][:,0].numpy(),q[:,None],off,scale)
    goal=np.zeros((n,6,6),dtype=np.float32);cup_goal=goal.copy();bank_goal=np.zeros((n,6,6,6),dtype=np.float32)
    for node,indices in enumerate(channels):
        goal[:,node,:len(indices)]=(actual-cup)[:,indices]
        cup_goal[:,node,:len(indices)]=(cup-q)[:,indices]
        bank_goal[:,node,:,:len(indices)]=(bank-cup[:,None])[...,indices]
    reference=np.zeros((n,6,6),dtype=np.float32);reference[:,:,1]=1
    na=np.concatenate((goal,b['candidate_weights'].numpy()[ids,chosen]-reference),-1)
    errors['reference_pd_action']=float(np.max(np.abs(na-d['node_action'].numpy())))
    errors['current_cup_goal']=float(np.max(np.abs(cup_goal-d['node_context'].numpy()[...,24:30])))
    errors['current_bank_pd']=float(np.max(np.abs(bank_goal.reshape(n,6,36)-d['node_context'].numpy()[...,30:])))
    if any(v>2e-5 for k,v in errors.items() if 'label_' not in k):raise ValueError('independent input PD')
    # Poison every future-only field and later within-window observations/banks.
    poisoned=dict(b)
    for k in ('future_state','future_native_observation','future_hand_force','future_object_force','future_clearance'):
        poisoned[k]=torch.full_like(b[k],float('nan'))
    poisoned['native_observation']=b['native_observation'].clone();poisoned['native_observation'][:,1:]=float('nan')
    poisoned['expert_bank']=b['expert_bank'].clone();poisoned['expert_bank'][:,1:]=float('nan')
    other=record_features(poisoned)
    if any(not torch.equal(d[k],other[k]) for k in d):raise ValueError('future leakage')
    signature=str(inspect.signature(current_features))
    if 'future' in signature:raise ValueError('current-only interface')
    norm=normalization(d,torch.ones(n,dtype=torch.bool));f={k:v.cuda() for k,v in normalize(d,norm).items()}
    y={k:v.cuda() for k,v in labels.items()}
    torch.manual_seed(12801);model=StructuredContactConsequence(d['physical'].shape[-1]).cuda().eval()
    f['node_action']=f['node_action'].detach().requires_grad_(True)
    output=model(**f);marginals=event_marginals(output['event_probability'])
    subset=(output['lift_probability'][:,None]-marginals).max().item()
    support_subset=(output['support_probability'][:,None]-marginals).max().item()
    if subset>1e-6 or support_subset>1e-6 or (output['supported_height']<0).any():raise ValueError('joint event containment')
    loss=objective(output,y,'cm');loss.backward()
    gradient=float(f['node_action'].grad.abs().max())
    if not torch.isfinite(loss) or not torch.isfinite(f['node_action'].grad).all() or gradient<=1e-8:
        raise ValueError('finite action gradient required')
    state=StructuredContactConsequence(d['physical'].shape[-1],'state_only').cuda().eval()
    a=state(**{k:v.detach() for k,v in f.items()})
    alternate={k:v.detach().clone() for k,v in f.items()};alternate['node_action']=torch.randn_like(alternate['node_action']);alternate['law']=1-alternate['law']
    c=state(**alternate)
    invariance=max(float((a[k]-c[k]).abs().max()) for k in a)
    if invariance!=0:raise ValueError('state-only action invariance')
    torch.cuda.synchronize()
    if any(sha(Path(k))!=v for k,v in hashes.items()):raise ValueError('engineering input drift')
    result=dict(run_status='COMPLETED',engineering_passed=True,excluded_seed=590,rows=n,
        independent_errors=errors,future_poison_unchanged=True,current_only_signature=signature,
        subset_max_excess=subset,support_subset_max_excess=support_subset,
        finite_action_gradient_max=gradient,state_only_action_invariance=invariance,
        input_sha256=hashes,gpu=admission,model_updates=0,elapsed_seconds=time.monotonic()-started,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        scope='input/label/event/gradient engineering only; no fitting or utility claim')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','gpu')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu',type=int,default=1)
    run(parser.parse_args())
