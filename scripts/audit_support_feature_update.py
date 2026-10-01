#!/usr/bin/env python3
"""Independent NumPy network, score-gradient and Adam-update reconstruction."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha


def forward(x, weights):
    activations=[x.astype(np.float64)]
    for layer in ('0','2'):
        activations.append(np.maximum(activations[-1]@weights[layer+'.weight'].numpy().astype(np.float64).T+weights[layer+'.bias'].numpy(),0))
    logits=activations[-1]@weights['4.weight'].numpy().astype(np.float64).T+weights['4.bias'].numpy()
    return logits,activations


def probability(logit):
    e=np.exp(logit-logit.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)


def feature_inputs(raw,motion,source):
    saved=torch.load(source/'fit/predictions.pt',map_location='cpu',weights_only=False)
    result={};norm=None
    for variant in ('cm','state_only'):
        p=torch.load(source/'fit'/(variant+'.pt'),map_location='cpu',weights_only=False)
        z=np.clip((raw-p['mean'].numpy())/p['std'].numpy(),-10,10)
        norm=z
        macro=np.array([[0,0,0],[0,0,-1],[0,0,-.5],[0,0,.5],[-1,0,0],[1,0,0],[0,-1,0],[0,1,0]],dtype=np.float64)
        aa=np.broadcast_to(macro if variant=='cm' else np.zeros_like(macro),(len(raw),8,3))
        x=np.concatenate((np.broadcast_to(z[:,None],(len(raw),8,69)),aa),-1).reshape(-1,72)
        logits,_=forward(x,p['model'])
        pp=1/(1+np.exp(-np.clip(logits,-80,80)))
        result[variant]=np.concatenate((z,2*pp.reshape(len(raw),8)-1),-1)
    gp=saved['global_fit_probabilities'].numpy()[motion]
    result['global_motion_arm']=np.concatenate((norm,2*gp-1),-1)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    root=a.directory;out=root/'gradient_audit.json'
    if out.exists():raise ValueError('retain prior audit')
    begin=time.monotonic();torch.set_num_threads(2)
    packet=torch.load(root/'update_packet.pt',map_location='cpu',weights_only=False)
    after=torch.load(root/'policy_heads.pt',map_location='cpu',weights_only=False)
    raw=packet['raw_state'].numpy();motion=packet['motion'].numpy();reward=packet['reward'].numpy()
    action=packet['action'].numpy();behavior=packet['behavior_probability'].numpy();baseline=packet['state_baseline'].numpy()
    if len(raw)!=768 or np.any(behavior!=.125):raise ValueError('full panel/behavior law')
    source=Path(packet['source']);fit=torch.load(source/'dataset.pt',map_location='cpu',weights_only=False)
    ids=fit['seed']<525
    expected_b=np.array([float(fit['physical105'][ids & (fit['motion']==m)].float().mean()) for m in range(3)])
    if not np.array_equal(baseline,expected_b[motion]):raise ValueError('state-only FIT baseline')
    features=feature_inputs(raw,motion,source);maximum=dict(feature=0.,logits=0.,probability=0.,gradient=0.,adam_parameter=0.)
    names=['0.weight','0.bias','2.weight','2.bias','4.weight','4.bias']
    for variant,record in packet['records'].items():
        x=features[variant]
        maximum['feature']=max(maximum['feature'],float(np.max(np.abs(x-record['input'].numpy()))))
        weights=record['before'];logits,acts=forward(record['input'].numpy(),weights);pi=probability(logits)
        maximum['logits']=max(maximum['logits'],float(np.max(np.abs(logits-record['logits'].numpy()))))
        maximum['probability']=max(maximum['probability'],float(np.max(np.abs(pi-record['probability'].numpy()))))
        rho=pi[np.arange(768),action]/behavior
        dz=pi.copy();dz[np.arange(768),action]-=1;dz*=((rho*(reward-baseline))/768)[:,None]
        gradients={}
        for layer,k in (('4',2),('2',1),('0',0)):
            gradients[layer+'.weight']=dz.T@acts[k]
            gradients[layer+'.bias']=dz.sum(0)
            if k:
                dz=(dz@weights[layer+'.weight'].numpy())*(acts[k]>0)
        norm=np.sqrt(sum(np.sum(v*v) for v in gradients.values()))
        if abs(norm-record['gradient_norm'])>1e-5:raise ValueError('gradient norm')
        clip=min(1.,10/(record['gradient_norm']+1e-6))
        opt=record['optimizer_before'];params=opt['param_groups'][0]['params']
        for name,param in zip(names,params):
            grad=record['gradient_before_clipping'][name].numpy().astype(np.float64)
            maximum['gradient']=max(maximum['gradient'],float(np.max(np.abs(grad-gradients[name]))))
            state=opt['state'].get(param,{})
            m=state['exp_avg'].numpy().astype(np.float64) if state else np.zeros_like(grad)
            v=state['exp_avg_sq'].numpy().astype(np.float64) if state else np.zeros_like(grad)
            t=float(state['step'])+1 if state else 1.
            g=grad*clip;m=.9*m+.1*g;v=.999*v+.001*g*g
            expected=weights[name].numpy()-(.01/(1-.9**t))*m/(np.sqrt(v)/np.sqrt(1-.999**t)+1e-8)
            got=after['variants'][variant]['model'][name].numpy()
            maximum['adam_parameter']=max(maximum['adam_parameter'],float(np.max(np.abs(expected-got))))
    limits=dict(feature=2e-5,logits=2e-5,probability=2e-6,gradient=2e-5,adam_parameter=2e-5)
    if any(maximum[k]>limits[k] for k in maximum):raise ValueError(('independent numeric drift',maximum))
    result=dict(run_status='COMPLETED',rows=768,update=packet['update'],seed=packet['seed'],maximum_error=maximum,
        all_three_variants=True,real_reward105=True,state_only_baseline=True,
        all_head_and_trunk_gradients_reconstructed=True,independent_adam_update=True,
        device_reason='CPU independent NumPy numeric audit of GPU inference/gradients, not training',
        input_sha256={str(root/f):sha(root/f) for f in ('update_packet.pt','policy_heads.pt')},wall_seconds=time.monotonic()-begin)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
