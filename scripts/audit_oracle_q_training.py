"""Independent NumPy reference for normalization and first3 AdamW steps/arm.

CPU is intentional: separate arithmetic/runtime from actual CUDA PyTorch fit.
Remaining5988 optimizer steps are not replayed; final ALL-eval inference has
its separate NumPy audit in the experiment analyzer.
"""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    import numpy as np,torch
    from scripts.run_contact_response_probe import sha
    out=a.source;begin=time.monotonic();load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    packet=load(out/'fit/feature_packet.pt')['train'];cp=load(out/'fit/selector.pt');schedule=np.load(out/'fit/minibatches.npy')
    x=packet['features'].reshape(-1,cp['input_dim']).astype(np.float64);y=packet['labels'].reshape(-1).astype(np.float64)
    mean=x.mean(0);std=np.maximum(x.std(0),.01);mean_error=float(np.max(np.abs(mean-cp['mean'].numpy())));std_error=float(np.max(np.abs(std-cp['std'].numpy())))
    if mean_error>1e-4 or std_error>1e-4:raise ValueError('TRAIN-only CUDA normalization')
    z=np.clip((x-cp['mean'].numpy())/cp['std'].numpy(),-10,10);errors={}
    for arm in ('state','effect','interaction','joint'):
        inputs=z.copy();c,e=cp['common_dim'],cp['effect_dim']
        if arm in ('state','interaction'):inputs[:,c:c+e]=0
        if arm in ('state','effect'):inputs[:,c+e:]=0
        weights={k:v.numpy().astype(np.float64).copy() for k,v in cp['initial'].items()};m={k:np.zeros_like(v) for k,v in weights.items()};v={k:np.zeros_like(v) for k,v in weights.items()};loss_errors=[]
        for step in range(1,4):
            ids=schedule[step-1];a0=inputs[ids];z1=a0@weights['network.0.weight'].T+weights['network.0.bias'];a1=np.maximum(z1,0);z2=a1@weights['network.2.weight'].T+weights['network.2.bias'];a2=np.maximum(z2,0);logits=(a2@weights['network.4.weight'].T+weights['network.4.bias']).ravel()
            loss=np.mean(np.maximum(logits,0)-logits*y[ids]+np.log1p(np.exp(-np.abs(logits))))
            loss_errors.append(abs(float(loss)-cp['losses'][arm][step-1]))
            delta=((1/(1+np.exp(-np.clip(logits,-80,80))))-y[ids])[:,None]/len(ids)
            gradient={};gradient['network.4.weight']=delta.T@a2;gradient['network.4.bias']=delta.sum(0)
            d2=(delta@weights['network.4.weight'])*(z2>0);gradient['network.2.weight']=d2.T@a1;gradient['network.2.bias']=d2.sum(0)
            d1=(d2@weights['network.2.weight'])*(z1>0);gradient['network.0.weight']=d1.T@a0;gradient['network.0.bias']=d1.sum(0)
            norm=np.sqrt(sum(np.sum(g*g) for g in gradient.values()));clip=min(1.,1./(norm+1e-6))
            for key in weights:
                g=gradient[key]*clip;m[key]=.9*m[key]+.1*g;v[key]=.999*v[key]+.001*g*g
                weights[key]=weights[key]*(1.-3e-4*1e-4)-3e-4*(m[key]/(1.-.9**step))/(np.sqrt(v[key]/(1.-.999**step))+1e-8)
        errors[arm]=loss_errors
        if max(loss_errors)>1e-4:raise ValueError('independent first3 AdamW/BCE/clip steps: '+arm)
        del inputs,weights,m,v
    result=dict(run_status='COMPLETED',engineering_only=True,independent_cpu_numpy_reason='separate numerical backend for CUDA implementation audit',normalization_mean_max_error=mean_error,normalization_std_max_error=std_error,first3_loss_errors=errors,actual_optimizer_steps=6000,independently_replayed_steps=12,remaining_optimizer_steps_not_replayed=5988,input_sha256={str(out/'fit'/f):sha(out/'fit'/f) for f in ('selector.pt','feature_packet.pt','minibatches.npy')},wall_seconds=time.monotonic()-begin)
    with (out/'q_training_audit.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
