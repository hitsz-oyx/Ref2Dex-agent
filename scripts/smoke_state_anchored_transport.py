"""Synthetic four-row differentiable mixture/zero-far-gradient NumPy smoke."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.task.CmResidual.state_anchored_transport import TransportHead,tokens,log_prior,predict,flow_loss
from scripts.audit_state_anchored_transport import replay


def main():
    torch.set_num_threads(2);rng=np.random.default_rng(4300)
    raw=rng.normal(size=(4,15,120)).astype(np.float32);raw[:,:,:99]=raw[:,0:1,:99]
    base=rng.normal(scale=.001,size=(4,64,3));endpoints=rng.normal(scale=.002,size=(4,15,64,3));target=rng.normal(scale=.001,size=(4,64,3))
    gate=np.asarray([True,False,True,False]);maximum=loss_max=0.
    for arm in ('full','state_only','shuffled'):
        torch.manual_seed(4301);model=TransportHead();initial={k:v.detach().clone() for k,v in model.state_dict().items()}
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);x=torch.from_numpy(tokens(raw,arm));prior=torch.from_numpy(log_prior(arm))
        ids=np.asarray([0,1]*6+[0]) if arm=='state_only' else np.arange(2,15)
        b=torch.from_numpy(base).float();e=torch.from_numpy(endpoints[:,ids]).float();g=torch.from_numpy(gate);permutation=np.asarray([3,0,1,2])
        y=torch.from_numpy(target[permutation] if arm=='shuffled' else target).float();losses=[];early=[]
        optimizer.zero_grad(set_to_none=True);out=predict(model,x,b,e,torch.zeros_like(g),prior);loss=flow_loss(out['prediction'],y);loss.backward()
        assert all(torch.count_nonzero(v.grad)==0 for v in model.parameters()) and torch.equal(out['prediction'],b)
        with torch.no_grad():
            initial_out=predict(model,x,b,e,g,prior);assert torch.allclose(initial_out['weights'][g,0],torch.full((2,),.99),atol=1e-6)
        for step in range(3):
            optimizer.zero_grad(set_to_none=True);out=predict(model,x,b,e,g,prior);loss=flow_loss(out['prediction'],y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
            losses.append(float(loss.detach()));early.append({k:v.detach().clone() for k,v in model.state_dict().items()})
        ck=dict(arm=arm,common_initial=initial,input_mean=torch.zeros(120),input_std=torch.ones(120),batch_schedule=torch.tensor([[0,1,2,3]]*3),losses=losses,early_states=early,label_permutation=torch.from_numpy(permutation) if arm=='shuffled' else None)
        error,delta=replay(ck,raw,base,endpoints,target,gate);maximum=max(maximum,error);loss_max=max(loss_max,delta)
        with torch.no_grad():
            before=predict(model,x,b,e,g,prior);target[:]+=123 # future label has no deploy argument
            after=predict(model,x,b,e,g,prior);assert all(torch.equal(v,after[k]) for k,v in before.items())
        target[:]-=123
    print(json.dumps(dict(engineering_smoke='PASS',synthetic_rows=4,all3arms_3updates=True,zero_far_gradient=True,
                         future_outcome_isolation=True,independent_numpy_parameter_max=maximum,independent_numpy_loss_max=loss_max,
                         cpu_reason='tiny synthetic gradient engineering cheaper than GPU startup')),flush=True)


if __name__=='__main__':main()
