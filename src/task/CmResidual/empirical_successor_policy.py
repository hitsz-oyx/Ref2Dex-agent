"""Physical-only conditional law over complete observed successor feature tuples."""
import numpy as np
import torch
from src.task.CmResidual.option_model_policy import DYNAMIC

SCHEMA='observed-successor-atomic-law-kernel-h8-v1'


def encoder():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3560)
        return torch.nn.Sequential(torch.nn.Linear(164,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,16))


def inputs(current,raw,variant):
    action=torch.tanh(raw) if variant=='cm' else torch.zeros_like(raw)
    assert variant in ('cm','dynamics_off')
    return torch.cat((current,action),-1)


def log_weights(network,query,donors,own=None):
    q=network(query);k=network(donors)
    logits=-(q[:,None]-k[None]).square().sum(-1)/16
    if own is not None:
        exclude=torch.arange(len(donors),device=logits.device)[None]==own[:,None]
        logits=logits.masked_fill(exclude,float('-inf'))
    return torch.log_softmax(logits,-1)


def physical_kernel(future):
    # Pure statistics in float64 prevent Gram-matrix cancellation. No network
    # inference or scientific model fit is moved off GPU here.
    x=np.asarray(future[:,DYNAMIC],np.float64);out=np.empty((len(x),len(x)),np.float64)
    for start in range(0,len(x),32):
        out[start:start+32]=-np.sum((x[start:start+32,None]-x[None])**2,-1)/(2*.5**2)
    return out


def expected_value(value,physical,known,action,weights):
    # Exact algebraic decomposition of the first affine layer avoids expanding
    # [batch,1536,164] inputs; the later nonlinearities are still applied to
    # EACH complete observed tuple, before probability averaging.
    first=value[0]
    atom=physical@first.weight[:,DYNAMIC].T
    query=known@first.weight[:,51:72].T+action@first.weight[:,152:].T+first.bias
    hidden=torch.relu(atom[None]+query[:,None])
    hidden=torch.relu(torch.nn.functional.linear(hidden,value[2].weight,value[2].bias))
    probabilities=torch.sigmoid(torch.nn.functional.linear(hidden,value[4].weight,value[4].bias)).squeeze(-1)
    return (probabilities*weights).sum(-1)


def score(network,value,current,raw,known,donor_current,donor_raw,physical,variant,own=None):
    logw=log_weights(network,inputs(current,raw,variant),inputs(donor_current,donor_raw,variant),own)
    return expected_value(value,physical,known,torch.tanh(raw),logw.exp())
