"""Observed-support expectation/gradient and off-action seam smoke, no data."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.task.CmResidual.option_model_policy import DYNAMIC
from src.task.CmResidual.empirical_successor_policy import expected_value,log_weights,inputs

value=torch.nn.Sequential(torch.nn.Linear(164,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1),torch.nn.Sigmoid()).double()
for p in value.parameters():p.data.zero_()
value[0].weight.data[0,0]=1;value[0].bias.data[0]=3;value[2].weight.data[0,0]=1;value[4].weight.data[0,0]=2;value[4].bias.data[0]=-6
physical=torch.zeros((2,131),dtype=torch.float64);physical[:,0]=torch.tensor([-2.,2.]);known=torch.zeros((1,21),dtype=torch.float64);action=torch.zeros((1,12),dtype=torch.float64);p=torch.tensor(.25,dtype=torch.float64,requires_grad=True);w=torch.stack((1-p,p))[None]
expected=expected_value(value,physical,known,action,w);state=torch.zeros((1,152),dtype=torch.float64);state[:,DYNAMIC]=w@physical;mean=value(torch.cat((state,action),-1)).flatten()
assert abs(float(expected-mean))>.1 and abs(float(torch.autograd.grad(expected,p,retain_graph=True)[0]-torch.autograd.grad(mean,p)[0]))>.1
class ActionEncoder(torch.nn.Module):
    def forward(self,x):return x[:,152:]
current=torch.zeros((1,152));raw=torch.ones((1,12),requires_grad=True);donor=torch.zeros((2,152));donor_raw=torch.stack((torch.zeros(12),torch.ones(12)))
for variant in ('cm','dynamics_off'):
    probability=log_weights(ActionEncoder(),inputs(current,raw,variant),inputs(donor,donor_raw,variant)).exp()[0,1]
    derivative=torch.autograd.grad(probability,raw,retain_graph=True)[0] if probability.requires_grad else torch.zeros_like(raw)
    assert (derivative.abs().sum()>0) if variant=='cm' else not derivative.any()
print('PASS full-tuple expectation differs from value-of-mean in value AND gradient; action-removed law deletes its action path. CPU analytic smoke only.')
