"""Analytic control-intervention regression test at the actual composed score seam."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.task.CmResidual.option_model_policy import scores,initialized_network


class Dynamics(torch.nn.Module):
    def forward(self,x):
        result=x.new_zeros((len(x),131));result[:,38]=x[:,152];return result


class Value(torch.nn.Module):
    def forward(self,x):return torch.sigmoid(x[:,38:39])


state=torch.zeros((4,152));known=torch.zeros((4,21));mean=torch.zeros(131);std=torch.ones(131);models=dict(cm=Dynamics(),dynamics_off=Dynamics(),value=Value(),direct_q=Value())
for variant,expected in [('cm',.25),('dynamics_off',0.)]:
    request=torch.zeros((4,12),requires_grad=True);value=scores(state,request,known,models,mean,std,variant)
    gradient=torch.autograd.grad(value.sum(),request)[0];assert torch.allclose(gradient[:,0],torch.full((4,),expected),atol=1e-7,rtol=0) and not gradient[:,1:].any()
actor=initialized_network('actor');opt=torch.optim.SGD(actor.parameters(),lr=.1)
before=actor(state).detach();assert not before.any();loss=-scores(state,actor(state),known,models,mean,std,'cm').mean();loss.backward();opt.step();assert (actor(state)[:,0]>0).all()
print('PASS analytic dV/drequest=.25 for physical Cm path, zero for action-removed path; actual initialized actor optimization changes the proposal. Tiny CPU smoke, no research data.')
