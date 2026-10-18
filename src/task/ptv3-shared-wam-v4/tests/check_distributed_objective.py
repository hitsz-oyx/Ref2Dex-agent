"""Actual NCCL/DDP counterpart of the complete-objective partition regression."""
from datetime import timedelta
import json
import os

import torch
import torch.distributed as dist

from test_audit_regressions import objective_fixture
from test_v4_contract import config, stats
from ptv3_shared_wam_v4.objectives import forecast_loss
from ptv3_shared_wam_v4.normalization import HorizonNormalizer


class Predictions(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values=torch.nn.ParameterDict({k:torch.nn.Parameter(v.detach().clone()) for k,v in values.items()})

    def forward(self, ids):
        return {k:v[ids] for k,v in self.values.items()}


def main():
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
    device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(seconds=60))
    try:
        rank,world=dist.get_rank(),dist.get_world_size()
        b,p=objective_fixture()
        b={k:v.to(device) for k,v in b.items()}
        normalizer=HorizonNormalizer(stats()).to(device)
        m=Predictions(p).to(device)
        wrapped=torch.nn.parallel.DistributedDataParallel(m,device_ids=[local])
        full={k:v.detach().clone().requires_grad_() for k,v in m.values.items()}
        reference,_=forecast_loss(full,b,config(),normalizer)
        expected=torch.autograd.grad(reference,list(full.values()))
        discrepancies=[]
        for interleaved in (False,True):
            wrapped.zero_grad(set_to_none=True)
            ids=(torch.arange(rank,8,world,device=device) if interleaved else
                 torch.arange(rank*8//world,(rank+1)*8//world,device=device))
            loss,_=forecast_loss(wrapped(ids),{k:v[ids] for k,v in b.items()},config(),normalizer,distributed=True)
            loss.backward()
            for parameter,gradient in zip(m.values.values(),expected):
                torch.testing.assert_close(parameter.grad,gradient,atol=1e-7,rtol=1e-5)
                discrepancies.append(float((parameter.grad-gradient).abs().max()))
            average=loss.detach().clone();dist.all_reduce(average);average/=world
            torch.testing.assert_close(average,reference.detach(),atol=1e-6,rtol=1e-6)
        if rank==0:
            print(json.dumps(dict(status='PASS',world_size=world,complete_loss=float(reference),
                max_gradient_error=max(discrepancies),partitions=['grouped','interleaved'],
                scope='actual NCCL/DDP, synthetic masked labels, complete v4 loss')),flush=True)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':main()
