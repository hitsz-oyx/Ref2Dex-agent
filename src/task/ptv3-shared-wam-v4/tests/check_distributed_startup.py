"""Run with torchrun: rank0 evaluation must finish before collective training."""
from datetime import timedelta
import json
import os

import torch
from torch import nn
import torch.distributed as dist

from test_v4_contract import model, batch, config
from ptv3_shared_wam_v4.objectives import forecast_loss
from ptv3_shared_wam_v4.training import gradient_audit
from ptv3_shared_wam.loading import to_device


def main():
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    torch.set_num_threads(1)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", timeout=timedelta(seconds=30))
    try:
        m = model().to(device)
        data = to_device(batch(), device)
        audit = gradient_audit(m, data, config())
        wrapped = nn.parallel.DistributedDataParallel(m, device_ids=[local])
        if rank == 0:
            with torch.no_grad():
                loss, _ = forecast_loss(m(data), data, config(), m.normalizer, distributed=False)
            assert torch.isfinite(loss)
            print(json.dumps(dict(stage="rank0_evaluation_passed", loss=float(loss))), flush=True)
        # This reproduces the production call order: only rank0 evaluates;
        # all remaining ranks are already waiting at this barrier.
        dist.barrier()
        m.train()
        optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4)
        before = m.interaction_head[-1].weight.detach().clone()
        loss, _ = forecast_loss(wrapped(data), data, config(), m.normalizer, distributed=True)
        loss.backward()
        assert torch.isfinite(loss)
        optimizer.step()
        assert not torch.equal(before, m.interaction_head[-1].weight.detach())
        dist.barrier()
        if rank == 0:
            print(json.dumps(dict(status="PASS", world_size=dist.get_world_size(),
                training_loss=float(loss), interaction_gradient_audit=audit,
                scope="synthetic batch, real v4 modules, stand-in geometry; production data smoke required")), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
