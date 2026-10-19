"""Independent v5 pretraining entry point and bounded real-data engineering smoke."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import multiprocessing
import shutil
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v5 import paths
from ptv3_shared_wam_v5.contracts import SCHEMA,INPUT_SEMANTICS,LABEL_SEMANTICS
from ptv3_shared_wam_v5.data import V5Windows,collate
from ptv3_shared_wam_v5.model import SharedWAMV5
from ptv3_shared_wam_v5.storage import read_json
from ptv3_shared_wam_v5.checkpoints import checkpoint_due,retained_steps
from ptv3_shared_wam_v5.metrics import forecast_metrics,merge_metrics,summarize_metrics
from ptv3_shared_wam_v5.objectives import forecast_loss,loss_components,OBJECTIVE_SEMANTICS
from ptv3_shared_wam_v4.objectives import normalized_components,combine_components


def write(path,value):
    tmp=path.with_suffix('.json.part');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    path.chmod(0o664)


class Steps:
    def __init__(self,dataset,batch,seed,rank,start,last):
        self.dataset,self.batch,self.seed,self.rank,self.start,self.last=dataset,batch,seed,rank,start,last
    def __iter__(self):
        for step in range(self.start,self.last+1):yield self.dataset.draw(self.batch,self.seed,step,self.rank)
    def __len__(self):return self.last-self.start+1


def evaluate(model,dataset,config,device):
    if not len(dataset):return dict(status='NO_DEV_WINDOWS')
    was_training=model.training;model.eval();rng=np.random.default_rng(20261019);sources={}
    with torch.inference_mode():
        for sid,pool in dataset.pools.items():
            chosen=rng.choice(pool,min(len(pool),config.get('validation_samples',32)),replace=False)
            components=None;hand_sum=hand_count=0;diagnostics={}
            for start in range(0,len(chosen),config.get('validation_microbatch_size',8)):
                batch=collate([dataset[int(i)] for i in chosen[start:start+config.get('validation_microbatch_size',8)]])
                batch={k:v.to(device) for k,v in batch.items()}
                with torch.autocast('cuda' if device.startswith('cuda') else 'cpu',dtype=torch.bfloat16,enabled=device.startswith('cuda') and config.get('amp',True)):
                    prediction=model(batch)
                current=loss_components(prediction,batch,config,model.normalizer,reduction_dtype=torch.float64)
                if components is None:components=current
                else:components={k:(components[k][0]+v[0],components[k][1]+v[1]) for k,v in current.items()}
                mask=batch['trajectory_valid']&batch['start_hand_valid'][:,None]
                error=(prediction['trajectory']-torch.where(mask[...,None],batch['trajectory_target'],0.)).norm(dim=-1)
                hand_sum+=float(torch.where(mask,error,0.).sum())*1000;hand_count+=int(mask.sum())
                merge_metrics(diagnostics,forecast_metrics(prediction,batch))
            loss,terms=combine_components(normalized_components(components,distributed=False),config)
            sources[sid]=dict(objective=float(loss),hand_epe_mm=hand_sum/max(hand_count,1),
                metrics=summarize_metrics(diagnostics),
                components={k:float(v) for k,v in terms.items()})
    model.train(was_training)
    return dict(status='COMPLETED',sources=sources,source_macro_objective=float(np.mean([v['objective'] for v in sources.values()])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','manifest','stats','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true')
    p.add_argument('--steps',type=int);p.add_argument('--batch-size',type=int);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int)
    p.add_argument('--resume',type=Path);p.add_argument('--git-commit',required=True);a=p.parse_args()
    config=read_json(a.config);stats=read_json(a.stats)
    dataset=V5Windows(a.manifest,'train');last=a.steps or config['updates'];batch_size=a.batch_size or config['batch_size']
    if a.smoke:
        if not 1<=last<=16:raise ValueError('engineering smoke requires1..16 bounded steps')
    elif dataset.meta.get('training_allowed') is not True:
        raise ValueError('full training requires an explicitly admitted frozen v5 manifest')
    if last<1 or batch_size<1:raise ValueError('positive bounded steps/batch required')
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    device=f'cuda:{local}' if world>1 else a.device
    if device.startswith('cuda'):
        if not torch.cuda.is_available():raise ValueError('requested CUDA unavailable')
        torch.cuda.set_device(torch.device(device));torch.backends.cuda.matmul.allow_tf32=config.get('tf32',True)
    if world>1:dist.init_process_group('nccl' if device.startswith('cuda') else 'gloo')
    if not a.smoke and a.seed is None and 'seed' not in config:raise ValueError('full training requires an explicitly allocated --seed')
    seed=0 if a.smoke else int(a.seed if a.seed is not None else config['seed'])
    config=dict(config,seed=seed)
    random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed+rank)
    torch.set_num_threads(2)
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'result.json').exists() or any(a.output.glob('step*.pt*')):
        raise ValueError('fresh output required; resume into a new directory without overwriting or rotating previous checkpoints')
    identity=hashlib.sha256(a.manifest.read_bytes()+a.stats.read_bytes()).hexdigest()
    model=SharedWAMV5(stats,config).to(device)
    optimizer=torch.optim.AdamW(model.parameter_groups(config['geometry_lr'],config['shared_lr']),
        weight_decay=config['weight_decay'],fused=device.startswith('cuda'))
    first=1
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu')
        if (saved.get('schema')!=SCHEMA or saved.get('data_identity')!=identity
                or saved.get('config')!=config or saved.get('world_size')!=world):raise ValueError('v5 resume config/data/world-size mismatch')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);first=int(saved['step'])+1
        random.setstate(saved['random_state'][rank]['python']);np.random.set_state(saved['random_state'][rank]['numpy'])
        torch.set_rng_state(saved['random_state'][rank]['torch'])
        if device.startswith('cuda'):torch.cuda.set_rng_state(saved['random_state'][rank]['cuda'])
    wrapped=DDP(model,device_ids=[local] if device.startswith('cuda') else None) if world>1 else model
    kwargs=dict(dataset=dataset,batch_sampler=Steps(dataset,batch_size,seed,rank,first,last),collate_fn=collate,
                num_workers=a.workers,pin_memory=device.startswith('cuda'),
                generator=torch.Generator().manual_seed(seed*1009+rank))
    if a.workers:
        ipc=Path('/dev/shm')/f'ptv3-wam-v5-{os.getuid()}'
        ipc.mkdir(exist_ok=True,mode=0o700)
        os.environ['TMPDIR']=str(ipc);tempfile.tempdir=str(ipc)
        multiprocessing.current_process()._config['tempdir']=str(ipc)
        kwargs.update(multiprocessing_context='fork',persistent_workers=True,prefetch_factor=2)
    loader=DataLoader(**kwargs)
    dev=V5Windows(a.manifest,'dev') if rank==0 and not a.smoke else None
    if rank==0:
        write(a.output/'run-manifest.json',dict(schema=SCHEMA,work_version=SCHEMA,git_commit=a.git_commit,
            run_id=a.output.name,data_identity=identity,input_semantics=INPUT_SEMANTICS,label_semantics=LABEL_SEMANTICS,
            objective_semantics=OBJECTIVE_SEMANTICS,engineering_smoke=a.smoke,device=device,world_size=world,
            python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,steps=last,batch_size=batch_size,
            source_ids=dataset.meta['source_ids'],max_seconds=config['max_seconds'],min_free_gib=20,config=config))
    start=time.monotonic();times=[];events=[];seen=set();loss_value=None;step=first-1;completed_step=first-1;checkpoints_written=0
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    wrapped.train()
    for step,batch in zip(range(first,last+1),loader):
        stop=(a.output/'STOP').exists() or time.monotonic()-start>config['max_seconds'] or shutil.disk_usage(a.output).free<20*2**30
        flag=torch.tensor(int(stop),device=device)
        if world>1:dist.all_reduce(flag,op=dist.ReduceOp.MAX)
        if flag.item():break
        seen.update(map(int,batch['source_id'].tolist()))
        batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        multiplier=min(1.,step/max(1,config.get('warmup_updates',0))) if not a.smoke else 1.
        optimizer.param_groups[0]['lr']=config['geometry_lr']*multiplier
        optimizer.param_groups[1]['lr']=config['shared_lr']*multiplier
        ev=[torch.cuda.Event(enable_timing=True) for _ in range(4)] if device.startswith('cuda') else None
        if ev:ev[0].record()
        with torch.autocast('cuda' if device.startswith('cuda') else 'cpu',dtype=torch.bfloat16,enabled=device.startswith('cuda') and config.get('amp',True)):
            prediction=wrapped(batch);loss,terms=forecast_loss(prediction,batch,config,model.normalizer,world>1)
        if ev:ev[1].record()
        loss.backward()
        if ev:ev[2].record()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_grad'],error_if_nonfinite=True)
        optimizer.step()
        completed_step=step
        if ev:ev[3].record();events.append(ev)
        if len(events)>64:events.pop(0)
        if not torch.isfinite(loss.detach()):raise ValueError('nonfinite loss')
        loss_value=float(loss.detach());times.append(time.monotonic()-start)
        if rank==0 and (step<=3 or step%config['log_interval']==0):
            print(json.dumps(dict(step=step,loss=loss_value,grad_norm=float(norm),elapsed_s=times[-1],source_ids=sorted(seen))),flush=True)
        if not a.smoke and step%config['validation_interval']==0:
            if world>1:dist.barrier()
            if rank==0:write(a.output/f'dev-step{step:06d}.json',evaluate(model,dev,config,device))
            if world>1:dist.barrier()
        if not a.smoke and checkpoint_due(step,last,config):
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state() if device.startswith('cuda') else None)
            states=[None]*world
            if world>1:dist.all_gather_object(states,rng)
            else:states[0]=rng
            if rank==0:
                target=a.output/f'step{step:06d}.pt';pending=target.with_suffix('.pt.part')
                if target.exists() or pending.exists():raise ValueError('never overwrite existing v5 checkpoint or partial checkpoint')
                torch.save(dict(schema=SCHEMA,data_identity=identity,config=config,step=step,world_size=world,
                    model=model.state_dict(),optimizer=optimizer.state_dict(),random_state=states),pending)
                pending.replace(target)
                checkpoints_written+=1
                recent=sorted(a.output.glob('step*.pt'))
                protected=retained_steps([int(f.stem[4:]) for f in recent],config)
                for old in recent:
                    if int(old.stem[4:]) not in protected:old.unlink()
                if sum(f.stat().st_size for f in a.output.iterdir() if f.is_file())>config['artifact_budget_bytes']:
                    raise RuntimeError('owned v5 artifact budget exceeded')
    if device.startswith('cuda'):torch.cuda.synchronize()
    if rank==0:
        measured=events[min(2,len(events)-1):]
        timing={name:float(np.mean([ev[i].elapsed_time(ev[i+1]) for ev in measured])) for i,name in enumerate(('forward_loss_ms','backward_ms','optimizer_ms'))} if measured else {}
        result=dict(status='COMPLETED' if completed_step==last else 'STOPPED',engineering_smoke=a.smoke,optimizer_updates=len(times),
            last_step=completed_step,last_loss=loss_value,elapsed_s=time.monotonic()-start,source_ids_seen=sorted(seen),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.startswith('cuda') else None,
            timing=timing,checkpoint_written=bool(checkpoints_written))
        write(a.output/'result.json',result);print(json.dumps(result),flush=True)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
