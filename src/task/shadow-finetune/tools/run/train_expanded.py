"""Resume the existing Shadow policy on two GPUs with a fixed expanded budget."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))


def batch_seed(seed, rank, step):
    return seed + 1000003*rank + 10007*step


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--controls',type=Path,required=True)
    p.add_argument('--updates',type=int,default=20000)
    p.add_argument('--batch-size',type=int,default=256,help='Per-rank batch')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--max-seconds',type=int,default=43200)
    p.add_argument('--validation-interval',type=int,default=1000)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    from shadow_finetune import paths
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader,IterableDataset,get_worker_info
    from shadow_finetune.data import ShadowWindows,collate
    from shadow_finetune.model import ShadowPolicy,loss
    from ptv3_shared_wam.loading import to_device
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    if world!=2 or not 1<=a.batch_size<=256 or not 0<=a.workers<=4:
        raise ValueError('Exactly two ranks, batch<=256/rank, workers<=4/rank')
    physical=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[local]
    if subprocess.check_output(['nvidia-smi','-i',physical,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Assigned GPU already occupied')
    out=a.output.resolve()
    if out.exists() or paths.ROOT/'outputs/shadow-finetune' not in out.parents:
        raise ValueError('Fresh Shadow output required')
    torch.set_num_threads(2);torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    dist.init_process_group('nccl')
    controls=json.loads(a.controls.read_text())
    packet=torch.load(a.resume,map_location='cpu',weights_only=False)
    if (packet.get('schema')!=paths.SCHEMA or packet.get('engineering_only')
            or packet.get('sampler',{}).get('startup_fraction')!=.5
            or packet.get('comparison')!=controls or packet.get('step')!=1000):
        raise ValueError('Existing completed startup1000 checkpoint and original controls required')
    first=int(packet['step']);target=first+4 if a.smoke else a.updates
    if not first<target<=20000:
        raise ValueError('Bounded continuation through total20000')
    config=packet['config'];stats=packet['stats'];seed=214
    torch.manual_seed(seed+rank);np.random.seed(seed+rank);random.seed(seed+rank)
    model=ShadowPolicy(stats,config)
    model.load_state_dict(packet['model'],strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('Nonfinite initial weights')
    normalization={k:v.detach().cpu().clone() for k,v in model.actor.state_dict().items()
                   if k.endswith('_mean') or k.endswith('_scale')}
    model.cuda(local)
    optimizer=torch.optim.AdamW(model.groups(),weight_decay=.01,fused=True)
    optimizer.load_state_dict(packet['optimizer'])
    initial_optimizer_steps=sorted({int(v['step'].item()) for v in optimizer.state.values() if 'step' in v})
    if initial_optimizer_steps!=[first]:
        raise ValueError('Optimizer did not resume at1000')
    del packet
    data=controls['training_caches']['startup']
    dev=ShadowWindows(controls['validation_cache'],'dev')
    panels={'common_dev':controls['validation_ids'],'startup_dev':controls['startup_validation_ids']}
    manifest=dict(schema=paths.SCHEMA,status='RUNNING',engineering_only=a.smoke,
        experiment_id='P-20261015-shadow-expanded-training',run_id=out.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TASK,text=True).strip(),
        code_root=str(TASK.parents[2]),resume=str(a.resume.resolve()),resume_step=first,
        config=config,stats=stats,data=data,controls=str(a.controls.resolve()),
        target_step=target,world_size=world,batch_per_rank=a.batch_size,global_batch=a.batch_size*world,
        startup_fraction=.5,seed=seed,validation_panels=panels,
        initial_optimizer_steps=initial_optimizer_steps,
        optimizer_groups=[dict(name=g['name'],lr=g['lr']) for g in optimizer.param_groups],
        sampling='Per-step deterministic new streams after global batch/world-size change; original TRAIN-only normalizers retained',
        selection=dict(best='common DEV normalized action L1',forecast_best='common DEV hand+scene normalized MSE'),
        max_seconds=a.max_seconds,artifact_budget_bytes=32*10**9)
    def write(name,value):
        t=out/(name+'.part');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(out/name)
    if rank==0:
        out.mkdir(parents=True);write('manifest.json',manifest)
        np.save(out/'validation.npy',np.asarray(panels['common_dev']))
        np.savez(out/'initial-actor.npz',**{k:v.detach().cpu().numpy() for k,v in model.actor.state_dict().items()})
        write('resume-audit.json',dict(status='PASS',resume_step=first,optimizer_steps=initial_optimizer_steps,
            strict_model_load=True,normalizers_retained=True,groups=manifest['optimizer_groups']))
    dist.barrier()
    class Batches(IterableDataset):
        def __iter__(self):
            worker=get_worker_info();wid=0 if worker is None else worker.id
            count=1 if worker is None else worker.num_workers
            train=ShadowWindows(data,'train')
            for step in range(first+1+wid,target+1,count):
                rng=np.random.default_rng(batch_seed(seed,rank,step))
                ids=train.draw_startup(rng,a.batch_size,.5)
                b=collate([train[int(i)] for i in ids])
                b['_draw_counts']=torch.tensor([len(ids),sum(train.keys[i][1]<8 for i in ids),sum(train.keys[i][1]==0 for i in ids)])
                b['_step']=torch.tensor(step)
                yield b
    kwargs=dict(batch_size=None,num_workers=a.workers,pin_memory=True)
    if a.workers:kwargs.update(prefetch_factor=2)
    loader=iter(DataLoader(Batches(),**kwargs))
    wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local],gradient_as_bucket_view=True)
    started=time.monotonic();curve=[];best=float('inf');forecast_best=float('inf')
    counts=torch.zeros(3,device='cuda',dtype=torch.long)
    try:
        for step in range(first+1,target+1):
            raw=next(loader)
            if int(raw.pop('_step'))!=step:raise RuntimeError('Loader step ordering differs')
            counts+=raw.pop('_draw_counts').cuda(local)
            batch=to_device(raw,'cuda:'+str(local),non_blocking=True)
            wrapped.train();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=wrapped(batch);total,terms=loss(pred,batch,config,model)
            if not torch.isfinite(total):raise FloatingPointError('Nonfinite objective')
            total.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            if rank==0 and (step==first+1 or step%20==0 or step==target):
                torch.cuda.synchronize();elapsed=time.monotonic()-started
                row=dict(step=step,total_target=target,loss=float(total),terms={k:float(v) for k,v in terms.items()},
                    elapsed_s=elapsed,seconds_per_update=elapsed/(step-first),eta_s=elapsed/(step-first)*(target-step),
                    global_batch=a.batch_size*world,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                write('progress.json',row)
                with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
            if step%a.validation_interval==0 or step==target:
                dist.barrier();metrics={}
                if rank==0:
                    model.eval()
                    for panel,ids in panels.items():
                        sums={}
                        with torch.no_grad():
                            for off in range(0,len(ids),16):
                                selected=ids[off:off+16]
                                b=to_device(collate([dev[int(i)] for i in selected]),'cuda:'+str(local))
                                pred=model(b);value,terms=loss(pred,b,config,model)
                                row=dict(total=value,**terms,
                                    command_first_mae=(pred['action'][:,0]-b['action_target'][:,0]).abs().mean())
                                for k,v in row.items():sums[k]=sums.get(k,0.)+float(v)*len(selected)/len(ids)
                        metrics[panel]=sums
                    curve.append(dict(step=step,panels=metrics));write('curve.json',curve)
                    for k,v in normalization.items():
                        if not torch.equal(model.actor.state_dict()[k].cpu(),v):raise ValueError('Normalizer drift')
                rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
                rngs=[None]*world;dist.all_gather_object(rngs,rng)
                total_counts=counts.clone();dist.all_reduce(total_counts)
                if rank==0:
                    saved=dict(manifest,step=step,model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        optimizer=optimizer.state_dict(),rank_rng=rngs,sampling_counts=total_counts.cpu().tolist())
                    path=out/('step%d.pt'%step);part=path.with_suffix('.part');torch.save(saved,part);part.replace(path)
                    aliases=['latest.pt'];m=metrics['common_dev']
                    if m['action']<best:best=m['action'];aliases.append('best.pt')
                    if m['hand']+m['scene']<forecast_best:
                        forecast_best=m['hand']+m['scene'];aliases.append('forecast_best.pt')
                    for name in aliases:
                        link=out/(name+'.part');link.symlink_to(path.name);link.replace(out/name)
                    print(json.dumps(dict(checkpoint=path.name,metrics=metrics)),flush=True)
                dist.barrier()
            if time.monotonic()-started>a.max_seconds:raise TimeoutError('Expanded training deadline')
            if step%100==0:
                if shutil.disk_usage(out).free<20*2**30:raise RuntimeError('20GiB free floor')
                if rank==0 and sum(p.stat().st_size for p in out.glob('step*.pt'))>32*10**9:
                    raise RuntimeError('32GB checkpoint budget')
        if rank==0:
            manifest.update(status='COMPLETED',elapsed_s=time.monotonic()-started)
            write('manifest.json',manifest)
    except BaseException as error:
        if rank==0:manifest.update(status='FAILED',error=repr(error));write('manifest.json',manifest)
        raise
    finally:
        dist.destroy_process_group()


if __name__=='__main__':main()
