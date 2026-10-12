#!/usr/bin/env python3
"""One Goal-Point-WAM model trained with synchronous GPU DDP."""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import train_three_domain as base
from oakink_wm.multisource import MixedWindows


def rank_stream(indices, microbatch, rank, world):
    """Partition sample occurrences, preserving each global optimizer batch."""
    if not 0 <= rank < world or len(indices) % (world * microbatch):
        raise ValueError('invalid global stream/rank')
    return indices.reshape(-1, world, microbatch)[:, rank, :].reshape(-1)


def aggregate_evaluation(parts, names):
    merged = {}
    for name in names:
        prefix = 'source/' + name + '/'
        counts = {k: sum(p[prefix+'counts'][k] for p in parts)
                  for k in ('trajectory_points', 'objects', 'moving_objects', 'static_objects')}
        merged[prefix+'counts'] = counts
        for metric, count in [('trajectory_epe_m','trajectory_points'),
                              ('world_point_epe_m_h24','objects'),
                              ('moving_world_point_epe_m_h24','moving_objects'),
                              ('static_world_point_epe_m_h24','static_objects')]:
            total = sum(p[prefix+metric] * p[prefix+'counts'][count] for p in parts
                        if p[prefix+'counts'][count])
            merged[prefix+metric] = total / counts[count] if counts[count] else None
    for metric in ('trajectory_epe_m', 'world_point_epe_m_h24'):
        merged['macro/'+metric] = float(np.mean([merged['source/'+n+'/'+metric] for n in names
                                                if merged['source/'+n+'/'+metric] is not None]))
    return merged


def save(path, model, optimizer, step, config, identity, best, rank, world):
    rng = dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
               numpy_rng=np.random.get_state(), python_rng=random.getstate())
    states = [None] * world
    dist.all_gather_object(states, rng)
    if rank == 0:
        state = dict(checkpoint_kind='goal-point-wam.single.v1', model=model.state_dict(),
                     optimizer=optimizer.state_dict(), step=step, config=config,
                     identity=identity, best=best, distributed_rng=states, **states[0])
        part = path.with_suffix('.pt.part')
        torch.save(state, part)
        part.replace(path)
    dist.barrier()


def link_checkpoint(source, destination):
    part = destination.with_suffix('.pt.link')
    if part.exists():
        raise FileExistsError(part)
    os.link(source, part)
    part.replace(destination)


def run(args):
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    config = json.loads(args.config.read_text())
    if config.get('schema') != 'goal-point-wam.three-domain-no-video.v1':
        raise ValueError('unexpected model/data configuration')
    if world != config['world_size'] or not 1 <= world <= 4 or config['accumulation'] != 1:
        raise ValueError('fixed world-size1..4 and per-rank accumulation1 required')
    if args.smoke:
        config.update(seed=278, split_seed=279, updates=4,
                      microbatch=config['microbatch'] if args.smoke_full_batch else 4,
                      workers=config['workers'] if args.smoke_full_batch else 0,
                      validation_samples=12, validation_interval=4, checkpoint_interval=4)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    dist.init_process_group('nccl', timeout=timedelta(seconds=180))
    log = None
    step = 0
    output = args.output.resolve()
    try:
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            if (output/'input_manifest.json').exists() and not args.resume:
                raise FileExistsError('fresh DDP run requires a new output directory')
            base.atomic_json(output/'config.json', config)
        dist.barrier()
        seed = int(config['seed'])
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        train_base, val_base = MixedWindows(args.data,'train'), MixedWindows(args.data,'val')
        names = [s['name'] for s in train_base.meta['sources']]
        stats = json.loads(args.stats.read_text())
        if (names != list(base.MAIN_SOURCE_NAMES) or stats['sources'] != names or
                stats['split'] != 'train' or train_base.meta['supervision'] != 'measured_dynamic_hand_main' or
                stats['normalization_source_manifest_sha256'] != train_base.meta['normalization_source_manifest_sha256']):
            raise ValueError('frozen train normalization/source contract mismatch')
        train, val = [base.GoalWindows(d, goal_type=config['goal_type'], goal_points=config['goal_points'])
                      for d in (train_base,val_base)]
        identity = dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                        data_manifest_sha256=base.digest(args.data/'processed/manifest.json'),
                        stats_sha256=base.digest(args.stats), config_sha256=base.digest(args.config),
                        script_sha256=base.digest(Path(__file__)), single_trainer_sha256=base.digest(Path(base.__file__)),
                        model_sha256=base.digest(base.TASK/'src/goal_point_wam/model.py'),
                        sources=names, no_video=True, distributed=True, world_size=world,
                        global_batch=world*config['microbatch'],
                        objective='DDP mean of per-rank locally masked WAM losses; not a monolithic global-mask mean')
        model = base.GoalPointWAM(stats, config)
        identity['pretrained'] = base.initialize_pw(model,args.init_pw,config,identity['stats_sha256'])
        identity['initial_parameter_sha256'] = base.parameter_hash(model)
        identity['model_parameters'] = sum(p.numel() for p in model.parameters())
        hashes = [None]*world
        dist.all_gather_object(hashes,identity['initial_parameter_sha256'])
        if len(set(hashes)) != 1:
            raise ValueError('initial parameters differ across ranks')
        model = model.to(device)
        distributed = DistributedDataParallel(model,device_ids=[local_rank],broadcast_buffers=False,
                                               find_unused_parameters=True,gradient_as_bucket_view=True)
        optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        best = float('inf')
        resume_rng = None
        global_indices = base.mixed_indices(train_base,config['updates']*world*config['microbatch'],config['split_seed'])
        identity['global_train_occurrences_sha256'] = hashlib.sha256(global_indices.tobytes()).hexdigest()
        if args.smoke:
            validation = base.mixed_indices(val_base,12,config['validation_seed'],equal_sources=True)
        else:
            panel = json.loads((args.validation_indices.parent/'panel.json').read_text())
            if (panel['data_manifest_sha256'] != identity['data_manifest_sha256'] or
                    panel['stats_sha256'] != identity['stats_sha256'] or
                    panel['validation_indices_sha256'] != base.digest(args.validation_indices)):
                raise ValueError('frozen panel identity drift')
            validation = np.load(args.validation_indices)
            if (validation.ndim != 1 or validation.dtype.kind not in 'iu' or
                    len(validation) != config['validation_samples'] or len(np.unique(validation)) != len(validation) or
                    validation.min() < 0 or validation.max() >= len(val_base)):
                raise ValueError('invalid frozen panel')
            source_ids = np.searchsorted(val_base.offsets,validation,side='right')-1
            if not np.all(np.bincount(source_ids,minlength=len(names))==len(validation)//len(names)):
                raise ValueError('unbalanced source panel')
            identity['frozen_validation_panel_sha256'] = base.digest(args.validation_indices.parent/'panel.json')
        # Every worker has an independent stochastic serialization stream after equal initialization.
        random.seed(seed+rank); np.random.seed(seed+rank); torch.manual_seed(seed+rank)
        torch.cuda.manual_seed(seed+rank)
        if args.resume:
            state = torch.load(args.resume,map_location=device,weights_only=False)
            for key in ('data_manifest_sha256','stats_sha256','model_sha256','script_sha256',
                        'single_trainer_sha256','world_size','frozen_validation_panel_sha256',
                        'global_train_occurrences_sha256'):
                if state['identity'].get(key) != identity.get(key):
                    raise ValueError('DDP resume identity drift: '+key)
            if state['config'] != config:
                raise ValueError('DDP resume config drift')
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            step,best = state['step'],state['best']; identity=state['identity']
            resume_rng = state['distributed_rng'][rank]
            del state
        local_indices = rank_stream(global_indices,config['microbatch'],rank,world)[step*config['microbatch']:]
        ranks = [None]*world
        visible = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        dist.all_gather_object(ranks,dict(rank=rank,pid=os.getpid(),physical_gpu=int(visible[local_rank])))
        identity['ranks'] = ranks
        if rank == 0:
            np.save(output/'train_indices.npy',global_indices)
            np.save(output/'validation_balanced.npy',validation)
            identity['train_indices_sha256']=base.digest(output/'train_indices.npy')
            identity['validation_indices_sha256']=base.digest(output/'validation_balanced.npy')
        identities = [identity if rank == 0 else None]
        dist.broadcast_object_list(identities,src=0); identity=identities[0]
        if rank == 0:
            base.atomic_json(output/'input_manifest.json',identity)
        np.save(output/('rank'+str(rank)+'_train_indices.npy'),local_indices)
        batches = iter(base.loader(train,local_indices,config['microbatch'],config['workers']))
        # DataLoader iterator creation consumes a CPU base seed; restore after
        # it so resumed PTv3 CPU serialization and CUDA DropPath do not drift.
        if resume_rng is not None:
            torch.set_rng_state(resume_rng['torch_rng'].cpu())
            torch.cuda.set_rng_state(resume_rng['cuda_rng'].cpu())
            np.random.set_state(resume_rng['numpy_rng']); random.setstate(resume_rng['python_rng'])
        stopped = [False]
        signal.signal(signal.SIGUSR1,lambda *_: stopped.__setitem__(0,True))
        started=time.time(); telemetry={}; measured=None
        torch.cuda.reset_peak_memory_stats()
        log=(output/('rank'+str(rank)+'.jsonl')).open('a')
        while step < config['updates']:
            if step % 100 == 0:
                stop=torch.tensor(int(stopped[0] or time.time()-started>=config['group_seconds']),device=device)
                dist.all_reduce(stop,op=dist.ReduceOp.MAX)
                if stop.item():
                    break
            tick=time.monotonic()
            scale = ((step+1)/config['warmup_updates'] if step<config['warmup_updates'] else
                     .1+.9*.5*(1+math.cos(math.pi*(step-config['warmup_updates'])/
                                         max(1,config['updates']-config['warmup_updates']))))
            for group in optimizer.param_groups:
                group['lr']=config['learning_rate']*scale
            batch=base.move_batch(next(batches),device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=config['amp']):
                prediction=distributed(batch)
            loss,terms=model.loss(prediction,batch)
            finite=torch.isfinite(loss).to(torch.int32)
            dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError('nonfinite rank loss')
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_grad'],error_if_nonfinite=True)
            optimizer.step(); step+=1
            if step==1 or step%config['gpu_snapshot_interval']==0:
                telemetry=dict(base.gpu_snapshot(device),sampled_at_step=step,
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved())
            row=dict(step=step,rank=rank,loss=float(loss.detach()),gradient_norm=float(grad),
                     seconds=time.monotonic()-tick,elapsed_seconds=time.time()-started,
                     learning_rate=optimizer.param_groups[0]['lr'],gpu=telemetry)
            row.update({k:float(v) for k,v in terms.items()})
            log.write(json.dumps(row)+'\n'); log.flush()
            if step==1 or step%10==0:
                base.atomic_json(output/('rank'+str(rank)+'_progress.json'),dict(status='RUNNING',**row))
                if rank==0:
                    base.atomic_json(output/'progress.json',dict(status='RUNNING',world_size=world,**row))
            if step==1 or step%100==0:
                print(json.dumps(row),flush=True)
            if step%config['validation_interval']==0 or step==config['updates']:
                local=base.evaluate(model,val,validation[rank::world],config['microbatch'],config['workers'],device)
                parts=[None]*world; dist.all_gather_object(parts,local)
                measured=aggregate_evaluation(parts,names)
                score=measured['macro/world_point_epe_m_h24']+measured['macro/trajectory_epe_m']
                improved=score<best; best=min(best,score)
                save(output/'latest.pt',model,optimizer,step,config,identity,best,rank,world)
                if rank==0:
                    base.atomic_json(output/'validation_latest.json',dict(step=step,metrics=measured))
                    with (output/'validation.jsonl').open('a') as handle:
                        handle.write(json.dumps(dict(step=step,metrics=measured))+'\n')
                    if improved:
                        link_checkpoint(output/'latest.pt',output/'best.pt')
                dist.barrier()
            elif step%config['checkpoint_interval']==0:
                save(output/'latest.pt',model,optimizer,step,config,identity,best,rank,world)
        if measured is None or step%config['validation_interval']:
            local=base.evaluate(model,val,validation[rank::world],config['microbatch'],config['workers'],device)
            parts=[None]*world; dist.all_gather_object(parts,local); measured=aggregate_evaluation(parts,names)
        save(output/'latest.pt',model,optimizer,step,config,identity,best,rank,world)
        hashes=[None]*world; dist.all_gather_object(hashes,base.parameter_hash(model))
        if len(set(hashes))!=1:
            raise ValueError('final rank parameters diverged')
        if rank==0:
            link_checkpoint(output/'latest.pt',output/'final.pt')
            result=dict(status='COMPLETED' if step==config['updates'] else 'BUDGET_STOP',step=step,
                        elapsed_seconds=time.time()-started,world_size=world,global_batch=world*config['microbatch'],
                        validation=measured,engineering_only=args.smoke,rank_parameter_sha256=hashes,
                        gpu=base.gpu_snapshot(device))
            base.atomic_json(output/'result.json',result); base.atomic_json(output/'progress.json',result)
            print(json.dumps(result),flush=True)
        dist.barrier()
    except BaseException as error:
        if output.exists():
            base.atomic_json(output/('rank'+str(rank)+'_error.json'),dict(step=step,error=repr(error)))
        raise
    finally:
        if log:
            log.close()
        dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','stats','init-pw','config','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--validation-indices',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-full-batch',action='store_true',
                        help='Use configured per-rank batch/workers for the four-update engineering smoke')
    args=parser.parse_args()
    if args.smoke_full_batch and not args.smoke:
        parser.error('--smoke-full-batch requires --smoke')
    run(args)
