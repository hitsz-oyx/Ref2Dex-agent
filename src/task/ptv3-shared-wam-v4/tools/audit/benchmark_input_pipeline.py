"""Bounded real-reader equivalence and full-model input throughput inspection.

CPU mode checks every returned field against the original query/reader path.
GPU mode compares the same sampled optimizer steps at fixed model/RNG state;
it writes no checkpoints and never selects a model from benchmark updates.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import time

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))
from ptv3_shared_wam_v4 import paths
from ptv3_shared_wam_v4.data import NineWindows, loader
from pose_mesh_data import v4_interaction as field
from pose_mesh_data.v4_data import V4NativeWindows
from pose_mesh_data.v4_windows import v4_pose_mesh_sample
import numpy as np
import torch

FAST_QUERY = field._query_voxel_counts
FAST_NATIVE = V4NativeWindows.__getitem__


def legacy_native(self, index):
    window = self.base[index]
    record = window.provenance
    row = self.base.meta['sequences'][record.row]
    sample = v4_pose_mesh_sample(self.base._sequence(record.row),
        [self.base.geometry[k] for k in row['objects']], record.tick, record.target_slot,
        record.source_id, [record.source_id, record.row, record.tick], record.profile, self.policy)
    sample['provenance'] = asdict(record)
    return sample


def implementation(mode):
    field._query_voxel_counts = field._query_voxel_counts_reference if mode == 'reference' else FAST_QUERY
    V4NativeWindows.__getitem__ = FAST_NATIVE if mode == 'optimized' else legacy_native


def dataset_from_recipe(recipe):
    ds = NineWindows(recipe['main3_root'], recipe['native_root'], 'train', recipe)
    for source in ds.main.base.sources:
        source.sequence = lru_cache(maxsize=16)(source.sequence)
    return ds


def digest_sample(sample):
    digest = hashlib.sha256()
    for key, value in sorted(sample.items()):
        digest.update(key.encode())
        if isinstance(value, dict):
            digest.update(json.dumps(value, sort_keys=True).encode())
        else:
            value = np.asarray(value)
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def cpu_check(recipe, repeats):
    ds = dataset_from_recipe(recipe)
    panel = ds.panel(8, 172)
    indices = [i for ids in panel.values() for i in ids] + ds.draw(128, 170, 240, 1)
    hashes, reports = None, {}
    for mode in ('reference', 'lookup', 'optimized'):
        implementation(mode)
        times = []
        for repeat in range(repeats):
            started = time.monotonic()
            samples = [ds[i][0] for i in indices]
            times.append(time.monotonic()-started)
            actual = [digest_sample(s) for s in samples]
            if hashes is None:
                hashes = actual
            if actual != hashes:
                mismatches = [indices[i] for i,(a,b) in enumerate(zip(actual,hashes)) if a != b]
                raise AssertionError('real-data field mismatch: '+repr(mismatches))
        reports[mode] = dict(seconds=times, median_seconds=float(np.median(times)),
            samples_per_second=len(indices)/float(np.median(times)))
        print(json.dumps(dict(mode=mode,**reports[mode])),flush=True)
    implementation('optimized')
    return dict(samples=len(indices), source_counts=dict(Counter(str(ds[i][1]['source_id']) for i in indices)),
        all_fields_bitwise_equal=True, input_sha256=hashlib.sha256(''.join(hashes).encode()).hexdigest(),
        lookup_speedup=reports['reference']['median_seconds']/reports['lookup']['median_seconds'],
        total_speedup=reports['reference']['median_seconds']/reports['optimized']['median_seconds'],
        implementations=reports)


def gpu_check(args, recipe):
    import torch.distributed as dist
    from ptv3_shared_wam_v4.model import SharedWAMV4
    from ptv3_shared_wam_v4.checkpoint import load_checkpoint, restore_optimizer, validate_resume_contract
    from ptv3_shared_wam_v4.objectives import forecast_loss
    from ptv3_shared_wam_v4.lr_transition import apply_learning_rates
    from ptv3_shared_wam_v4.performance import StepTimings
    from ptv3_shared_wam_v4.data import DevInputCache
    from ptv3_shared_wam_v4.training import evaluate
    from ptv3_shared_wam.loading import to_device
    world,rank,local = [int(os.environ.get(k,v)) for k,v in
                       [('WORLD_SIZE','1'),('RANK','0'),('LOCAL_RANK','0')]]
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl')
    parent = load_checkpoint(args.checkpoint)
    config, stats, first = parent['config'], parent['stats'], parent['step']
    if stats['recipe'] != recipe:
        raise ValueError('benchmark must use identical checkpoint recipe')
    validate_resume_contract(parent,config,stats,world)
    torch.backends.cuda.matmul.allow_tf32 = config['tf32']
    torch.backends.cudnn.allow_tf32 = config['tf32']
    ds = dataset_from_recipe(recipe)
    model = SharedWAMV4(stats,config).cuda(local)
    optimizer = torch.optim.AdamW(model.parameter_groups(config['geometry_lr'],config['shared_lr']),
        weight_decay=config['weight_decay'],fused=True)
    wrapped = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local],
        gradient_as_bucket_view=True) if world > 1 else model
    reports=[]
    ids_by_rank=None
    for mode in args.modes.split(','):
        implementation(mode)
        for workers in map(int,args.workers.split(',')):
            model.load_state_dict(parent['model'],strict=True)
            restore_optimizer(optimizer,parent['optimizer'])
            torch.set_rng_state(parent['rank_rng'][rank]['torch'])
            torch.cuda.set_rng_state(parent['rank_rng'][rank]['cuda'])
            model.train()
            batches=iter(loader(ds,config,rank,first+1,first+args.steps,workers))
            timer=StepTimings(args.steps)
            ids=hashlib.sha256(); losses=[]
            for offset in range(args.steps):
                if offset == args.warmup:
                    torch.cuda.synchronize()
                    if world > 1: dist.barrier()
                    started=time.monotonic()
                    timer=StepTimings(args.steps)
                timer.before_batch()
                bound=next(batches)
                timer.after_batch()
                ids.update(json.dumps(bound['provenance'],sort_keys=True).encode())
                batch=to_device(bound['model_inputs'],torch.device('cuda',local),non_blocking=True)
                timer.mark(1)
                apply_learning_rates(optimizer,config,first+offset+1)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['amp']):
                    prediction=wrapped(batch)
                    loss,_=forecast_loss(prediction,batch,config,model.normalizer,distributed=world>1)
                timer.mark(2)
                if not torch.isfinite(loss): raise ValueError('nonfinite benchmark loss')
                loss.backward();timer.mark(3)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_grad'])
                if not torch.isfinite(norm): raise ValueError('nonfinite benchmark gradient')
                optimizer.step();timer.mark(4)
                losses.append(float(loss.detach()))
            torch.cuda.synchronize()
            elapsed=time.monotonic()-started
            report=dict(mode=mode,workers=workers,rank=rank,seconds=elapsed,
                measured_steps=args.steps-args.warmup,updates_per_second=(args.steps-args.warmup)/elapsed,
                timings=timer.summary(),sample_order_sha256=ids.hexdigest(),losses=losses)
            gathered=[None]*world
            if world>1: dist.all_gather_object(gathered,report)
            else:gathered=[report]
            signature=[r['sample_order_sha256'] for r in gathered]
            if ids_by_rank is None: ids_by_rank=signature
            if signature != ids_by_rank:raise AssertionError('worker/mode changed sampled order')
            if rank==0:
                reports.append(gathered)
                print(json.dumps(dict(mode=mode,workers=workers,rank0_rate=report['updates_per_second'],
                                      input_order_equal=True)),flush=True)
            del batches,batch,prediction,loss
            torch.cuda.empty_cache()
            if world>1:dist.barrier()
    dev_report=None
    if rank==0:
        implementation('optimized')
        dev=NineWindows(recipe['main3_root'],recipe['native_root'],'dev',recipe)
        panel=dev.panel(config['validation_samples']//len(dev.counts),config['validation_seed'])
        cache=DevInputCache(dev,panel,config.get('validation_microbatch_size',8))
        scores=[];times=[]
        for input_cache in (None,cache):
            torch.cuda.synchronize();started=time.monotonic()
            result=evaluate(model,dev,panel,config,torch.device('cuda',local),input_cache=input_cache)
            torch.cuda.synchronize();times.append(time.monotonic()-started)
            scores.append(result)
        if scores[0] != scores[1]:
            raise AssertionError('real-model cached DEV differs from streaming DEV')
        dev_report=dict(all_metrics_equal=True,objective=scores[0]['objective'],
            streaming_seconds=times[0],cached_seconds=times[1],
            cache_build_seconds=cache.build_seconds,cache_bytes=cache.bytes)
    if world>1:
        dist.barrier()
        dist.destroy_process_group()
    return dict(config=config,checkpoint=str(args.checkpoint.resolve()),parent_step=first,
        all_sample_orders_equal=True,runs=reports,dev_cache=dev_report) if rank==0 else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--workers',default='2,4')
    p.add_argument('--modes',default='reference,optimized')
    args=p.parse_args()
    if not 1<=args.repeats<=3 or not 1<=args.warmup<args.steps<=100:
        p.error('bounded repeats1--3 and1<=warmup<steps<=100 required')
    if (any(x not in ('reference','lookup','optimized') for x in args.modes.split(','))
            or any(x not in ('0','1','2','4') for x in args.workers.split(','))):
        p.error('known implementation and bounded workers required')
    if args.output.exists(): raise FileExistsError('fresh benchmark output required')
    torch.set_num_threads(1 if args.checkpoint is None else 2)
    recipe=json.loads((args.recipe/'recipe.json').read_text())
    result=gpu_check(args,recipe) if args.checkpoint else cpu_check(recipe,args.repeats)
    if result is not None:
        result['environment']=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
            cuda=torch.version.cuda,code_commit=__import__('subprocess').check_output(
                ['git','rev-parse','HEAD'],cwd=paths.ROOT,text=True).strip())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:json.dump(result,stream,indent=2)


if __name__=='__main__':main()
