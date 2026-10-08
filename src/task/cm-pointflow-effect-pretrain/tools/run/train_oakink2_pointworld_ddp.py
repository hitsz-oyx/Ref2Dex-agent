#!/usr/bin/env python3
"""torchrun DDP entry preserving the frozen temporal trainer's microbatch objective."""
import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))
from oakink_wm.data import Windows, balanced_indices
from oakink_wm.pointworld_temporal import model_from_config, VENDOR
from oakink_wm.distributed import (local_accumulation, rank_indices, sync_context,
                                   capture_rng, restore_rng, gather_rng, any_rank)

spec = importlib.util.spec_from_file_location('temporal_reference_trainer', TASK/'tools/run/train_oakink2_pointworld_temporal.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def parameter_hash(model):
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def source_identity():
    sources = base.implementation_sources()
    for p in (Path(__file__).resolve(), TASK/'src/oakink_wm/distributed.py',
              TASK/'src/oakink_wm/pointworld_performance.py', TASK/'src/oakink_wm/continuation.py',
              TASK/'src/oakink_wm/multisource.py', TASK/'src/oakink_wm/loss_normalization.py'):
        sources[str(p.relative_to(TASK))] = base.digest(p)
    return sources


def sources_drifted(identity, config_path, stats_path):
    try:
        return (any(base.digest(TASK/key)!=h for key,h in identity['implementation_sources'].items()) or
                any(base.digest(VENDOR/key)!=h for key,h in identity['vendor_sources'].items()) or
                base.digest(config_path)!=identity['input_config_sha256'] or
                base.digest(stats_path)!=identity['stats_sha256'] or
                (identity.get('loss_stats_path') is not None and
                 base.digest(Path(identity['loss_stats_path']))!=identity['loss_stats_sha256']))
    except FileNotFoundError:
        return True


def save_checkpoint(path, model, optimizer, step, config, best, identity, rank, world_size):
    rngs = gather_rng(world_size)
    if rank == 0:
        state = dict(checkpoint_kind='pointworld-temporal.ddp.v1', world_size=world_size,
                     model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                     config=config, dataset_hash=identity['dataset_hash'], best=best,
                     identity=identity, rank_rngs=rngs,
                     torch_rng=rngs[0]['torch'], cuda_rng=rngs[0]['cuda'],
                     numpy_rng=rngs[0]['numpy'], python_rng=rngs[0]['python'])
        part = path.with_suffix('.pt.part')
        torch.save(state, part)
        part.replace(path)
    dist.barrier()


def load_checkpoint(state, model, optimizer, config, identity, rank, world_size, import_single=False):
    same = (state['config']==config and state['dataset_hash']==identity['dataset_hash'] and
            state['identity']['arm']==identity['arm'] and
            state['identity']['stats_sha256']==identity['stats_sha256'] and
            state['identity'].get('loss_stats_sha256')==identity.get('loss_stats_sha256') and
            state['identity']['vendor_sources']==identity['vendor_sources'] and
            state['identity'].get('performance_backend', 'reference')==identity.get('performance_backend', 'reference'))
    expected = base.implementation_sources() if import_single else identity['implementation_sources']
    if not same or state['identity'].get('implementation_sources') != expected:
        raise ValueError('checkpoint input/config/implementation mismatch')
    if import_single:
        if state.get('checkpoint_kind') is not None:
            raise ValueError('import requires a single-GPU temporal checkpoint')
    elif state.get('checkpoint_kind')!='pointworld-temporal.ddp.v1' or state.get('world_size')!=world_size:
        raise ValueError('DDP resume requires the same world size')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    if import_single:
        if rank == 0:
            restore_rng(dict(torch=state['torch_rng'], cuda=state['cuda_rng'],
                             numpy=state['numpy_rng'], python=state['python_rng']))
        else:
            seed = config['seed'] + 100003*rank + state['step']
            torch.manual_seed(seed); torch.cuda.manual_seed(seed)
            np.random.seed(seed); random.seed(seed)
    else:
        restore_rng(state['rank_rngs'][rank])
    return state['step'], state['best']


def initialize_weights(state, model, config, identity):
    """New recipe: import model only, preserving the data/architecture contract."""
    recipe_keys = {'seed', 'microbatch', 'accumulation', 'updates', 'learning_rate',
                   'weight_decay', 'warmup_updates', 'clip_grad', 'validation_samples',
                   'validation_interval', 'validation_seed', 'natural_validation_seed',
                   'checkpoint_interval', 'workers', 'group_seconds', 'validation_microbatch'}
    previous = state['identity']
    if (state['dataset_hash'] != identity['dataset_hash'] or
            previous['arm'] != identity['arm'] or
            previous['stats_sha256'] != identity['stats_sha256'] or
            previous['vendor_sources'] != identity['vendor_sources'] or
            previous.get('implementation_sources') != base.implementation_sources()):
        raise ValueError('initialization checkpoint data/arm/stats/source mismatch')
    for key in set(config) | set(state['config']):
        if key not in recipe_keys and config.get(key) != state['config'].get(key):
            raise ValueError('initialization changes model/data semantics: '+key)
    model.load_state_dict(state['model'], strict=True)
    return dict(parent_step=state['step'], weights_only=True,
                optimizer_reset=True, schedule_reset=True, draw_reset=True,
                changed_recipe={key: dict(previous=state['config'].get(key), current=config.get(key))
                                for key in recipe_keys if state['config'].get(key)!=config.get(key)})


def run(args):
    if 'RANK' not in os.environ:
        raise RuntimeError('launch this entry with torchrun, including world-size1')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if not 1 <= world <= 4:
        raise ValueError('campaign allows at most4 distributed GPU ranks')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=180))
    stop = [False]
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: stop.__setitem__(0, True))
    trainlog = None
    try:
        config = json.loads(args.config.read_text())
        if args.steps is not None:
            config['updates'] = args.steps
        if args.smoke:
            config.update(microbatch=2, accumulation=2*world if world>1 else 1, workers=0,
                          validation_samples=12, validation_interval=3, checkpoint_interval=3)
        accumulation = local_accumulation(config['accumulation'], world)
        effective = config['microbatch']*config['accumulation']
        if args.stop_after is not None and not (args.smoke or args.engineering):
            raise ValueError('--stop-after is an engineering-smoke control')
        out = args.output.resolve()
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            if (out/'input_manifest.json').exists() and not args.resume:
                raise FileExistsError('use a fresh output directory or explicit DDP resume')
            if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:
                raise ValueError('existing run config drift')
            base.atomic_json(out/'config.json', config)
        dist.barrier()
        seed = config['seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
        base.configure_numerics()
        if args.fused_hilbert:
            from oakink_wm.pointworld_performance import install_fused_hilbert
            install_fused_hilbert()
        if args.mixed_data:
            from oakink_wm.multisource import MixedWindows, mixed_indices, validate_pretrained
            train, val = MixedWindows(args.data, 'train'), MixedWindows(args.data, 'val')
            draw = mixed_indices
        else:
            train, val = Windows(args.data, 'train'), Windows(args.data, 'val')
            draw = balanced_indices
        dataset_hash = base.digest(args.data/'processed/manifest.json')
        stats = json.loads(args.stats.read_text())
        stats_dataset_hash = train.meta['normalization_source_manifest_sha256'] if args.mixed_data else dataset_hash
        if (stats['split']!='train' or stats['input_manifest_sha256']!=stats_dataset_hash
                or (args.mixed_data and base.digest(args.stats) != train.meta['normalization_stats_sha256'])
                or (args.mixed_data and train.meta.get('normalization_mode')
                    and stats.get('normalization_mode') != train.meta['normalization_mode'])
                or (args.mixed_data and config.get('normalization_mode')
                    and config.get('normalization_mode') != train.meta.get('normalization_mode'))):
            raise ValueError('normalization identity mismatch')
        model = model_from_config(stats, config).to(device)
        loss_scales = None
        if args.loss_stats:
            if not args.mixed_data: raise ValueError('separate physical loss statistics require mixed main dynamics')
            from oakink_wm.loss_normalization import validate_loss_statistics, physical_loss
            loss_scales = validate_loss_statistics(json.loads(args.loss_stats.read_text()), dataset_hash,
                                                  [d['name'] for d in train.meta['sources']])
            loss_scales = {key: value.to(device) for key, value in loss_scales.items()}
        if args.mixed_data and train.meta.get('supervision') == 'measured_dynamic_hand_main' and loss_scales is None:
            raise ValueError('three-source main dynamics requires explicit train-only --loss-stats')
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        initial_hash = parameter_hash(model)
        identity = dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            dataset_hash=dataset_hash, config_hash=base.digest(out/'config.json'),
            input_config_sha256=base.digest(args.config), arm=args.arm, smoke=args.smoke,
            parameters=sum(p.numel() for p in model.parameters()), world_size=world,
            microbatch_per_rank=config['microbatch'], accumulation_per_rank=accumulation,
            global_effective_batch=effective, objective='mean of per-microbatch normalized losses; larger microbatches change reference objective',
            performance_backend='integer-exact-triton-hilbert' if args.fused_hilbert else 'reference',
            engineering_only=args.smoke or args.engineering or args.benchmark,
            script_sha256=base.digest(Path(__file__)), model_sha256=base.digest(TASK/'src/oakink_wm/pointworld_temporal.py'),
            stats_sha256=base.digest(args.stats), stats=stats, implementation_sources=source_identity(),
            loss_stats_sha256=base.digest(args.loss_stats) if args.loss_stats else None,
            loss_stats_path=str(args.loss_stats.resolve()) if args.loss_stats else None,
            loss_normalization='main-source train-only residual scales' if args.loss_stats else 'forward statistics reused for loss',
            initial_parameter_sha256=initial_hash,
            pointworld_commit=subprocess.check_output(['git','-C',str(VENDOR),'rev-parse','HEAD'],text=True).strip(),
            vendor_sources={str(f.relative_to(VENDOR)):base.digest(f) for f in (VENDOR/'ptv3').rglob('*') if f.suffix in ('.py','.yaml')})
        if args.mixed_data:
            identity.update(mixed_data=True, source_manifest=train.meta,
                            normalization_source_manifest_sha256=stats_dataset_hash)
        gathered = [None]*world
        dist.all_gather_object(gathered, dict(rank=rank, pid=os.getpid(), local_rank=local_rank,
             visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES',''), initial_hash=initial_hash,
             dataset_hash=dataset_hash, config_hash=identity['config_hash'],
             stats_hash=identity['stats_sha256'], sources=identity['implementation_sources'],
             loss_stats_hash=identity['loss_stats_sha256'],
             vendor_sources=identity['vendor_sources']))
        for k in ('initial_hash','dataset_hash','config_hash','stats_hash','loss_stats_hash','sources','vendor_sources'):
            if any(x[k]!=gathered[0][k] for x in gathered):
                raise ValueError('rank identity mismatch: '+k)
        identity['ranks'] = gathered
        if rank > 0:
            torch.manual_seed(seed+rank); torch.cuda.manual_seed(seed+rank)
            np.random.seed(seed+rank); random.seed(seed+rank)
        step, best = 0, float('inf')
        if args.init_weights:
            state = torch.load(args.init_weights, map_location='cpu', weights_only=False, mmap=True)
            identity['initialization'] = (validate_pretrained(state, model, config, identity, base.implementation_sources())
                if args.mixed_data else initialize_weights(state, model, config, identity))
            identity['parent_checkpoint_sha256'] = base.digest(args.init_weights)
            identity['parent_checkpoint_path'] = str(args.init_weights.resolve())
            del state
        if args.continue_from:
            from oakink_wm.continuation import continue_optimizer
            state = torch.load(args.continue_from, map_location='cpu', weights_only=False, mmap=True)
            identity['initialization'] = continue_optimizer(state, model, optimizer, config, identity, rank, world)
            identity['parent_checkpoint_sha256'] = base.digest(args.continue_from)
            identity['parent_checkpoint_path'] = str(args.continue_from.resolve())
            del state
        if args.random_init:
            identity['initialization'] = dict(random_init=True, optimizer_reset=True,
                                               schedule_reset=True, draw_reset=True,
                                               normalization='shared multisource train-union forward statistics')
        checkpoint = args.resume or args.import_single_checkpoint
        if checkpoint:
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            step, best = load_checkpoint(state, model, optimizer, config, identity, rank, world,
                                         import_single=bool(args.import_single_checkpoint))
            identity['parent_checkpoint_sha256'] = base.digest(checkpoint)
            identity['imported_single_checkpoint'] = bool(args.import_single_checkpoint)
            identity['exact_rng_continuation'] = not args.import_single_checkpoint or world == 1
            del state
        identity['random_initial_parameter_sha256'] = initial_hash
        identity['initial_parameter_sha256'] = parameter_hash(model)
        hashes = [None]*world
        dist.all_gather_object(hashes, identity['initial_parameter_sha256'])
        if len(set(hashes)) != 1:
            raise ValueError('initialized rank parameters differ')
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=False, find_unused_parameters=args.arm=='history')
        raw_model = model.module
        all_indices = draw(train, config['updates']*effective, seed+1)
        indices = rank_indices(all_indices, config['microbatch'], config['accumulation'], rank, world, step)
        identity['global_draw_sha256'] = hashlib.sha256(all_indices.tobytes()).hexdigest()
        if rank == 0:
            base.atomic_json(out/'input_manifest.json', identity)
        validation = (mixed_indices(val, config['validation_samples'], config['validation_seed'], equal_sources=True)
                      if args.mixed_data else balanced_indices(val, config['validation_samples'], config['validation_seed']))
        natural = np.random.default_rng(config['natural_validation_seed']).choice(
            len(val), config['validation_samples'], replace=len(val)<config['validation_samples'])
        if rank == 0:
            np.save(out/'validation_balanced.npy', validation); np.save(out/'validation_natural.npy', natural)
        batches = iter(base.loader(train, indices, config['microbatch'], config['workers']))
        started = time.time()
        deadline = args.deadline or started+config['group_seconds']
        deadlines = [deadline if rank == 0 else None]
        dist.broadcast_object_list(deadlines, src=0)
        deadline = deadlines[0]
        optimizer.zero_grad(set_to_none=True)
        if rank == 0:
            trainlog = (out/'train.jsonl').open('a')
        evaluation_batch = config.get('validation_microbatch', config['microbatch'])
        if args.mixed_data:
            standard_evaluate = base.evaluate
            def mixed_evaluate(net, dataset, ids, arm, batch_size, amp, intervention=False):
                measured = standard_evaluate(net, dataset, ids, arm, batch_size, amp, intervention)
                source_metrics = []
                for source_id, desc in enumerate(dataset.meta['sources']):
                    selected = ids[(ids >= dataset.offsets[source_id]) & (ids < dataset.offsets[source_id+1])]
                    if not len(selected): continue
                    metrics = standard_evaluate(net, dataset, selected, arm, batch_size, amp, intervention)
                    source_metrics.append(metrics)
                    for key, value in metrics.items(): measured['source/'+desc['name']+'/'+key] = value
                key = 'model/anchor/cat0/h24/point_epe'
                if len(source_metrics) == len(dataset.sources) and all(key in m for m in source_metrics):
                    for common in set.intersection(*(set(m) for m in source_metrics)):
                        measured['pooled/'+common] = measured[common]
                        measured[common] = sum(m[common] for m in source_metrics)/len(source_metrics)
                elif not intervention and dataset is val and np.array_equal(ids, validation):
                    raise ValueError('fixed source validation panel missing moving anchors')
                return measured
            base.evaluate = mixed_evaluate
        engineering = args.smoke or args.engineering or args.benchmark
        if (args.init_weights or args.continue_from) and not engineering:
            measured = base.evaluate(raw_model, val, validation, args.arm, evaluation_batch, config['amp']) if rank==0 else None
            selection = base.validation_selection(measured) if rank==0 else None
            score = [selection['score'] if rank==0 else None]
            dist.broadcast_object_list(score, src=0)
            best = score[0]
            if rank == 0:
                baseline = dict(step=0, parent_step=identity['initialization']['parent_step'],
                                metrics=measured, selection=selection)
                base.atomic_json(out/'validation_initial.json', baseline)
                with (out/'validation.jsonl').open('a') as f:
                    f.write(json.dumps(baseline)+'\n')
            save_checkpoint(out/'best.pt', raw_model, optimizer, step, config, best, identity, rank, world)
        torch.cuda.reset_peak_memory_stats(device)
        reason = None
        while step < config['updates']:
            drift = rank == 0 and step % 20 == 0 and sources_drifted(identity, args.config, args.stats)
            requested = stop[0] or time.time()>=deadline or drift or (args.stop_after is not None and step>=args.stop_after)
            if any_rank(requested, device):
                reason = 'source_drift' if drift else 'stop_or_budget'
                break
            before = time.monotonic()
            if config.get('learning_rate_schedule', 'cosine') == 'constant':
                scale = 1.
            elif step < config['warmup_updates']:
                scale = (step+1)/config['warmup_updates']
            else:
                scale = .1+.9*.5*(1+math.cos(math.pi*(step-config['warmup_updates'])/max(1,config['updates']-config['warmup_updates'])))
            for group in optimizer.param_groups:
                group['lr'] = config['learning_rate']*scale
            update_loss, input_seconds = 0., 0.
            for micro in range(accumulation):
                fetch_started = time.monotonic()
                cpu_batch = next(batches)
                input_seconds += time.monotonic()-fetch_started
                batch = base.device_batch(cpu_batch)
                global_micro = micro*world+rank
                donor_seed = seed+step*config['accumulation']+global_micro
                use = base.shuffled_with_donors(batch, train, donor_seed) if args.arm=='shuffle' else batch
                with sync_context(model, micro, accumulation):
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config['amp']):
                        pred = model(use, args.arm)
                    loss, _ = (physical_loss(raw_model, pred, batch, loss_scales) if loss_scales is not None
                               else raw_model.loss(pred, batch))
                    if any_rank(not bool(torch.isfinite(loss)), device):
                        raise FloatingPointError('a rank produced nonfinite loss')
                    (loss/accumulation).backward()
                update_loss += float(loss.detach())/accumulation
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'], error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            step += 1
            torch.cuda.synchronize(device)
            values = torch.tensor([update_loss, time.monotonic()-before,
                                   torch.cuda.max_memory_allocated(device)/1048576,
                                   torch.cuda.max_memory_reserved(device)/1048576, input_seconds], device=device)
            all_values = [torch.empty_like(values) for _ in range(world)]
            dist.all_gather(all_values, values)
            values_cpu = torch.stack(all_values).cpu().tolist()
            row = dict(step=step, loss=sum(x[0] for x in values_cpu)/world, gradient_norm=float(grad),
                       seconds=max(x[1] for x in values_cpu), elapsed_seconds=time.time()-started,
                       peak_allocated_mib=max(x[2] for x in values_cpu),
                       peak_reserved_mib=max(x[3] for x in values_cpu),
                       input_seconds=max(x[4] for x in values_cpu),
                       learning_rate=optimizer.param_groups[0]['lr'], world_size=world, global_effective_batch=effective)
            if rank == 0:
                trainlog.write(json.dumps(row)+'\n');trainlog.flush()
                base.atomic_json(out/'progress.json', dict(status='RUNNING', **row))
                if step == 1 or step % 20 == 0 or args.smoke:
                    print(json.dumps(row), flush=True)
            can_evaluate = not any_rank(stop[0] or time.time()>=deadline, device)
            if can_evaluate and not (args.benchmark or args.engineering) and (step % config['validation_interval']==0 or step==config['updates']):
                measured = base.evaluate(raw_model, val, validation, args.arm, evaluation_batch, config['amp']) if rank==0 else None
                selection = base.validation_selection(measured) if rank==0 else None
                score = [selection['score'] if rank==0 else None]
                dist.broadcast_object_list(score, src=0)
                if rank == 0:
                    base.atomic_json(out/'validation_latest.json',
                                     dict(step=step, metrics=measured, selection=selection))
                    with (out/'validation.jsonl').open('a') as f:
                        f.write(json.dumps(dict(step=step, metrics=measured,
                                                selection=selection))+'\n')
                if score[0] < best:
                    best = score[0]
                    if not args.smoke:
                        save_checkpoint(out/'best.pt', raw_model, optimizer, step, config, best, identity, rank, world)
            if not engineering and step % config['checkpoint_interval']==0:
                save_checkpoint(out/'latest.pt', raw_model, optimizer, step, config, best, identity, rank, world)
        hashes = [None]*world
        dist.all_gather_object(hashes, parameter_hash(raw_model))
        if len(set(hashes))!=1:
            raise AssertionError('rank parameters diverged')
        if not engineering:
            save_checkpoint(out/'latest.pt', raw_model, optimizer, step, config, best, identity, rank, world)
        if not args.benchmark:
            save_checkpoint(out/'final.pt', raw_model, optimizer, step, config, best, identity, rank, world)
        if rank == 0:
            final = dict(status='COMPLETED' if step==config['updates'] else 'BUDGET_STOP', step=step,
                         elapsed_seconds=time.time()-started, reason=reason, world_size=world,
                         rank_parameter_hashes=hashes, rank_parameters_identical=True,
                         engineering_only=engineering,
                         best_validation_score=best if math.isfinite(best) else None)
            if time.time()<deadline and not stop[0] and not (args.benchmark or args.engineering):
                final['balanced'] = base.evaluate(raw_model, val, validation, args.arm, evaluation_batch, config['amp'])
                final['natural'] = base.evaluate(raw_model, val, natural, args.arm, evaluation_batch, config['amp'])
                if args.arm=='action':
                    final['validation_shuffle'] = base.evaluate(raw_model, val, validation, args.arm, evaluation_batch, config['amp'], True)
            base.atomic_json(out/'result.json', final)
            base.atomic_json(out/'progress.json', {k:v for k,v in final.items() if k not in ('balanced','natural','validation_shuffle')})
            print(json.dumps({k:v for k,v in final.items() if k not in ('balanced','natural','validation_shuffle')}), flush=True)
        dist.barrier()
    except BaseException as exc:
        if rank == 0 and args.output.exists():
            base.atomic_json(args.output/'progress.json', dict(status='FAILED', error=str(exc)))
        raise
    finally:
        if trainlog is not None:
            trainlog.close()
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--config', type=Path, default=TASK/'configs/pointworld_temporal_wm24.json')
    p.add_argument('--stats', type=Path, required=True)
    p.add_argument('--loss-stats', type=Path, help='Separate main-source train-only physical loss scales; preserves pretrained forward buffers')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=['history','action','shuffle'], required=True)
    p.add_argument('--steps', type=int)
    p.add_argument('--deadline', type=float)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--stop-after', type=int)
    p.add_argument('--fused-hilbert', action='store_true')
    p.add_argument('--engineering', action='store_true', help='Keep requested batch, save final checkpoint, skip evaluation')
    p.add_argument('--mixed-data', action='store_true', help='Use frozen registered-source manifest and source-balanced draws')
    p.add_argument('--benchmark', action='store_true', help='Engineering throughput only; skip checkpoints and validation')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--resume', type=Path)
    group.add_argument('--import-single-checkpoint', type=Path)
    group.add_argument('--init-weights', type=Path, help='New recipe initialized from single-GPU model weights only')
    group.add_argument('--continue-from', type=Path, help='Two-rank recipe migration retaining trained AdamW moments')
    group.add_argument('--random-init', action='store_true', help='Start a fresh model with the configured seed')
    run(p.parse_args())


if __name__=='__main__':
    main()
