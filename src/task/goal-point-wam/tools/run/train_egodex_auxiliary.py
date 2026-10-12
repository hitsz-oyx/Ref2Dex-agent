#!/usr/bin/env python3
"""Single-model main-three WAM + native EgoDex actor auxiliary, local or DDP."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

import train_three_domain as base
from oakink_wm.multisource import MixedWindows
from goal_point_wam.egodex_auxiliary import EgoDexActorWindows, ego_collate, auxiliary_model_class


@torch.no_grad()
def evaluate_ego(model, dataset, indices, device, batch_size):
    sums, counts, static = {}, {}, {}
    model.eval()
    for cpu in DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, collate_fn=ego_collate):
        batch = base.move_batch(cpu, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(batch)['trajectory']
        error = torch.linalg.vector_norm(pred-batch['trajectory_target'], dim=-1)
        reference = torch.linalg.vector_norm(batch['trajectory_target'], dim=-1)
        for i, record in enumerate(batch['sample_id'][:, 0].tolist()):
            valid = batch['trajectory_valid'][i]
            sums[record] = sums.get(record, 0.)+float(error[i][valid].sum())
            static[record] = static.get(record, 0.)+float(reference[i][valid].sum())
            counts[record] = counts.get(record, 0)+int(valid.sum())
    model.train()
    records = {str(r):dict(epe_m=sums[r]/counts[r], static_epe_m=static[r]/counts[r], points=counts[r]) for r in sums}
    return dict(records=records, macro_epe_m=float(np.mean([r['epe_m'] for r in records.values()])),
                macro_static_epe_m=float(np.mean([r['static_epe_m'] for r in records.values()])))


@torch.no_grad()
def goal_sensitivity(model, dataset, indices, device):
    model.eval()
    sums = {n:dict(real=0., static=0.) for n in dataset.source_names}
    counts = {n:0 for n in dataset.source_names}
    for cpu in base.loader(dataset, indices, 16, 0):
        batch = base.move_batch(cpu, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            scene = model.scene(batch)
            for name, endpoint in [('real', batch['goal_object_points']), ('static', batch['start_object_points'])]:
                tokens, mask = model.goal(batch['start_object_points'], endpoint, batch['object_goal_valid'])
                pred = model.trajectory(scene['scene_tokens'], scene['scene_mask'], tokens, mask,
                                        model.observed_hand_condition(batch))
                error = torch.linalg.vector_norm(pred-batch['trajectory_target'], dim=-1)
                for row, source_id in enumerate(batch['source_id'].tolist()):
                    source = dataset.source_names[source_id]
                    sums[source][name] += float(error[row][batch['trajectory_valid'][row]].sum())
        for row, source_id in enumerate(batch['source_id'].tolist()):
            counts[dataset.source_names[source_id]] += int(batch['trajectory_valid'][row].sum())
    model.train()
    return {n:{k:v/counts[n] for k,v in values.items()} for n,values in sums.items() if counts[n]}


def save(path, model, optimizer, step, config, identity, rank, world):
    rng = dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
               numpy_rng=np.random.get_state(), python_rng=random.getstate())
    states = [None]*world
    if world > 1:
        dist.all_gather_object(states, rng)
    else:
        states = [rng]
    if rank == 0:
        part = path.with_suffix('.pt.part')
        torch.save(dict(checkpoint_kind='goal-point-wam.single.v1', model=model.state_dict(),
                        optimizer=optimizer.state_dict(), step=step, config=config, identity=identity,
                        distributed_rng=states, **rng), part)
        part.replace(path)
    if world > 1:
        dist.barrier()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'stats', 'ego-data', 'init-wam', 'config', 'validation-indices', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--engineering-only', action='store_true')
    p.add_argument('--no-checkpoint', action='store_true')
    a = p.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    config = json.loads(a.config.read_text())
    if world != config['world_size'] or config.get('goal_type') != 'object_points' or config['auxiliary_period'] != 4:
        raise ValueError('fixed object-goal/three-main:one-aux recipe required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for native WAM training')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(seconds=180))
    log, step = None, 0
    try:
        if rank == 0:
            a.output.mkdir(parents=True, exist_ok=False)
        if world > 1:
            dist.barrier()
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        stats = json.loads(a.stats.read_text())
        source_train, source_val = MixedWindows(a.data, 'train'), MixedWindows(a.data, 'val')
        if (stats['sources'] != list(base.MAIN_SOURCE_NAMES) or stats['split'] != 'train'
                or stats['normalization_source_manifest_sha256'] != source_train.meta['normalization_source_manifest_sha256']):
            raise ValueError('unchanged train-only main-three normalization required')
        train, val = [base.GoalWindows(d, 'object_points', config['goal_points']) for d in (source_train, source_val)]
        ego_train, ego_val = [EgoDexActorWindows(a.ego_data, s) for s in ('train', 'val')]
        panel = json.loads((a.validation_indices.parent/'panel.json').read_text())
        if panel['data_manifest_sha256'] != base.digest(a.data/'processed/manifest.json') or panel['stats_sha256'] != base.digest(a.stats):
            raise ValueError('main-three frozen panel drift')
        if panel['validation_indices_sha256'] != base.digest(a.validation_indices):
            raise ValueError('frozen panel index drift')
        original_indices = np.load(a.validation_indices)
        source_ids = np.searchsorted(source_val.offsets, original_indices, side='right')-1
        validation = np.concatenate([original_indices[source_ids == s][:config['main_validation_per_source']]
                                     for s in range(3)])
        ego_validation = np.concatenate([np.flatnonzero(ego_val.rows[:, 0] == r)[:64]
                                         for r in np.unique(ego_val.rows[:, 0])])
        ego_fit = np.arange(min(128, len(ego_train)))
        cls = auxiliary_model_class(base.GoalPointWAM)
        model = cls(stats, config)
        parent_sha = base.digest(a.init_wam)
        parent = torch.load(a.init_wam, map_location='cpu', weights_only=False)
        if base.digest(a.init_wam) != parent_sha:
            raise ValueError('warm-start checkpoint replaced while loading')
        if parent['checkpoint_kind'] != 'goal-point-wam.single.v1' or parent['identity']['stats_sha256'] != base.digest(a.stats):
            raise ValueError('native WAM/stat warm-start mismatch')
        if parent['identity']['model_sha256'] != base.digest(base.TASK/'src/goal_point_wam/model.py'):
            raise ValueError('warm-start model semantics changed')
        model.load_state_dict(parent['model'], strict=True)
        identity = dict(git_commit=base.subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                        script_sha256=base.digest(Path(__file__)), model_sha256=base.digest(base.TASK/'src/goal_point_wam/model.py'),
                        auxiliary_sha256=base.digest(base.TASK/'src/goal_point_wam/egodex_auxiliary.py'),
                        single_trainer_sha256=base.digest(Path(base.__file__)), stats_sha256=base.digest(a.stats),
                        data_manifest_sha256=base.digest(a.data/'processed/manifest.json'),
                        ego_manifest_sha256=base.digest(a.ego_data/'manifest.json'), config_sha256=base.digest(a.config),
                        parent_sha256=parent_sha, parent_step=parent['step'], optimizer_reset=True,
                        initial_parameter_sha256=base.parameter_hash(model), sources=list(base.MAIN_SOURCE_NAMES)+['egodex_actor_only'],
                        world_size=world, global_batch=world*config['microbatch'],
                        recipe='3 main WAM updates then 1 native hand auxiliary; no object goal/world labels on EgoDex',
                        ego_fit_windows=len(ego_train), ego_val_windows=len(ego_val), engineering_only=a.engineering_only)
        del parent
        model.to(device)
        wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                          find_unused_parameters=True) if world > 1 else model
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        main_steps = sum((s+1)%4 != 0 for s in range(config['updates']))
        aux_steps = config['updates']-main_steps
        seed = config['split_seed']+1000*rank
        main_draw = base.mixed_indices(source_train, main_steps*config['microbatch'], seed).astype('int32')
        ego_draw = np.random.default_rng(seed+1).integers(len(ego_train), size=aux_steps*config['microbatch'], dtype='int32')
        np.save(a.output/f'rank{rank}_main_indices.npy', main_draw)
        np.save(a.output/f'rank{rank}_ego_indices.npy', ego_draw)
        identity['rank_draw_hashes'] = dict(main=hashlib.sha256(main_draw.tobytes()).hexdigest(), ego=hashlib.sha256(ego_draw.tobytes()).hexdigest())
        ranks = [dict(rank=rank, pid=os.getpid())]
        if world > 1:
            ranks = [None]*world; dist.all_gather_object(ranks, dict(rank=rank, pid=os.getpid(), initial_parameter_sha256=identity['initial_parameter_sha256']))
            if len({r['initial_parameter_sha256'] for r in ranks}) != 1:
                raise ValueError('rank initialization mismatch')
        identity['ranks'] = ranks
        if rank == 0:
            base.atomic_json(a.output/'input_manifest.json', identity)
            base.atomic_json(a.output/'config.json', config)
            np.save(a.output/'validation_main.npy', validation); np.save(a.output/'validation_ego.npy', ego_validation)
        main_batches = iter(base.loader(train, main_draw, config['microbatch'], config['workers']))
        ego_batches = iter(DataLoader(Subset(ego_train, ego_draw.tolist()), batch_size=config['microbatch'],
                                     num_workers=config['auxiliary_workers'], pin_memory=True, collate_fn=ego_collate))
        random.seed(config['seed']+rank); np.random.seed(config['seed']+rank); torch.manual_seed(config['seed']+rank)
        stop = [False]
        signal.signal(signal.SIGUSR1, lambda *_:stop.__setitem__(0, True))
        started = time.time(); log = (a.output/f'rank{rank}.jsonl').open('w')
        observations = {}
        def evaluation():
            if world > 1:
                dist.barrier()
            if rank == 0:
                measured = dict(step=step, main=base.evaluate(model, val, validation, config['microbatch'], 0, device),
                    egodex_dev=evaluate_ego(model, ego_val, ego_validation, device, config['microbatch']),
                    egodex_fit=evaluate_ego(model, ego_train, ego_fit, device, config['microbatch']),
                    main_goal=goal_sensitivity(model, val, validation, device))
                with (a.output/'validation.jsonl').open('a') as f:
                    f.write(json.dumps(measured, allow_nan=False)+'\n')
                observations.update(measured)
            if world > 1:
                dist.barrier()
        if config.get('initial_evaluation', True):
            evaluation()
        for step in range(1, config['updates']+1):
            flag = torch.tensor(int(stop[0] or time.time()-started > config['group_seconds']), device=device)
            if world > 1:
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if flag.item():
                step -= 1
                break
            tick = time.monotonic()
            auxiliary = step%4 == 0
            batch = base.move_batch(next(ego_batches if auxiliary else main_batches), device)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = config['learning_rate']*(.1+.9*.5*(1+np.cos(np.pi*(step-1)/config['updates'])))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = wrapped(batch)
            loss, terms = model.loss(pred, batch)
            finite = torch.isfinite(loss).to(torch.int32)
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError('nonfinite native auxiliary loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'], error_if_nonfinite=True)
            optimizer.step()
            row = dict(step=step, auxiliary=auxiliary, loss=float(loss.detach()), gradient_norm=float(grad),
                       seconds=time.monotonic()-tick, elapsed_seconds=time.time()-started,
                       valid_trajectory_points=int(batch['trajectory_valid'].sum()), world=float(terms['world']))
            log.write(json.dumps(row)+'\n'); log.flush()
            if step == 1 or step%10 == 0:
                base.atomic_json(a.output/f'rank{rank}_progress.json', dict(status='RUNNING', **row))
                if rank == 0:
                    base.atomic_json(a.output/'progress.json', dict(status='RUNNING', **row,
                                     gpu=base.gpu_snapshot(device), world_size=world))
            if step%config['validation_interval'] == 0:
                evaluation()
            if step%config['checkpoint_interval'] == 0 and not a.no_checkpoint:
                save(a.output/'latest.pt', model, optimizer, step, config, identity, rank, world)
        if not observations or observations['step'] != step:
            evaluation()
        if not a.no_checkpoint:
            save(a.output/'latest.pt', model, optimizer, step, config, identity, rank, world)
            if rank == 0:
                os.link(a.output/'latest.pt', a.output/'final.pt')
        hashes = [base.parameter_hash(model)]
        if world > 1:
            hashes = [None]*world; dist.all_gather_object(hashes, base.parameter_hash(model))
        if len(set(hashes)) != 1:
            raise ValueError('distributed parameter divergence')
        if rank == 0:
            result = dict(status='COMPLETED' if step == config['updates'] else 'BUDGET_STOP', step=step,
                          elapsed_seconds=time.time()-started, engineering_only=a.engineering_only,
                          rank_parameter_sha256=hashes, validation=observations, gpu=base.gpu_snapshot(device))
            base.atomic_json(a.output/'result.json', result); base.atomic_json(a.output/'progress.json', result)
            print(json.dumps(result), flush=True)
    except BaseException as error:
        if a.output.exists():
            base.atomic_json(a.output/f'rank{rank}_error.json', dict(step=step, error=repr(error)))
        raise
    finally:
        if log:
            log.close()
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
