#!/usr/bin/env python3
"""Bounded model-only warm-start method Probe; no changes to the live DDP trainer."""
import argparse
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

from train_three_domain import (GoalWindows, loader, move_batch, evaluate, atomic_json,
                                digest, parameter_hash, gpu_snapshot)
from ref12_methods import Ref12Method
from oakink_wm.multisource import MixedWindows, mixed_indices


def save(path, model, optimizer, step, config, identity):
    part = path.with_suffix('.pt.part')
    torch.save(dict(checkpoint_kind='goal-point-wam.single.v1', model=model.state_dict(),
                    optimizer=optimizer.state_dict(), step=step, config=config, identity=identity,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                    numpy_rng=np.random.get_state(), python_rng=random.getstate()), part)
    part.replace(path)


def goal_ablation(model, dataset, indices, batch_size):
    totals = {name: dict(count=0, real=0., static=0., zero=0.) for name in dataset.source_names}
    model.eval()
    with torch.no_grad():
        for cpu in loader(dataset, indices, batch_size, 2):
            batch = move_batch(cpu, torch.device('cuda'))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                scene = model.scene(batch)
                def predict(goal):
                    return model.actor_from_scene(scene, dict(batch, goal_object_points=goal))['trajectory']
                real = predict(batch['goal_object_points'])
                static = predict(batch['start_object_points'])
            errors = {key: torch.linalg.vector_norm(value - batch['trajectory_target'], dim=-1)
                      for key, value in [('real', real), ('static', static), ('zero', torch.zeros_like(real))]}
            for row, sid in enumerate(batch['source_id'].tolist()):
                t = totals[dataset.source_names[sid]]; valid = batch['trajectory_valid'][row]
                t['count'] += int(valid.sum())
                for key, values in errors.items():
                    t[key] += float(values[row][valid].sum())
    return {name: {key + '_epe_m': value / t['count'] for key, value in t.items() if key != 'count'} for name, t in totals.items()}


def auxiliary_metrics(model, dataset, indices, batch_size):
    totals = {name: dict(count=0, prediction=0., zero=0.) for name in dataset.source_names}
    model.eval()
    with torch.no_grad():
        for cpu in loader(dataset, indices, batch_size, 2):
            batch = move_batch(cpu, torch.device('cuda'))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(batch)
            if model.method == 'change_supervision':
                from oakink_wm.model import rigid_points
                effect = batch['effect']
                centers = rigid_points(effect[..., :3, :3], effect[..., :3, 3], batch['points']).mean(-2)
                prior = torch.cat((batch['points'].mean(-2)[:, :, None], centers[:, :, :-1]), 2)
                target = centers - prior
                estimate = prediction['future_point_change']; mask = batch['object_valid'][:, :, None].expand_as(target[..., 0])
                zero = torch.zeros_like(target)
            else:
                from ref12_methods import goal_points_from_world
                rows = torch.arange(len(batch['goal_object_slot']), device=batch['goal_object_slot'].device)
                effect = batch['effect'][rows, batch['goal_object_slot']]
                if model.method == 'intermediate_plan':
                    target = torch.einsum('btij,bnj->btni', effect[..., :3, :3], batch['start_object_points']) + effect[..., None, :3, 3]
                    estimate = prediction['planned_object_points']
                    zero = batch['start_object_points'][:, None].expand_as(target)
                    mask = batch['object_goal_valid'][:, None].expand_as(target[..., 0])
                else:
                    target = batch['goal_object_points']; estimate = goal_points_from_world(prediction, batch)
                    zero = batch['start_object_points']; mask = batch['object_goal_valid']
            error = torch.linalg.vector_norm(estimate - target, dim=-1)
            zero_error = torch.linalg.vector_norm(zero - target, dim=-1)
            for row, sid in enumerate(batch['source_id'].tolist()):
                t = totals[dataset.source_names[sid]]; valid = mask[row]
                t['count'] += int(valid.sum()); t['prediction'] += float(error[row][valid].sum()); t['zero'] += float(zero_error[row][valid].sum())
    return {name: dict(valid_values=t['count'], prediction_epe_m=t['prediction']/t['count'], static_epe_m=t['zero']/t['count']) for name, t in totals.items()}


def run(args):
    config = json.loads(args.config.read_text())
    if args.smoke:
        config.update(updates=2, validation_samples=6, validation_interval=2, checkpoint_interval=2, workers=0)
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = True
    seed = config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    parent = torch.load(args.parent, map_location='cpu', weights_only=False)
    if parent['checkpoint_kind'] != 'goal-point-wam.single.v1' or parent.get('artificial_labels'):
        raise ValueError('real model-only parent required')
    stats = json.loads(args.stats.read_text())
    identity = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    stats_sha256=digest(args.stats), data_manifest_sha256=digest(args.data / 'processed/manifest.json'),
                    model_sha256=digest(Path(__file__).resolve().parents[2] / 'src/goal_point_wam/model.py'),
                    config_sha256=digest(args.config), script_sha256=digest(Path(__file__)), method_sha256=digest(Path(__file__).with_name('ref12_methods.py')),
                    parent_checkpoint_sha256=digest(args.parent), parent_step=parent['step'],
                    parent_identity=parent['identity'], no_video=True, transfer='model only; fresh optimizer', method=config['method'])
    for key in ('stats_sha256', 'data_manifest_sha256', 'model_sha256'):
        if identity[key] != parent['identity'][key]:
            raise ValueError('parent identity drift: ' + key)
    structural = ('hidden', 'patch_size', 'scene_tokens', 'goal_layers', 'trajectory_layers', 'world_layers',
                  'goal_type', 'goal_points', 'goal_coordinate_scale_m', 'object_goal_encoding',
                  'goal_injection', 'goal_query_adapter', 'observed_hand_query', 'voxel_m', 'voxel_origin_m')
    if any(config.get(k) != parent['config'].get(k) for k in structural):
        raise ValueError('warm-start model structure changed')
    model = Ref12Method(stats, config)
    keys = model.load_state_dict(parent['model'], strict=False)
    allowed = (('horizon_condition.',) if config['method'] == 'hindsight_prefix' else
               ('future_change.',) if config['method'] == 'change_supervision' else
               ('plan_decoder.', 'plan_head.') if config['method'] == 'intermediate_plan' else ())
    if keys.unexpected_keys or any(not k.startswith(allowed) for k in keys.missing_keys):
        raise ValueError('unaccounted model keys: ' + str(keys))
    identity.update(initial_parameter_sha256=parameter_hash(model), sources=['oakink2', 'grab', 'arctic'],
                    added_parameters=keys.missing_keys, physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    del parent
    model.cuda().train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'], weight_decay=.01)
    train_base = MixedWindows(args.data, 'train')
    train = GoalWindows(train_base, goal_type='object_points', goal_points=32)
    val = GoalWindows(MixedWindows(args.data, 'val'), goal_type='object_points', goal_points=32)
    indices = mixed_indices(train_base, config['updates'] * config['microbatch'], config['split_seed'])
    validation = np.load(args.indices)
    if args.smoke:
        validation = validation[:6]
    np.save(root / 'train_indices.npy', indices); np.save(root / 'validation_balanced.npy', validation)
    identity['train_indices_sha256'] = digest(root / 'train_indices.npy')
    identity['validation_indices_sha256'] = digest(root / 'validation_balanced.npy')
    atomic_json(root / 'input_manifest.json', identity); atomic_json(root / 'config.json', config)
    batches = iter(loader(train, indices, config['microbatch'], config['workers']))
    stopped = [False]; signal.signal(signal.SIGUSR1, lambda *_: stopped.__setitem__(0, True))
    started = time.time(); step = 0; curve = []
    try:
        with (root / 'train.jsonl').open('w') as log:
            while step < config['updates'] and time.time() < args.deadline and not stopped[0]:
                tick = time.monotonic()
                if step < 100:
                    scale = (step + 1) / 100
                else:
                    scale = .1 + .9 * .5 * (1 + math.cos(math.pi * (step - 100) / max(1, config['updates'] - 100)))
                for group in optimizer.param_groups:
                    group['lr'] = config['learning_rate'] * scale
                batch = move_batch(next(batches), torch.device('cuda'))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(batch)
                loss, terms = model.loss(pred, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite method loss')
                if args.smoke and config['method'] == 'frozen_goal_consistency':
                    goal_error = (pred['policy_goal_points'] - model._supervised_batch['goal_object_points']).square().mean()
                    goal_gradient, = torch.autograd.grad(goal_error, model.trajectory.head[-1].weight, retain_graph=True)
                    if not torch.isfinite(goal_gradient).all() or goal_gradient.norm() <= 0:
                        raise RuntimeError('goal-only loss does not reach actor through frozen world')
                    atomic_json(root / 'goal_gradient_smoke.json', dict(goal_only_actor_gradient_norm=float(goal_gradient.norm())))
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                if args.smoke and config['method'] == 'frozen_goal_consistency':
                    if any(p.grad is not None for module in (model.scene, model.action, model.world) for p in module.parameters()):
                        raise RuntimeError('frozen world acquired parameter gradients')
                    if not any(p.grad is not None and p.grad.norm() > 0 for p in model.trajectory.parameters()):
                        raise RuntimeError('actor gradient missing')
                optimizer.step(); step += 1
                row = dict(status='RUNNING', step=step, loss=float(loss), gradient_norm=float(grad),
                           seconds=time.monotonic() - tick, elapsed_seconds=time.time() - started,
                           learning_rate=optimizer.param_groups[0]['lr'], **{k: float(v) for k, v in terms.items()})
                log.write(json.dumps(row) + '\n'); log.flush()
                if step == 1 or step % 10 == 0:
                    row['gpu'] = gpu_snapshot(torch.device('cuda'))
                    atomic_json(root / 'progress.json', row); print(json.dumps(row), flush=True)
                if step % config['validation_interval'] == 0 or step == config['updates']:
                    measured = evaluate(model, val, validation, config['microbatch'], config['workers'], torch.device('cuda'))
                    curve.append(dict(step=step, metrics=measured))
                    atomic_json(root / 'validation_curve.json', dict(curve=curve))
                if step % config['checkpoint_interval'] == 0:
                    save(root / 'latest.pt', model, optimizer, step, config, identity)
        save(root / 'latest.pt', model, optimizer, step, config, identity)
        os.link(root / 'latest.pt', root / 'final.pt')
        result = dict(status='COMPLETED' if step == config['updates'] else 'BUDGET_STOP', step=step,
                      method=config['method'], elapsed_seconds=time.time() - started, engineering_only=args.smoke,
                      validation=curve[-1] if curve else None, gpu=gpu_snapshot(torch.device('cuda')))
        if args.smoke:
            model.eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                scene = model.scene(batch)
                first = model.actor_from_scene(scene, batch)['trajectory']
                changed = dict(batch, action=torch.zeros_like(batch['action']), effect=torch.zeros_like(batch['effect']))
                second = model.actor_from_scene(scene, changed)['trajectory']
                difference = float((first - second).abs().max())
            if difference != 0.:
                raise RuntimeError('future action/effect label reaches actor inference')
            result['cached_actor_label_leak_max'] = difference
        if not args.smoke and config['method'] != 'world_roi':
            result['goal_ablation'] = goal_ablation(model, val, validation, config['microbatch'])
        if not args.smoke:
            result['auxiliary_metrics'] = auxiliary_metrics(model, val, validation, config['microbatch'])
        atomic_json(root / 'result.json', result); atomic_json(root / 'progress.json', result)
    except BaseException as error:
        atomic_json(root / 'progress.json', dict(status='FAILED', step=step, error=repr(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'stats', 'config', 'output', 'parent', 'indices'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())
