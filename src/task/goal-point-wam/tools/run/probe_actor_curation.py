#!/usr/bin/env python3
"""Paired frozen-context decoder adaptation for actor window weighting."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from train_three_domain import GoalPointWAM, GoalWindows, loader, move_batch, gpu_snapshot
from oakink_wm.multisource import MixedWindows, mixed_indices
from goal_point_wam.curation import weighted_trajectory_loss

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'tools/audit'))
from audit_native_curation import masks


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


@torch.no_grad()
def cache(model, dataset, indices, label, include_static=False):
    values = {k: [] for k in ('scene', 'scene_mask', 'goal', 'goal_mask', 'hand', 'target', 'valid', 'source')}
    if include_static:
        values.update(static_goal=[], static_mask=[])
    started = time.time()
    for cpu in loader(dataset, indices, 32, 0):
        batch = move_batch(cpu, torch.device('cuda'))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            scene = model.scene(batch)
            goal, gm = model.goal(batch['start_object_points'], batch['goal_object_points'], batch['object_goal_valid'])
            hand = model.observed_hand_condition(batch)
            if include_static:
                sg, sm = model.goal(batch['start_object_points'], batch['start_object_points'], batch['object_goal_valid'])
                values['static_goal'].append(sg.detach()); values['static_mask'].append(sm)
        row = dict(scene=scene['scene_tokens'], scene_mask=scene['scene_mask'], goal=goal,
                   goal_mask=gm, hand=hand, target=batch['trajectory_target'],
                   valid=batch['trajectory_valid'], source=batch['source_id'])
        for key, value in row.items():
            values[key].append(value.detach())
        done = sum(len(x) for x in values['target'])
        if done % 1024 == 0 or done == len(indices):
            elapsed = time.time() - started
            print(json.dumps(dict(phase=label, done=done, total=len(indices), elapsed_s=elapsed,
                                 eta_s=elapsed / done * (len(indices) - done))), flush=True)
    return {key: torch.cat(value) for key, value in values.items()}


def predict(decoder, batch, static=False):
    return decoder(batch['scene'], batch['scene_mask'],
                   batch['static_goal'] if static else batch['goal'],
                   batch['static_mask'] if static else batch['goal_mask'], batch['hand'])


@torch.no_grad()
def evaluate(decoder, values, geometry):
    decoder.eval()
    arrays = {key: [] for key in ('first8_mm', 'all24_mm', 'static_all24_mm', 'goal_response_mm')}
    predictions = []
    for begin in range(0, len(values['target']), 64):
        batch = {key: value[begin:begin+64] for key, value in values.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred, static = predict(decoder, batch), predict(decoder, batch, True)
        target, valid = batch['target'], batch['valid']
        predictions.append(pred.cpu().numpy())
        for key, p, stop in [('first8_mm', pred, 8), ('all24_mm', pred, 24), ('static_all24_mm', static, 24)]:
            error = torch.linalg.vector_norm(p[:, :stop].float() - target[:, :stop], dim=-1)
            v = valid[:, :stop]
            arrays[key].append(((error * v).sum((1, 2, 3)) / v.sum((1, 2, 3)) * 1000).cpu().numpy())
        response = torch.linalg.vector_norm(pred.float() - static.float(), dim=-1)
        arrays['goal_response_mm'].append(((response * valid).sum((1, 2, 3)) / valid.sum((1, 2, 3)) * 1000).cpu().numpy())
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    source = values['source'].cpu().numpy()
    groups = dict(all=np.ones(len(source), bool), **masks(geometry))
    groups['static_departure'] = (geometry[:, 0] < 2) & (geometry[:, 4] <= 50) & (geometry[:, 6] > 100)
    groups['static_near_both_ends'] = (geometry[:, 0] < 2) & (geometry[:, 4] <= 50) & (geometry[:, 6] <= 50)
    report = {}
    for sid, name in enumerate(('oakink2', 'grab', 'arctic')):
        report[name] = {}
        for group, mask in groups.items():
            selected = mask & (source == sid)
            if selected.any():
                report[name][group] = dict(n=int(selected.sum()), **{key: float(v[selected].mean()) for key, v in arrays.items()})
    primary = np.mean([report[name][group]['first8_mm'] for name in report
                       for group in ('stratum/manipulation', 'static_goal_near') if group in report[name]])
    return dict(primary_first8_mm=float(primary), sources=report), arrays, np.concatenate(predictions)


def gates(a, b):
    checks = dict(primary_5pct=b['primary_first8_mm'] <= .95 * a['primary_first8_mm'])
    for name in a['sources']:
        for group in ('stratum/manipulation', 'static_goal_near'):
            left, right = a['sources'][name][group], b['sources'][name][group]
            checks[name + '/' + group + '/guard'] = right['first8_mm'] <= left['first8_mm'] * 1.05 + .1
        moving = b['sources'][name]['stratum/manipulation']
        checks[name + '/goal_utility'] = moving['all24_mm'] < moving['static_all24_mm']
    return {key: bool(value) for key, value in checks.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'checkpoint', 'stats', 'census', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    began = time.time(); seed = 299
    if not torch.cuda.is_available(): raise RuntimeError('idle GPU required')
    torch.set_num_threads(2); torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    train = MixedWindows(args.data, 'train'); val = MixedWindows(args.data, 'val')
    count, updates, batch_size = (192, 4, 16) if args.smoke else (9216, 1000, 96)
    indices = mixed_indices(train, count, seed)
    packet = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if packet['checkpoint_kind'] != 'goal-point-wam.single.v1' or packet.get('artificial_labels'):
        raise ValueError('real original single WAM checkpoint required')
    model = GoalPointWAM(json.loads(args.stats.read_text()), packet['config']).cuda().eval()
    model.load_state_dict(packet['model'], strict=True); model.requires_grad_(False)
    val_indices = np.load(args.checkpoint.parent / 'validation_balanced.npy')
    with np.load(args.census / 'model-r1/val.npz') as va:
        if not np.array_equal(va['index'], val_indices): raise ValueError('VAL panel mismatch')
        val_geometry = va['metrics'].copy()
    if args.smoke:
        take = np.concatenate([np.arange(i*512, i*512+16) for i in range(3)])
        val_indices, val_geometry = val_indices[take], val_geometry[take]
    train_geometry = np.empty((count, 9), np.float32)
    for sid, name in enumerate(('oakink2', 'grab', 'arctic')):
        selected = (indices >= train.offsets[sid]) & (indices < train.offsets[sid+1])
        with np.load(args.census / (name + '.npz')) as census:
            train_geometry[selected] = census['metrics'][indices[selected] - train.offsets[sid]]
    suspect = masks(train_geometry)['static_goal_far_moving'] & (indices < train.offsets[1])
    identity = dict(schema='ref2dex.actor-curation-paired.v1', status='RUNNING', engineering_only=args.smoke,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    checkpoint=str(args.checkpoint), checkpoint_step=packet['step'], seed=seed,
                    updates=updates, train_pool=count, batch=batch_size, lr=1e-5, downweight=.1,
                    suspected_train_windows=int(suspect.sum()), physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    world='all scene/goal/hand/action/world parameters and buffers frozen; no world optimization')
    write(args.output / 'manifest.json', identity)
    np.savez(args.output / 'inputs.npz', train_indices=indices, val_indices=val_indices,
             train_geometry=train_geometry, val_geometry=val_geometry, suspect=suspect)
    # Keep original checkpoint CPU state as the independent frozen-state reference.
    tc = cache(model, GoalWindows(train, 'object_points', 32), indices, 'cache_train')
    vc = cache(model, GoalWindows(val, 'object_points', 32), val_indices, 'cache_val', True)
    initial, initial_arrays, _ = evaluate(model.trajectory, vc, val_geometry)
    write(args.output / 'initial.json', initial)
    arms = {name: copy.deepcopy(model.trajectory).requires_grad_(True) for name in ('A', 'B')}
    optimizers = {name: torch.optim.AdamW(decoder.parameters(), lr=1e-5, weight_decay=.01)
                  for name, decoder in arms.items()}
    weights = torch.tensor(np.where(suspect, .1, 1.), dtype=torch.float32, device='cuda')
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, count, size=(updates, batch_size), dtype=np.int64)
    np.save(args.output / 'draws.npy', draws)
    history = []; training_start = time.time(); gross = 0
    for step, row in enumerate(draws, 1):
        ids = torch.tensor(row, device='cuda')
        batch = {key: value[ids] for key, value in tc.items()}
        losses = {}
        for name, decoder in arms.items():
            decoder.train(); torch.manual_seed(seed + step)
            optimizers[name].zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16): pred = predict(decoder, batch)
            weight = torch.ones(len(row), device='cuda') if name == 'A' else weights[ids]
            loss = weighted_trajectory_loss(pred, batch['target'], batch['valid'], weight, model.trajectory_scale_m)
            if not torch.isfinite(loss): raise ValueError('nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.)
            if not torch.isfinite(norm): raise ValueError('nonfinite gradient')
            optimizers[name].step(); losses[name] = float(loss)
        if step % 100 == 0 or step == updates:
            elapsed = time.time() - training_start
            progress = dict(step=step, total=updates, losses=losses, elapsed_s=elapsed,
                            eta_s=elapsed / step * (updates - step), gpu=gpu_snapshot(torch.device('cuda')))
            write(args.output / 'progress.json', progress); print(json.dumps(progress), flush=True)
        if step in (250, 500, updates):
            reports = {}
            for name, decoder in arms.items():
                report, arrays, predictions = evaluate(decoder, vc, val_geometry)
                reports[name] = report
                if step == updates:
                    np.savez_compressed(args.output / (name + '-final-eval.npz'), prediction=predictions,
                                        target=vc['target'].cpu().numpy(), valid=vc['valid'].cpu().numpy(), **arrays)
                    torch.save(dict(schema='ref2dex.curation-decoder.v1', trajectory=decoder.state_dict(),
                                    parent_checkpoint=str(args.checkpoint), step=step, arm=name, manifest=identity),
                               args.output / (name + '-final.pt'))
            check = gates(reports['A'], reports['B']) if not args.smoke else {}
            result = dict(step=step, reports=reports, gates=check)
            history.append(result); write(args.output / 'history.json', history)
            print(json.dumps(dict(evaluation=step, A=reports['A']['primary_first8_mm'],
                                  B=reports['B']['primary_first8_mm'], gates=check)), flush=True)
            gross = gross + 1 if reports['B']['primary_first8_mm'] > 1.25 * reports['A']['primary_first8_mm'] else 0
            if gross >= 2: raise RuntimeError('two consecutive gross regression checks')
        if time.time() - began > 1800: raise TimeoutError('30-minute total budget')
    differences = {key: float((value.detach().cpu().float() - packet['model'][key].float()).abs().max())
                   for key, value in model.state_dict().items() if not key.startswith('trajectory.')}
    if max(differences.values()) != 0: raise ValueError('frozen model state changed')
    outcome = 'ENGINEERING_ONLY' if args.smoke else ('PROMISING' if all(history[-1]['gates'].values()) else 'UNPROMISING')
    identity.update(status='COMPLETED', elapsed_s=time.time()-began)
    write(args.output / 'manifest.json', identity)
    write(args.output / 'result.json', dict(status='COMPLETED', outcome=outcome, elapsed_s=time.time()-began,
           initial=initial, final=history[-1], frozen_max_abs=max(differences.values()),
           weighted_draw_fraction=float(suspect[draws].mean()), gpu=gpu_snapshot(torch.device('cuda')),
           interpretation='One bounded decoder-only Probe; no conclusion on all curation, joint pretraining or robot success'))
    print(json.dumps(dict(status='COMPLETED', outcome=outcome, elapsed_s=time.time()-began)), flush=True)


if __name__ == '__main__': main()
