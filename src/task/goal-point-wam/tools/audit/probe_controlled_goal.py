#!/usr/bin/env python3
"""Engineering diagnosis with analytic multi-goal labels; no physical claim."""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'tools/run'))
from train_three_domain import (GoalPointWAM, atomic_json, digest, gpu_snapshot,
                                move_batch, parameter_hash)
from diagnose_learnability import tensor_digest
from oakink_wm.multisource import MAIN_SOURCE_NAMES


def controlled_batch(batch, shift):
    """Artificial equivariant teacher, deliberately separate from world labels.

    Same observed scene, translated object endpoint, hand target plus t/T shift.
    World action/effect are left untouched and receive no training loss here.
    """
    if shift.shape != (len(batch['source_id']), 3):
        raise ValueError('expected one 3D shift per scene')
    changed = dict(batch)
    changed['goal_object_points'] = batch['goal_object_points'] + shift[:, None] * batch['object_goal_valid'][..., None]
    phase = torch.arange(1, 25, device=shift.device, dtype=shift.dtype) / 24
    target = batch['trajectory_target'] + phase[None, :, None, None, None] * shift[:, None, None, None]
    changed['trajectory_target'] = torch.where(batch['trajectory_valid'][..., None], target, torch.zeros_like(target))
    return changed


def trajectory(model, scene, batch):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        goal, mask = model.goal(batch['start_object_points'], batch['goal_object_points'], batch['object_goal_valid'])
        return model.trajectory(scene['scene_tokens'], scene['scene_mask'], goal, mask)


@torch.no_grad()
def evaluate(model, scene, batch):
    model.goal.eval(); model.trajectory.eval()
    base = trajectory(model, scene, batch)
    # Fixed interpolation queries: not the random training offsets. These are
    # held analytic inputs within the same scenes, not held-out physical data.
    offsets = []
    for magnitude in (.025, .075):
        for axis in range(3):
            for sign in (-1, 1):
                xyz = [0., 0., 0.]; xyz[axis] = magnitude * sign
                offsets.append(xyz)
    offsets += [[.025, -.025, .025], [-.075, .025, -.025]]
    sums = {name: dict(trajectory_sum=0., blind_sum=0., count=0, endpoint_sum=0., endpoint_count=0, gains=[])
            for name in MAIN_SOURCE_NAMES}
    for xyz in offsets:
        shift = base.new_tensor(xyz)[None].expand(len(base), -1)
        changed = controlled_batch(batch, shift)
        pred = trajectory(model, scene, changed)
        error = torch.linalg.vector_norm(pred - changed['trajectory_target'], dim=-1)
        blind = torch.linalg.vector_norm(base - changed['trajectory_target'], dim=-1)
        displacement = pred[:, -1] - base[:, -1]
        response_error = torch.linalg.vector_norm(displacement - shift[:, None, None], dim=-1)
        for sid, name in enumerate(MAIN_SOURCE_NAMES):
            rows = batch['source_id'] == sid
            valid = batch['trajectory_valid'][rows]
            endpoint = valid[:, -1]
            row = sums[name]
            row['trajectory_sum'] += float(error[rows][valid].sum())
            row['blind_sum'] += float(blind[rows][valid].sum())
            row['count'] += int(valid.sum())
            row['endpoint_sum'] += float(response_error[rows][endpoint].sum())
            row['endpoint_count'] += int(endpoint.sum())
            projected = (displacement[rows] * shift[rows, None, None]).sum(-1) / shift[rows].square().sum(-1)[:, None, None]
            row['gains'].append(float(projected[endpoint].mean()))
    sources = {}
    for name, row in sums.items():
        epe = row['trajectory_sum'] / row['count']
        blind = row['blind_sum'] / row['count']
        gain = float(np.median(row['gains']))
        endpoint = row['endpoint_sum'] / row['endpoint_count']
        sources[name] = dict(analytic_trajectory_epe_m=epe, goal_blind_epe_m=blind,
                            error_ratio_vs_blind=epe / blind, endpoint_response_error_m=endpoint,
                            median_directional_endpoint_gain=gain,
                            gate=epe <= .005 and endpoint <= .005 and .75 <= gain <= 1.25 and epe / blind <= .25)
    model.goal.train(); model.trajectory.train()
    return dict(engineering_only=True, sources=sources, gate=sum(x['gate'] for x in sources.values()) >= 2,
                offsets_m=offsets, scope='analytic translation teacher on six fixed scenes; no physical validation')


def fit_arm(state, stats, scene, batch, shifts, args, scale, output, deadline, encoding='endpoint'):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    config = dict(state['config'], goal_coordinate_scale_m=scale, object_goal_encoding=encoding)
    model = GoalPointWAM(stats, config).cuda().eval()
    model.load_state_dict(state['model'], strict=True)
    for p in model.parameters(): p.requires_grad_(False)
    for module in (model.goal, model.trajectory):
        module.train()
        for p in module.parameters(): p.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    initial_hash = parameter_hash(model)
    initial = evaluate(model, scene, batch)
    output.mkdir()
    optimizer = torch.optim.AdamW(trainable, lr=.001, weight_decay=.01)
    started = time.monotonic()
    history = []
    with (output / 'train.jsonl').open('w') as log:
        for step, shift in enumerate(shifts, 1):
            if time.monotonic() > deadline:
                raise TimeoutError('controlled-goal diagnostic budget exhausted')
            changed = controlled_batch(batch, shift)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups: group['lr'] = .001 * min(1., step / 20.)
            pred = trajectory(model, scene, changed)
            loss = model.trajectory_loss(pred, changed)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite analytic loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            optimizer.step()
            row = dict(step=step, loss=float(loss), gradient_norm=float(grad), elapsed_seconds=time.monotonic() - started)
            if step == 1 or step % 100 == 0 or step == len(shifts):
                row['gpu'] = gpu_snapshot(torch.device('cuda'))
                row['measured'] = evaluate(model, scene, batch)
                history.append(dict(step=step, measured=row['measured']))
                atomic_json(args.output / 'progress.json', dict(status='RUNNING', arm=output.name, **row))
                print(json.dumps(dict(arm=output.name, **row)), flush=True)
            log.write(json.dumps(row) + '\n'); log.flush()
    final = evaluate(model, scene, batch)
    # Keep the frozen-world invariance audit separate from artificial labels.
    original = model.scene.forward
    model.scene.forward = lambda _: scene
    try:
        model.eval()
        with torch.no_grad():
            real = model(batch)
            altered = model(controlled_batch(batch, torch.ones_like(shifts[0]) * .1))
        isolation = max(float((real[k] - altered[k]).abs().max()) for k in ('translation', 'rotation'))
    finally:
        model.scene.forward = original
    if isolation > 1e-5: raise RuntimeError('cached scene world goal isolation failed')
    # These artificial-teacher weights must never initialize real training.
    torch.save(dict(checkpoint_kind='goal-point-wam.analytic-diagnosis.v1', diagnosis_only=True,
                    artificial_labels=True, model=model.state_dict(), config=config, step=len(shifts)), output / 'final.pt')
    result = dict(status='COMPLETED', engineering_only=True, coordinate_scale_m=scale, object_goal_encoding=encoding,
                  initial_parameter_sha256=initial_hash, trainable_parameters=sum(p.numel() for p in trainable),
                  initial=initial, final=final, cached_scene_world_isolation_max=isolation,
                  history=history, elapsed_seconds=time.monotonic() - started, gpu=gpu_snapshot(torch.device('cuda')))
    atomic_json(output / 'result.json', result)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    state = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
    if state['config'].get('goal_type') != 'object_points': raise ValueError('object-goal parent required')
    if state['identity']['stats_sha256'] != digest(args.stats): raise ValueError('stats mismatch')
    stats = json.loads(args.stats.read_text())
    cpu_batch = torch.load(args.batch, weights_only=False, map_location='cpu')
    batch = move_batch(cpu_batch, torch.device('cuda'))
    if 'goal_hand' in batch: raise ValueError('hand endpoint goal fields in object diagnostic')
    generator = torch.Generator().manual_seed(args.offset_seed)
    shifts = torch.empty(args.updates, len(batch['source_id']), 3).uniform_(-.1, .1, generator=generator)
    torch.save(shifts, args.output / 'train_offsets.pt')
    cache_model = GoalPointWAM(stats, state['config']).cuda().eval()
    cache_model.load_state_dict(state['model'], strict=True)
    with torch.no_grad(): scene = {k: v.detach() for k, v in cache_model.scene(batch).items()}
    del cache_model
    scene_hash = tensor_digest({k: v.cpu() for k, v in scene.items()})
    identity = dict(schema='goal-point-wam.controlled-goal.v1', engineering_only=True, artificial_labels=True,
        split='train', git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        script_sha256=digest(Path(__file__)), model_sha256=digest(TASK / 'src/goal_point_wam/model.py'),
        parent_checkpoint_sha256=digest(args.checkpoint), parent_identity=state['identity'],
        batch_sha256=digest(args.batch), offsets_sha256=digest(args.output / 'train_offsets.pt'),
        scene_cache_sha256=scene_hash, seed=args.seed, offset_seed=args.offset_seed,
        updates=args.updates, lr=.001, warmup_updates=20, max_offset_m=.1, world_loss_used=False,
        synthetic_weights_for_real_training=False, compare_encoding=args.compare_encoding,
        budget_seconds=args.budget_seconds)
    atomic_json(args.output / 'input_manifest.json', identity)
    deadline = time.monotonic() + args.budget_seconds
    if args.compare_encoding:
        results = {name: fit_arm(state, stats, scene, batch, shifts.cuda(), args, .05, args.output / name, deadline, name)
                   for name in ('endpoint', 'start_delta')}
        if results['endpoint']['initial_parameter_sha256'] != results['start_delta']['initial_parameter_sha256']:
            raise RuntimeError('controlled arms have different initial parameters')
    else:
        results = {'raw': fit_arm(state, stats, scene, batch, shifts.cuda(), args, 1., args.output / 'raw', deadline)}
    if not args.compare_encoding and not results['raw']['final']['gate']:
        results['scaled'] = fit_arm(state, stats, scene, batch, shifts.cuda(), args, .05, args.output / 'scaled', deadline)
        if results['scaled']['initial_parameter_sha256'] != results['raw']['initial_parameter_sha256']:
            raise RuntimeError('controlled arms have different initial parameters')
    final = dict(status='COMPLETED', engineering_only=True, arms=results,
                 any_conditional_learning_gate=any(x['final']['gate'] for x in results.values()))
    atomic_json(args.output / 'result.json', final)
    atomic_json(args.output / 'progress.json', final)
    print(json.dumps(final), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'batch', 'stats', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--updates', type=int, default=600)
    parser.add_argument('--seed', type=int, default=247)
    parser.add_argument('--offset-seed', type=int, default=252)
    parser.add_argument('--budget-seconds', type=float, default=900)
    parser.add_argument('--compare-encoding', action='store_true', help='matched endpoint vs start_delta, both at .05m scale')
    args = parser.parse_args()
    if args.updates <= 0: parser.error('updates must be positive')
    existed = args.output.exists()
    try: run(args)
    except BaseException as exc:
        if not existed and args.output.is_dir():
            atomic_json(args.output / 'progress.json', dict(status='FAILED', error=repr(exc)))
        raise


if __name__ == '__main__': main()
