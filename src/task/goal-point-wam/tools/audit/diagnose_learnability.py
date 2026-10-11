#!/usr/bin/env python3
"""Train-only, fixed-window fitting diagnosis for the existing Goal-Point-WAM."""
import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'tools/run'))
from train_three_domain import (GoalPointWAM, GoalWindows, atomic_json, digest,
                                gpu_snapshot, move_batch, parameter_hash)
from goal_point_wam.data import goal_collate
from oakink_wm.multisource import MixedWindows, MAIN_SOURCE_NAMES
from oakink_wm.model import metrics, object_motion_masks


def tensor_digest(batch):
    h = hashlib.sha256()
    for name, value in sorted(batch.items()):
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


def choose_windows(base, selection_seed):
    """First qualifying windows from a seed-fixed train-only candidate order."""
    rng = np.random.default_rng(selection_seed)
    dataset = GoalWindows(base)
    samples, records = [], []
    for source_id, source in enumerate(base.sources):
        candidates = rng.choice(source.groups[0], min(64, len(source.groups[0])), replace=False)
        kept = 0
        for local_index in candidates:
            index = int(base.offsets[source_id] + local_index)
            sample = dataset[index]
            one = goal_collate([sample])
            static = dict(translation=torch.zeros_like(one['effect'][..., :3, 3]),
                          rotation=torch.eye(3).expand_as(one['effect'][..., :3, :3]))
            moving, _ = object_motion_masks(one)
            error = metrics(static, one)['point_epe'][..., -1]
            valid = sample['trajectory_valid']
            hand_motion = float(np.linalg.norm(sample['trajectory_target'], axis=-1)[valid].mean())
            if not moving.any() or float(error[moving].mean()) < .005 or hand_motion < .005:
                continue
            # Check against raw source coordinates, not just the adapter target.
            seq_id, anchor, tick, category = map(int, source.rows[local_index])
            raw = source.sequence(source.sequences[seq_id])
            current_anchor = np.linalg.inv(raw['poses'][tick, anchor])
            raw_hand = raw['hand'][tick:tick + 25].reshape(25, 2, 11, 3)
            hand = np.einsum('ij,thnj->thni', current_anchor[:3, :3], raw_hand)
            hand += current_anchor[:3, 3]
            action_valid = sample['action_valid'].reshape(24, 2, 11)
            start_valid = sample['start_hand_valid']
            action = sample['action'].reshape(24, 2, 11, 9)
            checks = dict(
                current_hand_max=float(np.abs(sample['start_hand'][start_valid] - hand[0][start_valid]).max()),
                future_hand_max=float(np.abs(action[..., :3][action_valid] - hand[1:][action_valid]).max()),
                current_delta_max=float(np.abs(action[..., 3:6][action_valid] - (hand[1:] - hand[0])[action_valid]).max()),
                step_delta_max=float(np.abs(action[..., 6:9][action_valid] - np.diff(hand, axis=0)[action_valid]).max()),
                trajectory_max=float(np.abs(sample['trajectory_target'][valid] - (hand[1:] - hand[0])[valid]).max()))
            if max(checks.values()) > 2e-5:
                raise ValueError('source/anchor/action target disagreement: ' + str(checks))
            selected = dict(source=dataset.source_names[source_id], mixed_index=index,
                            source_index=int(local_index), sample_id=sample['sample_id'].tolist(),
                            source_sequence=source.sequences[seq_id], anchor=anchor, tick=tick,
                            category=category, moving_objects=int(moving.sum()),
                            static_moving_epe_m_h24=float(error[moving].mean()),
                            static_trajectory_epe_m=hand_motion, raw_target_audit=checks)
            samples.append(sample); records.append(selected); kept += 1
            if kept == 2:
                break
        if kept != 2:
            raise ValueError('not enough qualifying fixed train windows: ' + dataset.source_names[source_id])
    batch = goal_collate(samples)
    for name, value in batch.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('nonfinite fixed batch: ' + name)
    return batch, records


@torch.no_grad()
def measure(model, batch):
    was_training = model.training
    model.eval()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        pred = model(batch)
    losses, terms = model.loss(pred, batch)
    if not torch.isfinite(losses):
        raise FloatingPointError('nonfinite evaluation loss')
    error = metrics(pred, batch)['point_epe'][..., -1]
    moving, _ = object_motion_masks(batch)
    trajectory = torch.linalg.vector_norm(pred['trajectory'] - batch['trajectory_target'], dim=-1)
    result = dict(loss=float(losses), terms={k: float(v) for k, v in terms.items()}, sources={})
    for source_id, name in enumerate(MAIN_SOURCE_NAMES):
        rows = batch['source_id'] == source_id
        valid = batch['trajectory_valid'][rows]
        world_mask = moving[rows]
        result['sources'][name] = dict(trajectory_epe_m=float(trajectory[rows][valid].mean()),
            moving_world_epe_m_h24=float(error[rows][world_mask].mean()))
    model.train(was_training)
    return result


def gradient_audit(model, batch):
    """Compare branch gradients on a shared parameter without optimizer mutation."""
    was_training = model.training
    model.eval()
    pred = model(batch)
    trajectory = model.trajectory_loss(pred['trajectory'], batch)
    world, _ = model.world_loss(pred, batch)
    shared = model.scene.scene_proj[0].weight
    grads = [torch.autograd.grad(loss, shared, retain_graph=True)[0].flatten()
             for loss in (trajectory, world)]
    norms = [float(g.norm()) for g in grads]
    if any(not math.isfinite(v) or v == 0 for v in norms):
        raise RuntimeError('missing/nonfinite branch gradient')
    cosine = float(torch.dot(*grads) / (grads[0].norm() * grads[1].norm()))
    oracle = dict(trajectory=batch['trajectory_target'],
                  translation=batch['effect'][..., :3, 3], rotation=batch['effect'][..., :3, :3])
    _, oracle_terms = model.loss(oracle, batch)
    if float(oracle_terms['trajectory']) != 0 or float(oracle_terms['world_point']) != 0:
        raise ValueError('oracle target does not minimize trajectory/point loss')
    model.train(was_training)
    return dict(trajectory_shared_gradient_norm=norms[0], world_shared_gradient_norm=norms[1],
                shared_gradient_cosine=cosine, oracle_terms={k: float(v) for k, v in oracle_terms.items()})


def fit_arm(state, stats, batch, learning_rate, args, output, deadline):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = GoalPointWAM(stats, state['config']).cuda()
    model.load_state_dict(state['model'], strict=True)
    # Inference-mode layers, with autograd enabled, remove dropout/order noise.
    # This measures fixed-input learnability, not the historical train recipe.
    model.train(args.forward_mode == 'train')
    initial_hash = parameter_hash(model)
    initial = measure(model, batch)
    audit = gradient_audit(model, batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    output.mkdir()
    started = time.monotonic()
    log = (output / 'train.jsonl').open('w')
    step = 0
    try:
        for step in range(1, args.updates + 1):
            if model.training != (args.forward_mode == 'train'):
                raise RuntimeError('diagnostic measurement changed the fit mode')
            if time.time() > deadline:
                raise TimeoutError('fixed-batch budget exhausted')
            optimizer.zero_grad(set_to_none=True)
            scale = min(1., step / 20.)
            for group in optimizer.param_groups: group['lr'] = learning_rate * scale
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred = model(batch)
            loss, terms = model.loss(pred, batch)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite fit loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            row = dict(step=step, loss=float(loss), gradient_norm=float(norm),
                       elapsed_seconds=time.monotonic() - started,
                       terms={k: float(v) for k,v in terms.items()})
            if step == 1 or step % 25 == 0 or step == args.updates:
                row['gpu'] = gpu_snapshot(torch.device('cuda'))
                row['measured'] = measure(model, batch)
                atomic_json(args.output / 'progress.json', dict(status='RUNNING', arm=output.name, **row))
                print(json.dumps(dict(arm=output.name, **row)), flush=True)
            log.write(json.dumps(row)+'\n'); log.flush()
        final = measure(model, batch)
        ratios = {name: {key: final['sources'][name][key] / initial['sources'][name][key]
                         for key in initial['sources'][name]} for name in MAIN_SOURCE_NAMES}
        passed = sum(max(values.values()) <= .5 for values in ratios.values()) >= 2
        # Save model-only once per arm; no redundant optimizer snapshots.
        torch.save(dict(model=model.state_dict(), config=state['config'], step=step,
                        diagnosis_only=True, parent_checkpoint_sha256=digest(args.checkpoint)), output / 'final.pt')
        result = dict(status='COMPLETED', step=step, learning_rate=learning_rate,
                      initial_parameter_sha256=initial_hash, initial=initial, final=final,
                      ratios=ratios, fit_gate=passed, gradient_audit=audit,
                      elapsed_seconds=time.monotonic()-started, gpu=gpu_snapshot(torch.device('cuda')))
        atomic_json(output / 'result.json', result)
        return result
    finally:
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--stats', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=247)
    parser.add_argument('--selection-seed', type=int, default=251)
    parser.add_argument('--updates', type=int, default=300)
    parser.add_argument('--budget-seconds', type=float, default=900)
    parser.add_argument('--forward-mode', choices=('eval', 'train'), default='eval')
    parser.add_argument('--only-learning-rate', type=float, choices=(.0001, .001))
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    deadline = time.time() + args.budget_seconds
    try:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if state.get('checkpoint_kind') != 'goal-point-wam.single.v1': raise ValueError('wrong checkpoint kind')
        if state['identity']['stats_sha256'] != digest(args.stats): raise ValueError('stats drift')
        if state['identity']['data_manifest_sha256'] != digest(args.data / 'processed/manifest.json'):
            raise ValueError('data drift')
        if state['identity']['model_sha256'] != digest(TASK / 'src/goal_point_wam/model.py'):
            raise ValueError('model drift')
        stats = json.loads(args.stats.read_text())
        base = MixedWindows(args.data, 'train')
        if tuple(x['name'] for x in base.meta['sources']) != MAIN_SOURCE_NAMES:
            raise ValueError('expected frozen three-source train corpus')
        cpu_batch, windows = choose_windows(base, args.selection_seed)
        torch.save(cpu_batch, args.output / 'fixed_train_batch.pt')
        identity = dict(schema='goal-point-wam.fixed-train-diagnosis.v1', split='train',
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            script_sha256=digest(Path(__file__)), parent_checkpoint=str(args.checkpoint.resolve()),
            parent_checkpoint_sha256=digest(args.checkpoint), parent_step=state['step'],
            data_manifest_sha256=state['identity']['data_manifest_sha256'],
            stats_sha256=state['identity']['stats_sha256'], windows=windows,
            batch_sha256=tensor_digest(cpu_batch), seed=args.seed, selection_seed=args.selection_seed,
            updates=args.updates, budget_seconds=args.budget_seconds,
            forward_mode=args.forward_mode, precision='BF16', batch_mode='fixed full six-window batch',
            learning_rates=[args.only_learning_rate] if args.only_learning_rate else [.0001, .001],
            conditional_second_arm='only if lr1e-4 fails fit gate')
        atomic_json(args.output / 'input_manifest.json', identity)
        batch = move_batch(cpu_batch, torch.device('cuda'))
        results = {}
        first_rate = args.only_learning_rate or .0001
        first_name = 'lr1e-3' if first_rate == .001 else 'lr1e-4'
        results[first_name] = fit_arm(state, stats, batch, first_rate, args, args.output / first_name, deadline)
        if not args.only_learning_rate and not results[first_name]['fit_gate']:
            torch.cuda.empty_cache()
            results['lr1e-3'] = fit_arm(state, stats, batch, .001, args, args.output / 'lr1e-3', deadline)
            if results['lr1e-3']['initial_parameter_sha256'] != results['lr1e-4']['initial_parameter_sha256']:
                raise RuntimeError('arm initialization mismatch')
        final = dict(status='COMPLETED', diagnosis_only=True, arms=results,
                     conclusion='PROMISING' if any(r['fit_gate'] for r in results.values()) else 'UNCLEAR')
        atomic_json(args.output / 'result.json', final)
        atomic_json(args.output / 'progress.json', final)
        print(json.dumps(final), flush=True)
    except BaseException as exc:
        atomic_json(args.output / 'progress.json', dict(status='FAILED', error=repr(exc)))
        raise


if __name__ == '__main__': main()
