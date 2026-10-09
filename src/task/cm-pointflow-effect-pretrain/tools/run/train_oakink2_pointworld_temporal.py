#!/usr/bin/env python3
"""Time-preserving PointWorld WM24 training; isolated from the frozen V1 runs."""
import argparse
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from oakink_wm.data import Windows, balanced_indices, shuffle_action
from oakink_wm.model import metrics, object_motion_masks
from oakink_wm.pointworld_temporal import model_from_config, capped_collate as collate, VENDOR


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_numerics():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True


def implementation_sources():
    return {str(f.relative_to(TASK)): digest(f) for f in (
        Path(__file__).resolve(), TASK/'tools/run/evaluate_oakink2_pointworld_temporal.py',
        TASK/'src/oakink_wm/pointworld_temporal.py', TASK/'src/oakink_wm/pointworld.py',
        TASK/'src/oakink_wm/model.py', TASK/'src/oakink_wm/data.py')}


def atomic_json(path, data):
    temp = path.with_suffix('.json.part')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def device_batch(batch): return {key: value.cuda(non_blocking=True) for key, value in batch.items()}


def shuffled_with_donors(batch, dataset, seed):
    shuffled, available = shuffle_action(batch, seed)
    if not available.all():
        rng = np.random.default_rng(seed)
        for i in np.flatnonzero(~available):
            wanted = batch['hand_presence'][i].cpu().numpy()
            recipient = batch['sample_id'][i].cpu().numpy()
            for attempt in range(1000):
                donor = dataset[int(rng.integers(len(dataset)))]
                if np.array_equal(donor['hand_presence'], wanted) and not np.array_equal(donor['sample_id'], recipient):
                    shuffled['action'][i] = torch.from_numpy(donor['action']).to(batch['action'].device)
                    shuffled['action_valid'][i] = torch.from_numpy(donor['action_valid']).to(batch['action'].device)
                    break
            else: raise RuntimeError('No valid same-hand donor for singleton group')
    return shuffled


def loader(dataset, indices, batch_size, workers=0):
    generator = torch.Generator().manual_seed(17)
    return DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=False,
                      collate_fn=collate, num_workers=workers, pin_memory=True,
                      persistent_workers=workers > 0, generator=generator)


# The selection strata are complementary object populations.  Both use
# category -1 (all categories) so that the macro score cannot mix a target
# anchor numerator with an all-object static denominator.
MOVING_SELECTION_KEY = 'model/moving_objects/cat-1/h24/point_epe'
STATIC_SELECTION_KEY = 'model/static_objects/cat-1/h24/point_epe'


def validation_selection(measured):
    """Select checkpoints with a fixed moving/static macro score.

    Keep the two physical components visible in every validation record.  A
    missing stratum is a protocol error rather than an implicit fallback to a
    moving-only score.
    """
    keys = (MOVING_SELECTION_KEY, STATIC_SELECTION_KEY)
    if any(key not in measured for key in keys):
        missing = [key for key in keys if key not in measured]
        raise ValueError('validation selection strata missing: ' + ', '.join(missing))
    moving, static = (float(measured[key]) for key in keys)
    if not all(math.isfinite(value) and value >= 0 for value in (moving, static)):
        raise ValueError('validation selection metrics must be finite and nonnegative')
    return dict(score=(moving + static) / 2,
                moving_objects_h24=moving, static_objects_h24=static,
                moving_key=MOVING_SELECTION_KEY, static_key=STATIC_SELECTION_KEY)


@torch.no_grad()
def evaluate(model, dataset, indices, arm, batch_size, amp, intervention=False):
    model.eval()
    totals, counts = {}, {}
    for number, cpu in enumerate(loader(dataset, indices, batch_size)):
        batch = device_batch(cpu)
        use = shuffled_with_donors(batch, dataset, 9000 + number) if arm == 'shuffle' or intervention else batch
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp): pred = model(use, arm)
        views = dict(model=pred, static=dict(translation=torch.zeros_like(pred['translation']),
                     rotation=torch.eye(3, device='cuda').expand_as(pred['rotation'])))
        for name, result in views.items():
            values = metrics(result, batch)
            moving_objects, static_objects = object_motion_masks(batch)
            scopes = {'scene': batch['object_valid'],
                      'anchor': batch['object_valid'] & (batch['object_features'][..., 13] > .5),
                      'moving_objects': moving_objects,
                      'static_objects': static_objects}
            for scope, scope_mask in scopes.items():
                for category in (-1, 0, 1, 2):
                    selected = scope_mask & ((batch['category'] == category)[:, None] if category >= 0 else True)
                    count = int(selected.sum())
                    if not count: continue
                    for horizon in (1, 4, 8, 12, 24):
                        for metric, value in values.items():
                            key = '%s/%s/cat%s/h%s/%s' % (name, scope, category, horizon, metric)
                            totals[key] = totals.get(key, 0.) + float(value[..., horizon-1][selected].sum())
                            counts[key] = counts.get(key, 0) + count
    model.train()
    return {key: totals[key] / counts[key] for key in totals}


def save_checkpoint(path, model, optimizer, step, config, dataset_hash, best, identity):
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, config=config,
                 dataset_hash=dataset_hash, best=best, identity=identity,
                 torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                 numpy_rng=np.random.get_state(), python_rng=random.getstate())
    part = path.with_suffix('.pt.part')
    torch.save(state, part)
    part.replace(path)


def run(args):
    stop_requested = [False]
    signal.signal(signal.SIGUSR1, lambda signum, frame: stop_requested.__setitem__(0, True))
    config = json.loads(args.config.read_text())
    if args.steps is not None: config['updates'] = args.steps
    if args.smoke:
        config.update(microbatch=2, accumulation=1, workers=0, validation_samples=12, validation_interval=3, checkpoint_interval=3)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'input_manifest.json').exists() and not args.resume:
        raise FileExistsError('existing run requires explicit resume or a fresh output directory')
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('existing run config drift')
    atomic_json(config_path, config)
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    configure_numerics()
    train, val = Windows(args.data, 'train'), Windows(args.data, 'val')
    dataset_hash = digest(args.data / 'processed/manifest.json')
    stats = json.loads(args.stats.read_text())
    if stats['split'] != 'train' or stats['input_manifest_sha256'] != dataset_hash:
        raise ValueError('normalization identity mismatch')
    model = model_from_config(stats, config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    identity = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    dataset_hash=dataset_hash, config_hash=digest(config_path), arm=args.arm, smoke=args.smoke,
                    parameters=sum(p.numel() for p in model.parameters()), device=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                    script_sha256=digest(Path(__file__)), model_sha256=digest(TASK / 'src/oakink_wm/pointworld_temporal.py'),
                    data_sha256=digest(TASK / 'src/oakink_wm/data.py'), architecture_sha256=digest(TASK / 'docs/user/ref/ref4.md'),
                    stats_sha256=digest(args.stats), stats=stats,
                    pointworld_commit=subprocess.check_output(['git', '-C', str(VENDOR), 'rev-parse', 'HEAD'], text=True).strip(),
                    vendor_sources={str(f.relative_to(VENDOR)): digest(f) for f in (VENDOR/'ptv3').rglob('*') if f.suffix in ('.py','.yaml')})
    identity['implementation_sources'] = implementation_sources()
    atomic_json(out / 'input_manifest.json', identity)
    initial_hash = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    step, best = 0, float('inf')
    if args.resume:
        state = torch.load(args.resume, map_location='cuda', weights_only=False)
        if (state['dataset_hash'] != dataset_hash or state['config'] != config or state['identity']['arm'] != args.arm
                or state['identity']['stats_sha256'] != identity['stats_sha256']
                or state['identity'].get('implementation_sources') != identity['implementation_sources']
                or state['identity']['vendor_sources'] != identity['vendor_sources']):
            raise ValueError('checkpoint identity mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        step, best = state['step'], state['best']
        torch.set_rng_state(state['torch_rng'].cpu()); torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
    identity['initial_parameter_sha256'] = initial_hash
    atomic_json(out / 'input_manifest.json', identity)
    effective = config['microbatch'] * config['accumulation']
    indices = balanced_indices(train, config['updates'] * effective, seed + 1)
    validation = balanced_indices(val, config['validation_samples'], config['validation_seed'])
    natural = np.random.default_rng(config['natural_validation_seed']).choice(len(val), config['validation_samples'], replace=len(val)<config['validation_samples'])
    np.save(out / 'validation_balanced.npy', validation); np.save(out / 'validation_natural.npy', natural)
    batches = iter(loader(train, indices[step*effective:], config['microbatch'], config['workers']))
    start = time.time()
    deadline = args.deadline if args.deadline else start + config['group_seconds']
    best_metrics = None
    optimizer.zero_grad(set_to_none=True)
    trainlog = (out / 'train.jsonl').open('a')
    try:
        while step < config['updates'] and time.time() < deadline and not stop_requested[0]:
            update_start = time.monotonic()
            if step < config['warmup_updates']: scale = (step+1) / config['warmup_updates']
            else: scale = .1 + .9 * .5 * (1 + math.cos(math.pi * (step-config['warmup_updates']) / max(1, config['updates']-config['warmup_updates'])))
            for group in optimizer.param_groups: group['lr'] = config['learning_rate'] * scale
            acc_loss = 0.
            for micro in range(config['accumulation']):
                batch = device_batch(next(batches))
                use = shuffled_with_donors(batch, train, seed + step * config['accumulation'] + micro) if args.arm == 'shuffle' else batch
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config['amp']):
                    pred = model(use, args.arm)
                # Physical loss/rotation projection are float32 outside autocast.
                loss, components = model.loss(pred, batch)
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                (loss / config['accumulation']).backward()
                acc_loss += float(loss.detach()) / config['accumulation']
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'], error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            step += 1
            row = dict(step=step, loss=acc_loss, gradient_norm=float(grad), seconds=time.monotonic()-update_start,
                       elapsed_seconds=time.time()-start, learning_rate=optimizer.param_groups[0]['lr'])
            trainlog.write(json.dumps(row)+'\n'); trainlog.flush()
            if step == 1 or step % 20 == 0 or args.smoke: print(json.dumps(row), flush=True)
            atomic_json(out / 'progress.json', dict(status='RUNNING', **row))
            if (step % config['validation_interval'] == 0 or step == config['updates']) and time.time() < deadline and not stop_requested[0]:
                measured = evaluate(model, val, validation, args.arm, config['microbatch'], config['amp'])
                selection = validation_selection(measured)
                score = selection['score']
                atomic_json(out / 'validation_latest.json',
                            dict(step=step, metrics=measured, selection=selection))
                if score < best:
                    best, best_metrics = score, measured
                    save_checkpoint(out / 'best.pt', model, optimizer, step, config, dataset_hash, best, identity)
            if step % config['checkpoint_interval'] == 0:
                save_checkpoint(out / 'latest.pt', model, optimizer, step, config, dataset_hash, best, identity)
        save_checkpoint(out / 'latest.pt', model, optimizer, step, config, dataset_hash, best, identity)
        save_checkpoint(out / 'final.pt', model, optimizer, step, config, dataset_hash, best, identity)
        final = dict(status='COMPLETED' if step == config['updates'] else 'BUDGET_STOP', step=step,
                     elapsed_seconds=time.time()-start, stop_requested=stop_requested[0],
                     best_validation_score=best if math.isfinite(best) else None)
        if best_metrics is not None:
            final['best_selection'] = validation_selection(best_metrics)
        if time.time() < deadline and not stop_requested[0]:
            final['natural'] = evaluate(model, val, natural, args.arm, config['microbatch'], config['amp'])
            final['balanced'] = evaluate(model, val, validation, args.arm, config['microbatch'], config['amp'])
            if args.arm == 'action':
                final['validation_shuffle'] = evaluate(model, val, validation, args.arm, config['microbatch'], config['amp'], True)
        else:
            final['metrics_deferred_due_budget'] = True
        if args.smoke:
            state = torch.load(out / 'final.pt', map_location='cpu', weights_only=False)
            for key, value in model.state_dict().items():
                if not torch.equal(value.cpu(), state['model'][key]): raise AssertionError('checkpoint mismatch: ' + key)
            final['checkpoint_roundtrip_exact'] = True
            final['engineering_only'] = True
        atomic_json(out / 'result.json', final)
        atomic_json(out / 'progress.json', {k: v for k, v in final.items() if k not in ('natural', 'balanced', 'validation_shuffle')})
        print(json.dumps({k: v for k, v in final.items() if k not in ('natural', 'balanced', 'validation_shuffle')}), flush=True)
    except BaseException as exc:
        atomic_json(out / 'progress.json', dict(status='FAILED', step=step, error=str(exc)))
        raise
    finally: trainlog.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--config', type=Path, default=TASK / 'configs/pointworld_temporal_wm24.json')
    p.add_argument('--stats', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=['history', 'action', 'shuffle'], required=True)
    p.add_argument('--steps', type=int)
    p.add_argument('--deadline', type=float)
    p.add_argument('--resume', type=Path)
    p.add_argument('--smoke', action='store_true')
    run(p.parse_args())


if __name__ == '__main__': main()
