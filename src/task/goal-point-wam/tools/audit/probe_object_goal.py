#!/usr/bin/env python3
"""Train-only object-goal contract, fixed-window fit and frozen response Probe."""
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
from train_three_domain import (GoalPointWAM, GoalWindows, atomic_json, digest,
                                gpu_snapshot, move_batch, parameter_hash)
from diagnose_learnability import gradient_audit, measure, tensor_digest
from goal_point_wam.data import goal_collate
from goal_point_wam.pretrained import initialize_pw
from oakink_wm.data import transform_points
from oakink_wm.multisource import MixedWindows, MAIN_SOURCE_NAMES


def prepare_batch(base, window_manifest, point_count):
    dataset = GoalWindows(base, goal_type='object_points', goal_points=point_count)
    samples, records = [], []
    for record in window_manifest['windows']:
        index, sid = int(record['mixed_index']), MAIN_SOURCE_NAMES.index(record['source'])
        source = base.sources[sid]
        local = index - int(base.offsets[sid])
        seq, anchor, tick, category = map(int, source.rows[local])
        if [seq, anchor, tick] != record['sample_id'] or source.sequences[seq] != record['source_sequence']:
            raise ValueError('fixed window identity drift')
        sample = dataset[index]
        raw = source.sequence(source.sequences[seq])
        canonical, _, _, _ = source.canonical(raw['objects'][anchor])
        ids = np.linspace(0, len(canonical) - 1, point_count, dtype=np.int64)
        frame = np.linalg.inv(raw['poses'][tick, anchor])
        start = transform_points(canonical[ids], frame @ raw['poses'][tick, anchor])
        goal = transform_points(canonical[ids], frame @ raw['poses'][tick + 24, anchor])
        errors = {name: float(np.abs(sample[key] - actual).max()) for name, key, actual in (
            ('raw_start_max_m', 'start_object_points', start),
            ('raw_goal_max_m', 'goal_object_points', goal))}
        # Future-hand positions only supervise trajectory/world execution; they
        # are absent from this goal contract. Recheck trajectory labels raw.
        hand = transform_points(raw['hand'][tick:tick + 25].reshape(25, 2, 11, 3), frame)
        valid = sample['trajectory_valid']
        errors['raw_trajectory_max_m'] = float(np.abs(
            sample['trajectory_target'][valid] - (hand[1:] - hand[0])[valid]).max())
        if max(errors.values()) > 2e-5:
            raise ValueError('raw current-frame object/hand target disagreement')
        samples.append(sample)
        records.append(dict(record, object_goal_slot=int(sample['goal_object_slot']),
                            object_goal_raw_audit=errors,
                            object_endpoint_rms_m=float(np.square(goal - start).mean() ** .5)))
    batch = goal_collate(samples)
    if any(not torch.isfinite(v).all() for v in batch.values() if v.is_floating_point()):
        raise ValueError('nonfinite object-goal batch')
    if 'goal_hand' in batch:
        raise ValueError('hand endpoint leaked into object-goal batch')
    return batch, records


@torch.no_grad()
def response_audit(model, batch):
    """Identical scene/action, symmetric object endpoint translations in X."""
    was_training = model.training
    model.eval()
    full = model(batch)
    repeat = model(batch)
    raw_repeat = max(float((full[k] - repeat[k]).abs().max()) for k in ('translation', 'rotation'))
    original_forward = model.scene.forward
    scene = original_forward(batch)
    model.scene.forward = lambda _: scene
    shifts = []
    try:
        reference = model(batch)
        for amount in (.01, -.01, .1, -.1):
            changed = dict(batch)
            changed['goal_object_points'] = batch['goal_object_points'].clone()
            changed['goal_object_points'][..., 0] += batch['object_goal_valid'].float() * amount
            output = model(changed)
            delta = output['trajectory'] - reference['trajectory']
            mask = batch['trajectory_valid'][:, -1]
            rows = []
            for i in range(len(mask)):
                rows.append(float(delta[i, -1, ..., 0][mask[i]].mean()) / amount)
            sources = {}
            for sid, name in enumerate(MAIN_SOURCE_NAMES):
                ids = batch['source_id'] == sid
                gains = np.asarray(rows)[ids.cpu().numpy()]
                sources[name] = dict(mean_endpoint_x_gain=float(gains.mean()),
                                     median_endpoint_x_gain=float(np.median(gains)))
            isolation = max(float((output[k] - reference[k]).abs().max()) for k in ('translation', 'rotation'))
            shifts.append(dict(shift_m=amount, endpoint_x_gain_by_window=rows, sources=sources,
                endpoint_response_rms_m=float(delta[:, -1][mask].square().mean().sqrt()),
                trajectory_response_rms_m=float(delta[batch['trajectory_valid']].square().mean().sqrt()),
                cached_scene_world_isolation_max=isolation))
        no_motion = dict(batch, goal_object_points=batch['start_object_points'])
        static_prediction = model(no_motion)
        original_error = torch.linalg.vector_norm(reference['trajectory'] - batch['trajectory_target'], dim=-1)
        no_goal_error = torch.linalg.vector_norm(static_prediction['trajectory'] - batch['trajectory_target'], dim=-1)
        valid = batch['trajectory_valid']
        ablation = dict(real_goal_trajectory_epe_m=float(original_error[valid].mean()),
                        static_object_goal_trajectory_epe_m=float(no_goal_error[valid].mean()))
    finally:
        model.scene.forward = original_forward
        model.train(was_training)
    strong_sources = [name for name in MAIN_SOURCE_NAMES if all(
        row['sources'][name]['median_endpoint_x_gain'] >= .1
        for row in shifts if abs(row['shift_m']) == .1)]
    isolated = all(row['cached_scene_world_isolation_max'] <= 1e-5 for row in shifts)
    return dict(diagnosis_only=True, split='train', shifts=shifts,
                unchanged_whole_forward_world_repeat_max=raw_repeat,
                static_object_goal_ablation=ablation, strong_response_sources=strong_sources,
                response_gate=len(strong_sources) >= 2, cached_scene_isolation_pass=isolated,
                scope='six train windows; object translation is not a labeled physical counterfactual')


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(2)
    config = json.loads(args.config.read_text())
    if config.get('goal_type') != 'object_points':
        raise ValueError('object point goal config required')
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    base = MixedWindows(args.data, 'train')
    if tuple(x['name'] for x in base.meta['sources']) != MAIN_SOURCE_NAMES:
        raise ValueError('expected frozen main-three train corpus')
    stats = json.loads(args.stats.read_text())
    old = json.loads(args.windows.read_text())
    if old['data_manifest_sha256'] != digest(args.data / 'processed/manifest.json') or old['stats_sha256'] != digest(args.stats):
        raise ValueError('data/stats drift from fixed-window diagnosis')
    cpu_batch, records = prepare_batch(base, old, config['goal_points'])
    torch.save(cpu_batch, args.output / 'fixed_train_batch.pt')
    batch = move_batch(cpu_batch, torch.device('cuda'))
    model = GoalPointWAM(stats, config)
    transfer = initialize_pw(model, args.init_pw, config, digest(args.stats))
    identity = dict(schema='goal-point-wam.object-goal-probe.v1', split='train', diagnosis_only=True,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        script_sha256=digest(Path(__file__)), model_sha256=digest(TASK / 'src/goal_point_wam/model.py'),
        adapter_sha256=digest(TASK / 'src/goal_point_wam/data.py'), config_sha256=digest(args.config),
        instruction_sha256=digest(TASK / 'docs/user/ref/ref1_1.md'),
        data_manifest_sha256=digest(args.data / 'processed/manifest.json'), stats_sha256=digest(args.stats),
        window_manifest_sha256=digest(args.windows), windows=records, batch_sha256=tensor_digest(cpu_batch),
        initial_parameter_sha256=parameter_hash(model), pretrained=transfer, seed=config['seed'],
        updates=args.updates, goal_type='object_points', goal_points=config['goal_points'],
        forward_mode='train', learning_rate=config['learning_rate'], warmup=20,
        budget_seconds=args.budget_seconds, device=gpu_snapshot(torch.device('cuda')))
    atomic_json(args.output / 'input_manifest.json', identity)
    model.cuda().train()
    initial = measure(model, batch)
    initial_response = response_audit(model, batch)
    atomic_json(args.output / 'initial_response.json', initial_response)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    with (args.output / 'train.jsonl').open('w') as log:
        for step in range(1, args.updates + 1):
            if time.monotonic() - started > args.budget_seconds:
                raise TimeoutError('object-goal fitting budget exhausted')
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = config['learning_rate'] * min(1., step / 20.)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred = model(batch)
            loss, terms = model.loss(pred, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite object-goal fitting loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'], error_if_nonfinite=True)
            optimizer.step()
            row = dict(step=step, loss=float(loss), terms={k: float(v) for k,v in terms.items()},
                       gradient_norm=float(norm), elapsed_seconds=time.monotonic() - started)
            if step == 1 or step % 25 == 0 or step == args.updates:
                row['gpu'] = gpu_snapshot(torch.device('cuda'))
                atomic_json(args.output / 'progress.json', dict(status='RUNNING', **row))
                print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + '\n'); log.flush()
    final = measure(model, batch)
    audit = gradient_audit(model, batch)
    responses = response_audit(model, batch)
    atomic_json(args.output / 'goal_response.json', responses)
    ratios = {name: {key: final['sources'][name][key] / initial['sources'][name][key]
                     for key in initial['sources'][name]} for name in MAIN_SOURCE_NAMES}
    fit_gate = sum(max(row.values()) <= .5 for row in ratios.values()) >= 2
    torch.save(dict(checkpoint_kind='goal-point-wam.single.v1', model=model.state_dict(),
                    config=config, step=args.updates, identity=identity, diagnosis_only=True), args.output / 'final.pt')
    result = dict(status='COMPLETED', initial=initial, final=final, ratios=ratios,
        fit_gate=fit_gate, response_gate=responses['response_gate'], gradient_audit=audit,
        cached_scene_isolation_pass=responses['cached_scene_isolation_pass'],
        conclusion='PROMISING' if fit_gate and responses['response_gate'] else 'UNCLEAR',
        elapsed_seconds=time.monotonic() - started, gpu=gpu_snapshot(torch.device('cuda')))
    atomic_json(args.output / 'result.json', result)
    atomic_json(args.output / 'progress.json', result)
    print(json.dumps(result), flush=True)
    if not responses['cached_scene_isolation_pass']:
        raise RuntimeError('object goal leaked into world branch at identical scene')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'stats', 'init-pw', 'config', 'windows', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--updates', type=int, default=300)
    parser.add_argument('--budget-seconds', type=float, default=900)
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error('updates must be positive')
    output_existed = args.output.exists()
    try:
        run(args)
    except BaseException as exc:
        if not output_existed and args.output.is_dir() and not (args.output / 'result.json').exists():
            atomic_json(args.output / 'progress.json', dict(status='FAILED', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
