#!/usr/bin/env python3
"""One-GPU bounded history-only video-point dynamics Decision Probe."""
import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from oakink_wm.video_points import VideoWindows, collate_video, balanced_flow_loss, flow_metrics
from oakink_wm.video_pointworld import VideoPointWorldWM
from oakink_wm.pointworld import VENDOR


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    part = path.with_suffix('.json.part')
    part.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    part.replace(path)


def observed(batch):
    return {key: batch[key] for key in ('xyz', 'features', 'point_valid')}


def device_batch(samples):
    return {key: value.cuda() for key, value in collate_video(samples).items()}


@torch.no_grad()
def evaluate(model, dataset, amp, horizon=24):
    model.eval()
    totals = {}
    clip_totals = {}
    for i in range(0, len(dataset), 2):
        batch = device_batch([dataset[j] for j in range(i, min(i + 2, len(dataset)))])
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            prediction = model(observed(batch))
        controls = dict(model=prediction, static=torch.zeros_like(prediction),
                         cv=batch['velocity'][:, :, None] * batch['elapsed'][:, None, :, None])
        for name, pred in controls.items():
            for key, (value, count) in flow_metrics(pred, batch, horizon).items():
                pair = totals.setdefault(name + '/' + key, [0., 0])
                pair[0] += value
                pair[1] += count
            for row in range(len(batch['xyz'])):
                scene = dataset.index[i + row][0]
                single = {k: v[row:row + 1] for k, v in batch.items()}
                for key, (value, count) in flow_metrics(pred[row:row + 1], single, horizon).items():
                    pair = clip_totals.setdefault(scene, {}).setdefault(name + '/' + key, [0., 0])
                    pair[0] += value
                    pair[1] += count
    model.train()
    result = {key: dict(epe_m=value / count if count else None, supported_point_horizons=count)
            for key, (value, count) in totals.items()}
    result['_per_clip'] = {scene: {key: dict(epe_m=value / count if count else None,
        supported_point_horizons=count) for key, (value, count) in metrics.items()}
        for scene, metrics in clip_totals.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--max-seconds', type=float, default=900)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--horizon', type=int, choices=(8, 24), default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(226)
    np.random.seed(226)
    torch.manual_seed(226)
    torch.cuda.manual_seed_all(226)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    stop = [False]
    signal.signal(signal.SIGUSR1, lambda *_: stop.__setitem__(0, True))
    manifest = json.loads((args.data / 'manifest.json').read_text())
    if manifest['protocol'].get('supervised_horizon', 24) != args.horizon:
        raise ValueError('training horizon differs from corpus qualification')
    for entry in manifest['sequences']:
        if sha(args.data / entry['file']) != entry['sha256']:
            raise ValueError('video source pack hash drift')
    if {e['video'] for e in manifest['sequences'] if e['split'] == 'train'} & {
       e['video'] for e in manifest['sequences'] if e['split'] == 'dev'}:
        raise ValueError('source video leakage across partitions')
    train, dev = VideoWindows(args.data, 'train'), VideoWindows(args.data, 'dev')
    features = np.concatenate([train[i]['features'] for i in range(len(train))])
    mean, std = features.mean(0), np.maximum(features.std(0), 1e-3)
    # Missing normals and constant categorical fields remain identity-scaled.
    std[features.std(0) < 1e-6] = 1.
    model = VideoPointWorldWM(mean, std).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    identity = dict(schema='ref2dex.video-point-training.v1', seed=226,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    data_manifest_sha256=sha(args.data / 'manifest.json'),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    gpu=torch.cuda.get_device_name(), steps=args.steps, smoke=args.smoke,
                    supervised_horizon=args.horizon,
                    normalization=dict(split='train', scene_mean=mean.tolist(), scene_std=std.tolist()),
                    train_windows=len(train), dev_windows=len(dev),
                    implementation_sha256={str(p.relative_to(TASK)): sha(p) for p in
                        (Path(__file__).resolve(), TASK / 'src/oakink_wm/video_points.py',
                         TASK / 'src/oakink_wm/video_pointworld.py', TASK / 'src/oakink_wm/pointworld.py',
                         TASK / 'src/oakink_wm/model.py')},
                    vendor_sources_sha256={str(p.relative_to(VENDOR)): sha(p)
                         for p in (VENDOR / 'ptv3').rglob('*') if p.suffix in ('.py', '.yaml')})
    atomic_json(args.output / 'input_manifest.json', identity)
    amp = torch.cuda.is_bf16_supported()
    started = time.monotonic()
    last_gpu_sample = -10.
    status, step = 'RUNNING', 0
    try:
        if not args.smoke:
            atomic_json(args.output / 'initial-development.json', evaluate(model, dev, amp, args.horizon))
        with (args.output / 'progress.jsonl').open('w', buffering=1) as log:
            for step in range(1, args.steps + 1):
                if stop[0] or time.monotonic() - started > args.max_seconds:
                    step -= 1
                    status = 'STOPPED_BUDGET_OR_SIGNAL'
                    break
                indices = np.random.randint(len(train), size=2)
                batch = device_batch([train[int(i)] for i in indices])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    pred = model(observed(batch))
                loss = balanced_flow_loss(pred, batch)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite video loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise ValueError('nonfinite video gradient')
                optimizer.step()
                if step == 1 or step % 10 == 0 or step == args.steps:
                    elapsed = time.monotonic() - started
                    record = dict(step=step, loss=float(loss), grad_norm=float(norm),
                                  elapsed_seconds=elapsed, updates_per_second=step / elapsed,
                                  estimated_remaining_seconds=(args.steps - step) * elapsed / step,
                                  peak_memory_mib=torch.cuda.max_memory_allocated() / 1024 ** 2)
                    if elapsed - last_gpu_sample >= 10. or step == args.steps:
                        record['nvml_gpu_rows'] = subprocess.check_output(
                            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                             '--format=csv,noheader,nounits'], text=True).strip().splitlines()
                        last_gpu_sample = elapsed
                    log.write(json.dumps(record) + '\n')
                    print(json.dumps(record), flush=True)
                if not args.smoke and step % 100 == 0:
                    atomic_json(args.output / ('development-step-%04d.json' % step), evaluate(model, dev, amp, args.horizon))
            else:
                status = 'COMPLETED'
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                          identity=identity, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                          numpy_rng=np.random.get_state(), python_rng=random.getstate())
        part = args.output / 'final.pt.part'
        torch.save(checkpoint, part)
        part.replace(args.output / 'final.pt')
        loaded = torch.load(args.output / 'final.pt', map_location='cuda', weights_only=False)
        if any(not torch.equal(value, loaded['model'][key]) for key, value in model.state_dict().items()):
            raise ValueError('checkpoint roundtrip changed model tensors')
        final = evaluate(model, dev, amp, args.horizon)
        atomic_json(args.output / 'final-development.json', final)
        suffix = 'h%d' % args.horizon
        obj = [final[name + '/object/' + suffix]['epe_m'] for name in ('model', 'static', 'cv')]
        bg = [final[name + '/background/' + suffix]['epe_m'] for name in ('model', 'static')]
        macro = None
        if any(x is None for x in obj + bg):
            conclusion = 'UNCLEAR'
        else:
            conclusion = 'PROMISING' if obj[0] <= .9 * min(obj[1:]) and bg[0] <= 1.2 * bg[1] else 'UNPROMISING'
        if args.horizon == 8:
            qualified = [v for v in final['_per_clip'].values()
                         if v['model/object/h8']['supported_point_horizons'] >= 16]
            if len(qualified) < 2:
                conclusion = 'UNCLEAR'
            else:
                learned = [v['model/object/h8']['epe_m'] for v in qualified]
                controls = [min(v['static/object/h8']['epe_m'], v['cv/object/h8']['epe_m']) for v in qualified]
                macro = dict(model_object_epe_m=float(np.mean(learned)),
                             best_control_object_epe_m=float(np.mean(controls)),
                             qualified_clips=len(qualified))
                passed = (np.mean(learned) <= .9 * np.mean(controls)
                          and all(a <= 1.2 * b for a, b in zip(learned, controls))
                          and bg[0] is not None and bg[1] is not None and bg[0] <= 1.2 * bg[1])
                conclusion = 'PROMISING' if passed else 'UNPROMISING'
        atomic_json(args.output / 'result.json', dict(status=status, steps=step, conclusion=conclusion,
                    elapsed_seconds=time.monotonic() - started, checkpoint_roundtrip=True,
                    metrics=final, smoke=args.smoke, clip_macro=macro,
                    qualification='engineering only' if args.smoke else 'single-seed weak-video Probe'))
    except Exception as exc:
        atomic_json(args.output / 'failure.json', dict(error=str(exc), step=step,
                    elapsed_seconds=time.monotonic() - started))
        raise


if __name__ == '__main__':
    main()
