#!/usr/bin/env python3
"""Matched current-image context and human future-shape sensor screens."""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from oakink_wm.raw_sensors import SensorWindows

ARMS = ('history', 'future_hand', 'history_rgb', 'future_hand_rgb', 'shuffled_future_hand_rgb', 'shuffled_rgb')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_file(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class Predictor(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, 128), nn.GELU(), nn.Linear(128, 128),
                                 nn.GELU(), nn.Linear(128, 24 * 512))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, observed_features):
        return self.net(observed_features).reshape(-1, 24, 512) * .05


def packs(dataset):
    samples = [dataset[i] for i in range(len(dataset))]
    arrays = {key: np.stack([s[key] for s in samples]) for key in
              ('sensor_history', 'hand_history', 'future_hand_shape', 'sensor_current', 'sensor_target', 'target_delta')}
    arrays['features'] = np.concatenate([arrays[k].reshape(len(samples), -1) for k in
        ('sensor_history', 'hand_history', 'future_hand_shape')], axis=1)
    # Donor from a DIFFERENT task, same fixed relative window index. No label use.
    groups = {}
    for i, (name, _) in enumerate(dataset.index):
        groups.setdefault(name, []).append(i)
    names = sorted(groups)
    if len(names) < 3:
        raise ValueError('fewer than3 tasks for matched task-separated shuffle')
    donors = np.zeros(len(samples), 'int64')
    for a, name in enumerate(names):
        other = groups[names[(a + 1) % len(names)]]
        for j, i in enumerate(groups[name]):
            donors[i] = other[j % len(other)]
    return arrays, groups, donors


def add_current_features(arrays, dataset, features):
    # The feature at each original current frame, never any future image.
    current = np.stack([features[name][start + 3] for name, start in dataset.index])
    if current.shape != (len(dataset), 512) or not np.isfinite(current).all():
        raise ValueError('current RGB feature schema differs')
    arrays['features'] = np.concatenate((arrays['features'], current), axis=1)


def variants(features, donors, future_begin, rgb_begin):
    result = {name: features.copy() for name in ARMS}
    result['history'][:, future_begin:] = 0
    result['future_hand'][:, rgb_begin:] = 0
    result['history_rgb'][:, future_begin:rgb_begin] = 0
    result['shuffled_future_hand_rgb'][:, future_begin:rgb_begin] = features[donors, future_begin:rgb_begin]
    result['shuffled_rgb'][:, rgb_begin:] = features[donors, rgb_begin:]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rgb-features', type=Path, required=True)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--max-seconds', type=float, default=300)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    np.random.seed(227)
    torch.manual_seed(227)
    torch.cuda.manual_seed_all(227)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest = json.loads((args.data / 'manifest.json').read_text())
    for e in manifest['records']:
        if sha(args.data / e['file']) != e['sha256']:
            raise ValueError('source hash drift')
    fit, held = SensorWindows(args.data, 'fit'), SensorWindows(args.data, 'held_task')
    if {name for name, _ in fit.index} & {name for name, _ in held.index}:
        raise ValueError('record leakage across partitions')
    if {str(Path(name).parent) for name, _ in fit.index} & {str(Path(name).parent) for name, _ in held.index}:
        raise ValueError('task leakage across partitions')
    train, groups, donors = packs(fit)
    dev, held_groups, held_donors = packs(held)
    future_begin = 4 * 512 + 4 * 2 * 21 * 3
    rgb_begin = train['features'].shape[1]
    feature_manifest = json.loads((args.rgb_features / 'manifest.json').read_text())
    if feature_manifest['status'] != 'COMPLETED_FRAME_LOCAL_RGB_FEATURES':
        raise ValueError('RGB features not qualified')
    if feature_manifest['source_label_manifest_sha256'] != manifest['source_manifest_sha256']:
        raise ValueError('RGB/sensor source bundle identity differs')
    image_features = {}
    for entry in feature_manifest['records']:
        if sha(args.rgb_features / entry['file']) != entry['sha256']:
            raise ValueError('RGB feature hash drift')
        with np.load(args.rgb_features / entry['file'], allow_pickle=False) as z:
            if not np.array_equal(z['source_frame_ids'], np.arange(entry['frames'])):
                raise ValueError('RGB frame identity differs')
            image_features[entry['record']] = z['features'].copy()
    for entry in manifest['records']:
        if image_features[entry['record']].shape != (entry['original_frames'], 512):
            raise ValueError('RGB/sensor frame count differs')
    add_current_features(train, fit, image_features)
    add_current_features(dev, held, image_features)
    mean, std = train['features'].mean(0), np.maximum(train['features'].std(0), .01)
    std[train['features'].std(0) < 1e-6] = 1
    active = np.asarray(manifest['train_dynamic_channels'], 'int64')
    left_mapping = json.loads((TASK.parents[2] / 'tmp/video-tactile-primary-sources/TouchAnything-configs__pressure_position_mapping_left.json').read_text())
    right_mapping = json.loads((TASK.parents[2] / 'tmp/video-tactile-primary-sources/TouchAnything-configs__pressure_position_mapping_right.json').read_text())
    group_a = sorted(set(left_mapping.values()) - set(range(208, 223))) + [i + 256 for i in sorted(set(right_mapping.values()) - set(range(33, 48)))]
    group_b = list(range(208, 223)) + list(range(256 + 33, 256 + 48))
    score_channels = dict(all_dynamic=active, normalization_group_a=np.intersect1d(active, group_a),
                          normalization_group_b=np.intersect1d(active, group_b))
    if min(len(v) for v in score_channels.values()) < 4:
        raise ValueError('score group support too small')
    tensors = {}
    for split, arrays, donation in [('fit', train, donors), ('held', dev, held_donors)]:
        features = (arrays['features'] - mean) / std
        inputs = variants(features, donation, future_begin, rgb_begin)
        tensors[split] = {name: torch.from_numpy(v).cuda() for name, v in inputs.items()}
        for key in ('sensor_current', 'sensor_target', 'target_delta'):
            tensors[split][key] = torch.from_numpy(arrays[key]).cuda()
    initial = Predictor(train['features'].shape[1]).cuda()
    models = {name: copy.deepcopy(initial) for name in ARMS}
    del initial
    opts = {name: torch.optim.AdamW(models[name].parameters(), lr=3e-4, weight_decay=.01) for name in ARMS}
    identity = dict(schema='ref2dex.rawsensor-visual-context.v1', seed=227,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        data_manifest_sha256=sha(args.data / 'manifest.json'), rgb_feature_manifest_sha256=sha(args.rgb_features/'manifest.json'),
        current_image_frame_rule='start+3; per-frame frozen ImageNet feature, no future RGB',
        encoder_identity=feature_manifest['encoder'], steps=args.steps,
        fit_windows=len(fit), held_windows=len(held), fit_tasks=list(groups), held_tasks=list(held_groups),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu=torch.cuda.get_device_name(),
        normalization=dict(split='FIT only', mean=mean.tolist(), std=std.tolist()),
        score_channels={k:v.tolist() for k,v in score_channels.items()},
        group_semantics='candidate processed normalization groups; hardware tactile/bend roles UNKNOWN',
        donor_rule='next sorted different task; modulo window index, fixed independently of sensor values',
        implementation_sha256={str(q.relative_to(TASK)):sha(q) for q in
            (Path(__file__).resolve(),TASK/'src/oakink_wm/raw_sensors.py')})
    json_file(args.output / 'input_manifest.json', identity)
    channel_tensor = torch.from_numpy(active).cuda()
    started, last_sample, status, step = time.monotonic(), -10., 'RUNNING', 0
    amp = torch.cuda.is_bf16_supported()
    try:
        with (args.output / 'progress.jsonl').open('w', buffering=1) as log:
            for step in range(1, args.steps + 1):
                if time.monotonic() - started > args.max_seconds:
                    status, step = 'STOPPED_BUDGET', step - 1
                    break
                # Matched draws, uniform task mass, four windows per fitting task.
                indices = np.concatenate([np.random.choice(rows, 4, replace=True) for rows in groups.values()])
                ix = torch.from_numpy(indices).cuda()
                losses = {}
                for name in ARMS:
                    opts[name].zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                        pred = models[name](tensors['fit'][name][ix])
                    target = tensors['fit']['target_delta'][ix]
                    loss = nn.functional.huber_loss(pred.float()[:, :, channel_tensor] / .02,
                        target[:, :, channel_tensor] / .02)
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite raw sensor loss')
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(models[name].parameters(), 1.)
                    if not torch.isfinite(norm):
                        raise ValueError('nonfinite raw sensor gradient')
                    opts[name].step()
                    losses[name] = float(loss)
                if step == 1 or step % 25 == 0 or step == args.steps:
                    elapsed = time.monotonic() - started
                    record = dict(step=step, loss=losses, elapsed_seconds=elapsed,
                        estimated_remaining_seconds=(args.steps-step)*elapsed/step,
                        peak_memory_mib=torch.cuda.max_memory_allocated()/1024**2)
                    if elapsed-last_sample>=10 or step==args.steps:
                        record['nvml_gpu_rows']=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()
                        last_sample=elapsed
                    log.write(json.dumps(record)+'\n')
                    print(json.dumps(record),flush=True)
            else:
                status='COMPLETED'
        torch.save(dict(models={name:m.state_dict() for name,m in models.items()},
                        optimizers={name:o.state_dict() for name,o in opts.items()},
                        step=step,identity=identity),args.output/'final.pt')
        restored=torch.load(args.output/'final.pt',map_location='cuda',weights_only=False)
        for name,m in models.items():
            if any(not torch.equal(v,restored['models'][name][k]) for k,v in m.state_dict().items()):
                raise ValueError('checkpoint roundtrip differs')
        del restored
        metrics={}
        with torch.no_grad():
            for split, partitions in [('fit',groups),('held',held_groups)]:
                predictions={}
                for name,m in models.items():
                    m.eval()
                    outputs=[]
                    for lo in range(0,len(tensors[split][name]),64):
                        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                            delta=m(tensors[split][name][lo:lo+64])
                        outputs.append((tensors[split]['sensor_current'][lo:lo+64,None]+delta.float()).clamp(0,1))
                    predictions[name]=torch.cat(outputs)
                predictions['persistence']=tensors[split]['sensor_current'][:,None].expand(-1,24,-1)
                for task,indices in partitions.items():
                    key=split+'/'+task
                    metrics[key]={}
                    for name,pred in predictions.items():
                        errors=(pred[indices]-tensors[split]['sensor_target'][indices]).abs().cpu().numpy()
                        metrics[key][name]={group:dict(mae=float(errors[:,:,channels].mean()),
                            h24_mae=float(errors[:,23,channels].mean()),windows=len(indices),channels=len(channels))
                            for group,channels in score_channels.items()}
        held_values=[v for k,v in metrics.items() if k.startswith('held/')]
        macro={name:float(np.mean([v[name]['normalization_group_a']['h24_mae'] for v in held_values]))
               for name in (*ARMS,'persistence')}
        def gate(arm, controls):
            return macro[arm] <= .9*min(macro[n] for n in controls) and all(
                v[arm]['normalization_group_a']['h24_mae'] <= 1.2*min(
                    v[n]['normalization_group_a']['h24_mae'] for n in controls) for v in held_values)
        context_passed = gate('history_rgb', ('history', 'persistence'))
        motion_passed = gate('future_hand_rgb', tuple(n for n in (*ARMS, 'persistence') if n != 'future_hand_rgb'))
        conclusion='PROMISING' if context_passed or motion_passed else 'UNPROMISING'
        if status!='COMPLETED':conclusion='UNCLEAR'
        json_file(args.output/'result.json',dict(status=status,steps=step,conclusion=conclusion,
            checkpoint_roundtrip=True,macro_group_a_h24_mae=macro,metrics=metrics,
            context_gate=context_passed, conditional_future_motion_gate=motion_passed,
            elapsed_seconds=time.monotonic()-started,peak_memory_mib=torch.cuda.max_memory_allocated()/1024**2,
            interpretation='current visual context and human future motion are separate gates; no calibrated pressure/contact/robot policy conclusion'))
    except Exception as exc:
        json_file(args.output/'failure.json',dict(error=str(exc),step=step,elapsed_seconds=time.monotonic()-started))
        raise


if __name__=='__main__':
    main()
