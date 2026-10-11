#!/usr/bin/env python3
"""Frozen, frame-local RGB features; downstream access is original current frame."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rgb', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-seconds', type=float, default=180)
    args = p.parse_args()
    source = json.loads((args.rgb / 'manifest.json').read_text())
    if source['status'] != 'COMPLETED_RGB_CLOCK_AUDIT' or not all(r['frame_identity_consistent'] for r in source['records']):
        raise ValueError('unqualified frame correspondence')
    weight_sha = sha(args.weights)
    if not weight_sha.startswith('f37072fd'):
        raise ValueError('local ResNet18 V1 weight identity differs')
    for r in source['records']:
        if sha(args.rgb / 'raw' / r['record'] / 'chest.mp4') != r['sha256']:
            raise ValueError('RGB source drift')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True), strict=True)
    model.fc = torch.nn.Identity()
    model.cuda().eval().requires_grad_(False)
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    amp = torch.cuda.is_bf16_supported()
    identity = dict(schema='ref2dex.egotouch-frame-rgb-features.v1', status='RUNNING', records=[],
        source_rgb_manifest_sha256=sha(args.rgb / 'manifest.json'), source_rgb_root=str(args.rgb.resolve()),
        source_label_manifest_sha256=source['source_label_manifest_sha256'],
        encoder='frozen ImageNet1K V1 ResNet18; 512-D penultimate feature; external ImageNet pretraining',
        weights_sha256=weight_sha, script_sha256=sha(__file__),
        preprocessing=repr(transform), gpu=torch.cuda.get_device_name(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        temporal_access='each feature uses ONLY its own original RGB frame; downstream window current is start+3',
        max_seconds=args.max_seconds, output_cap_bytes=16 * 1024**2)
    started, frames_done, output_bytes, last_sample = time.monotonic(), 0, 0, -10.
    try:
        with (args.output / 'progress.jsonl').open('w', buffering=1) as log, torch.inference_mode():
            for r in source['records']:
                cap = cv2.VideoCapture(str(args.rgb / 'raw' / r['record'] / 'chest.mp4'))
                features, batch = [], []
                while True:
                    ok, bgr = cap.read()
                    if ok:
                        batch.append(np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)))
                    if batch and (len(batch) == 64 or not ok):
                        if time.monotonic() - started > args.max_seconds:
                            cap.release()
                            raise RuntimeError('feature extraction deadline')
                        pixels = torch.from_numpy(np.stack(batch)).cuda()
                        pixels = transform(pixels)
                        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                            out = model(pixels)
                        if not torch.isfinite(out).all():
                            raise ValueError('nonfinite image feature')
                        features.append(out.float().cpu().numpy())
                        frames_done += len(batch)
                        batch = []
                        elapsed = time.monotonic() - started
                        if elapsed - last_sample >= 10:
                            progress = dict(frames=frames_done, elapsed_seconds=elapsed,
                                estimated_remaining_seconds=(sum(x['decoded_frames'] for x in source['records'])-frames_done)*elapsed/frames_done,
                                peak_memory_mib=torch.cuda.max_memory_allocated()/1024**2,
                                nvml_gpu_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip().splitlines())
                            log.write(json.dumps(progress)+'\n')
                            print(json.dumps(progress), flush=True)
                            last_sample = elapsed
                    if not ok:
                        break
                cap.release()
                array = np.concatenate(features)
                if array.shape != (r['decoded_frames'], 512):
                    raise ValueError('original feature/frame count mismatch')
                dest = args.output / (r['record'].replace('/', '__') + '.npz')
                np.savez_compressed(dest, features=array, source_frame_ids=np.arange(len(array), dtype='int64'))
                output_bytes += dest.stat().st_size
                if output_bytes > 16 * 1024**2:
                    raise RuntimeError('feature output byte cap')
                identity['records'].append(dict(record=r['record'], file=dest.name,
                    sha256=sha(dest), frames=len(array), video_sha256=r['sha256']))
            identity['status'] = 'COMPLETED_FRAME_LOCAL_RGB_FEATURES'
    except Exception as exc:
        identity['status'], identity['error'] = 'FAILED', str(exc)
        raise
    finally:
        identity.update(frames=frames_done, output_bytes=output_bytes,
            elapsed_seconds=time.monotonic()-started, peak_memory_mib=torch.cuda.max_memory_allocated()/1024**2)
        (args.output / 'manifest.json').write_text(json.dumps(identity, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
