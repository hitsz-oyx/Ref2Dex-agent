#!/usr/bin/env python3
"""Freeze bounded three-source train-only physical loss normalization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/'src'))
from oakink_wm.multisource import MixedWindows, MAIN_SOURCE_NAMES, mixed_indices, sha
from oakink_wm.pointworld import capped_collate
from oakink_wm.model import rigid_points, geodesic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=2048)
    p.add_argument('--seconds', type=int, default=600)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = p.parse_args()
    args.output.resolve().relative_to((ROOT/'outputs/cm-pointflow-effect-pretrain').resolve())
    if args.output.exists(): raise FileExistsError('preserve frozen loss statistics')
    if not 1 <= args.samples <= 4096 or not 1 <= args.seconds <= 600:
        raise ValueError('statistics bounded to <=4096 windows and <=600 seconds')
    started = time.monotonic()
    torch.set_num_threads(2)
    dataset = MixedWindows(args.data, 'train')
    names = tuple(d['name'] for d in dataset.meta['sources'])
    if names != MAIN_SOURCE_NAMES: raise ValueError('loss statistics require three-source main dynamics')
    indices = mixed_indices(dataset, args.samples, 216)
    sums, squares, counts = {}, {}, {}
    def add(key, values):
        values = values.double()
        sums[key] = sums.get(key, 0) + values.sum(0)
        squares[key] = squares.get(key, 0) + values.square().sum(0)
        counts[key] = counts.get(key, 0) + len(values)
    device = torch.device(args.device)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=8, num_workers=2,
                        collate_fn=capped_collate)
    with torch.no_grad():
        for i, cpu in enumerate(loader):
            if time.monotonic()-started >= args.seconds: raise TimeoutError('loss statistics deadline')
            batch = {k: v.to(device) for k,v in cpu.items()}
            gt = batch['effect']
            future = rigid_points(gt[..., :3, :3], gt[..., :3, 3], batch['points'])
            flow = future-batch['points'][:, :, None]
            add('flow', flow[batch['object_valid']].permute(0, 2, 1, 3).reshape(-1, 24, 3))
            add('translation', gt[..., :3, 3][batch['object_valid']])
            rotation = geodesic(torch.eye(3, device=device).expand_as(gt[..., :3, :3]), gt[..., :3, :3])
            add('rotation', rotation[batch['object_valid']])
            if i % 32 == 0:
                print(json.dumps(dict(batch=i, windows=min((i+1)*8,args.samples))), flush=True)
    source_ids = np.searchsorted(dataset.offsets, indices, side='right')-1
    result = dict(schema='pointworld-wm24.loss-normalization.v1', role='loss_only', split='train',
                  seed=216, samples=args.samples, sources=list(names),
                  source_weights=dataset.source_weights.tolist(),
                  source_window_counts=np.bincount(source_ids, minlength=3).tolist(),
                  counts=counts, input_manifest_sha256=sha(args.data/'processed/manifest.json'),
                  index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                  device=args.device, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  device_reason='physical tensor statistics; no neural model inference',
                  elapsed_s=time.monotonic()-started, seconds_cap=args.seconds,
                  forward_statistics='unchanged checkpoint input/output buffers')
    for key in ('flow', 'translation'):
        mean = sums[key]/counts[key]
        std = (squares[key]/counts[key]-mean.square()).clamp_min(.001**2).sqrt()
        result[key+'_mean'] = mean.cpu().tolist()
        result[key+'_std'] = std.cpu().tolist()
    result['rotation_scale'] = (squares['rotation']/counts['rotation']).clamp_min(.02**2).sqrt().cpu().tolist()
    payload = json.dumps(result, indent=2)+'\n'
    if len(payload.encode()) > 1 << 20: raise RuntimeError('statistics output unexpectedly exceeds1MiB')
    if time.monotonic()-started >= args.seconds: raise TimeoutError('loss statistics deadline')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output: output.write(payload)
    print(json.dumps(dict(status='COMPLETED', output=str(args.output), elapsed_s=result['elapsed_s'])), flush=True)


if __name__ == '__main__': main()
