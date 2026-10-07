#!/usr/bin/env python3
"""Freeze train-only per-horizon flow/effect and input normalization statistics."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))
from oakink_wm.data import Windows, balanced_indices
from oakink_wm.pointworld import capped_collate
from oakink_wm.model import rigid_points, geodesic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=4096)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError('preserve frozen statistics')
    torch.set_num_threads(2)
    dataset = Windows(args.data, 'train')
    indices = balanced_indices(dataset, args.samples, 216)
    sums, squares, counts = {}, {}, {}
    def add(key, values):
        values = values.double()
        sums[key] = sums.get(key, 0) + values.sum(0)
        squares[key] = squares.get(key, 0) + values.square().sum(0)
        counts[key] = counts.get(key, 0) + len(values)
    dl = DataLoader(Subset(dataset, indices.tolist()), batch_size=8, num_workers=2,
                    collate_fn=capped_collate)
    with torch.no_grad():
        for i, cpu in enumerate(dl):
            b = {k: v.cuda() for k,v in cpu.items()}
            gt = b['effect']
            future = rigid_points(gt[..., :3, :3], gt[..., :3, 3], b['points'])
            flow = future-b['points'][:, :, None]
            add('flow', flow[b['object_valid']].permute(0, 2, 1, 3).reshape(-1, 24, 3))
            add('translation', gt[..., :3, 3][b['object_valid']])
            angles = geodesic(torch.eye(3, device='cuda').expand_as(gt[..., :3, :3]), gt[..., :3, :3])
            add('rotation', angles[b['object_valid']])
            add('scene', b['features'][b['point_valid']])
            add('action', b['action'][b['action_valid']])
            if i % 64 == 0: print(json.dumps(dict(batch=i, windows=min((i+1)*8,args.samples))), flush=True)
    result = dict(schema='pointworld-wm24.normalization.v1', split='train', seed=216,
                  samples=args.samples, input_manifest_sha256=hashlib.sha256((args.data/'processed/manifest.json').read_bytes()).hexdigest(),
                  index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(), device=os.environ.get('CUDA_VISIBLE_DEVICES'))
    for key in ('flow', 'translation', 'scene', 'action'):
        mean = sums[key]/counts[key]
        floor = .001 if key in ('flow','translation') else .01
        std = (squares[key]/counts[key]-mean.square()).clamp_min(floor**2).sqrt()
        result[key+'_mean'] = mean.cpu().tolist()
        result[key+'_std'] = std.cpu().tolist()
    result['rotation_scale'] = (squares['rotation']/counts['rotation']).clamp_min(.02**2).sqrt().cpu().tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status='COMPLETED', output=str(args.output))), flush=True)


if __name__ == '__main__': main()
