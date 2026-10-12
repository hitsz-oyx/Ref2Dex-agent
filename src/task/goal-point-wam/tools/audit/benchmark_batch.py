#!/usr/bin/env python3
"""Engineering-only real-batch throughput and memory preflight; no checkpoint."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'tools/run'))
from train_three_domain import (GoalPointWAM, GoalWindows, atomic_json, digest, gpu_snapshot,
                                initialize_pw, loader, mixed_indices, move_batch)
from oakink_wm.multisource import MixedWindows


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = json.loads(args.config.read_text())
    if args.fused_hilbert:
        from oakink_wm.pointworld_performance import install_fused_hilbert
        install_fused_hilbert()
    if args.vectorized_scene_sampling:
        from goal_point_wam.performance import install_vectorized_sampling
        install_vectorized_sampling()
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    base = MixedWindows(args.data, 'train')
    dataset = GoalWindows(base, goal_type='object_points', goal_points=config['goal_points'])
    indices = mixed_indices(base, config['microbatch'] * 6, config['seed'] + 1)
    model = GoalPointWAM(json.loads(args.stats.read_text()), config)
    initialize_pw(model, args.init_pw, config, digest(args.stats))
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for cpu in loader(dataset, indices, config['microbatch'], 0):
        batch = move_batch(cpu, torch.device('cuda'))
        torch.cuda.synchronize(); start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            pred = model(batch)
        loss, _ = model.loss(pred, batch)
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite benchmark loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'], error_if_nonfinite=True)
        optimizer.step(); torch.cuda.synchronize()
        rows.append(dict(compute_seconds=time.monotonic()-start, loss=float(loss),
                         valid_scene_points=int(batch['point_valid'].sum()), gpu=gpu_snapshot(torch.device('cuda'))))
    atomic_json(args.output, dict(engineering_only=True, microbatch=config['microbatch'], rows=rows,
                                  steady_compute_seconds=float(np.median([r['compute_seconds'] for r in rows[1:]])),
                                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                  peak_reserved_bytes=torch.cuda.max_memory_reserved()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fused-hilbert', action='store_true')
    parser.add_argument('--vectorized-scene-sampling', action='store_true')
    for name in ('data', 'stats', 'init-pw', 'config', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
