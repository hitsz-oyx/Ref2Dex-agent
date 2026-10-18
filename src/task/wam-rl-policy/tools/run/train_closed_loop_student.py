#!/usr/bin/env python3
"""Train a closed-loop student BC policy and save deployable parameters.

The student consumes either the partial observation alone (``p2``) or the
partial observation plus the oracle 8-step 11-point plan (``p2_tau``).  Both
arms use the same architecture, data, and schedule.  The checkpoint stores the
input normalization because the closed-loop runner must reproduce it exactly.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2,
                               default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', choices=('p2', 'p2_tau'), required=True)
    p.add_argument('--success-only', action='store_true')
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()
    if args.output == Path('/tmp') or Path('/tmp') in args.output.parents:
        raise ValueError('checkpoint must remain in the project workspace')
    with np.load(args.data / 'identification_data.npz', allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    if args.success_only:
        seeds = []
        for seed in np.unique(data['seed']):
            mask = data['seed'] == seed
            if data['success'][mask][-1] > 0.5:
                seeds.append(int(seed))
        keep = np.isin(data['seed'], seeds)
        data = {key: value[keep] for key, value in data.items()}
    else:
        seeds = [int(s) for s in np.unique(data['seed'])]
    x = data['p2']
    if args.variant == 'p2_tau':
        x = np.concatenate([x, data['tau'].reshape(len(x), -1)], axis=1)
    y = data['action'].astype(np.float32)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    device = f'cuda:{args.gpu}'
    torch.manual_seed(args.seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], args.hidden), torch.nn.ELU(),
        torch.nn.Linear(args.hidden, args.hidden), torch.nn.ELU(),
        torch.nn.Linear(args.hidden, 24)).to(device)
    xt = torch.tensor((x - mean) / std, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    started = time.monotonic()
    for _ in range(args.steps):
        optimizer.zero_grad()
        loss = F.smooth_l1_loss(model(xt), yt, beta=0.1)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = model(xt)
        train_l1 = float((prediction - yt).abs().mean().item())
        per_dim = (prediction - yt).abs().mean(dim=0).cpu().numpy()
        baseline = float((yt - yt.mean(dim=0, keepdim=True)).abs().mean().item())
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), mean=mean, std=std,
                    variant=args.variant, hidden=args.hidden, seed=args.seed,
                    input_dim=int(x.shape[1]), rows=int(len(x)), seeds=seeds),
               args.output / 'model.pt')
    write_json(args.output / 'metrics.json', dict(
        schema='ref2dex.unidex-closed-loop-student.v1',
        variant=args.variant, rows=int(len(x)), seeds=seeds,
        input_dim=int(x.shape[1]), hidden=args.hidden, steps=args.steps,
        seed=args.seed, train_l1=train_l1, mean_baseline_l1=baseline,
        per_dim_l1=per_dim.tolist(),
        xyz_l1=float(per_dim[0:3].mean()), rot_l1=float(per_dim[3:6].mean()),
        fingers_l1=float(per_dim[6:24].mean()),
        elapsed_s=time.monotonic() - started))
    print(json.dumps(dict(variant=args.variant, rows=int(len(x)), train_l1=train_l1,
                          xyz_l1=float(per_dim[0:3].mean()),
                          rot_l1=float(per_dim[3:6].mean()),
                          fingers_l1=float(per_dim[6:24].mean())), indent=2))


if __name__ == '__main__':
    main()
