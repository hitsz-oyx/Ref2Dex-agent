#!/usr/bin/env python3
"""Train or fine-tune a closed-loop student BC policy.

The student consumes either the partial observation alone (``p2``) or the
partial observation plus the oracle 8-step 11-point plan (``p2_tau``).  A
DAgger packet can be mixed in, and an existing checkpoint can be used as the
initialization so the input normalization stays identical to deployment.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from wam_rl_policy.student_inputs import student_features


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2,
                               default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


def build_model(input_dim, hidden):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden), torch.nn.ELU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ELU(),
        torch.nn.Linear(hidden, 24))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', choices=('p2', 'p2_tau', 'full', 'full_tau'), required=True)
    p.add_argument('--success-only', action='store_true')
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--init', type=Path)
    p.add_argument('--dagger-data', type=Path)
    p.add_argument('--dagger-weight', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--pad-to', type=int, default=0,
                   help='zero-pad a state-only input to this dimension for a capacity control')
    p.add_argument('--shuffle-tau', action='store_true',
                   help='row-shuffle the tau channel during BC and DAgger')
    p.add_argument('--tau-shuffle-seed', type=int, default=20261018)
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

    def features(packet):
        observation = (packet['full'] if args.variant in ('full', 'full_tau')
                       else packet['p2'])
        value = student_features(observation, args.variant,
                                 tau=packet.get('tau'), pad_to=args.pad_to)
        if args.shuffle_tau:
            if args.variant not in ('p2_tau', 'full_tau'):
                raise ValueError('shuffle-tau requires a tau variant')
            state_dim = observation.shape[1]
            state, tau_part = value[:, :state_dim], value[:, state_dim:]
            permutation = np.random.default_rng(args.tau_shuffle_seed).permutation(len(value))
            value = np.concatenate([state, tau_part[permutation]], axis=1)
        return value

    x_bc = features(data)
    y_bc = data['action'].astype(np.float32)
    x = x_bc
    y = y_bc
    dagger_rows = 0
    x_dagger = None
    y_dagger = None
    if args.dagger_data:
        with np.load(args.dagger_data / 'dagger_data.npz', allow_pickle=False) as source:
            packet = {key: source[key] for key in source.files}
        x_dagger = features(packet)
        y_dagger = packet['action'].astype(np.float32)
        if x_dagger.shape[1] != x_bc.shape[1]:
            raise ValueError('DAgger packet feature dimension mismatch')
        x = np.concatenate([x_bc] + [x_dagger] * args.dagger_weight, axis=0)
        y = np.concatenate([y_bc] + [y_dagger] * args.dagger_weight, axis=0)
        dagger_rows = int(len(x_dagger) * args.dagger_weight)
    device = f'cuda:{args.gpu}'
    torch.manual_seed(args.seed)
    if args.init:
        checkpoint = torch.load(args.init, map_location=device, weights_only=False)
        if checkpoint['variant'] != args.variant or checkpoint['input_dim'] != x.shape[1]:
            raise ValueError('init checkpoint contract mismatch')
        mean = checkpoint['mean']
        std = checkpoint['std']
        model = build_model(int(x.shape[1]), int(checkpoint['hidden'])).to(device)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        mean = x_bc.mean(axis=0)
        std = np.where(x_bc.std(axis=0) < 1e-6, 1.0, x_bc.std(axis=0))
        model = build_model(int(x.shape[1]), args.hidden).to(device)
    xt = torch.tensor((x - mean) / std, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    bc_t = torch.tensor((x_bc - mean) / std, dtype=torch.float32, device=device)
    bc_y = torch.tensor(y_bc, dtype=torch.float32, device=device)
    dagger_t = (torch.tensor((x_dagger - mean) / std, dtype=torch.float32, device=device)
                if x_dagger is not None else None)
    dagger_y = (torch.tensor(y_dagger, dtype=torch.float32, device=device)
                if y_dagger is not None else None)
    with torch.no_grad():
        before_bc = float((model(bc_t) - bc_y).abs().mean().item())
        before_dagger = (float((model(dagger_t) - dagger_y).abs().mean().item())
                         if dagger_t is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
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
        after_bc = float((model(bc_t) - bc_y).abs().mean().item())
        after_dagger = (float((model(dagger_t) - dagger_y).abs().mean().item())
                        if dagger_t is not None else None)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), mean=mean, std=std,
                    variant=args.variant, hidden=int(model[0].out_features), seed=args.seed,
                    input_dim=int(x.shape[1]), rows=int(len(x)), seeds=seeds,
                    pad_to=args.pad_to, shuffle_tau=bool(args.shuffle_tau),
                    init=str(args.init) if args.init else None,
                    dagger_data=str(args.dagger_data) if args.dagger_data else None),
               args.output / 'model.pt')
    write_json(args.output / 'metrics.json', dict(
        schema='ref2dex.unidex-closed-loop-student.v1',
        variant=args.variant, rows=int(len(x)), bc_rows=int(len(x_bc)),
        dagger_rows=dagger_rows, dagger_weight=args.dagger_weight,
        seeds=seeds, input_dim=int(x.shape[1]), steps=args.steps, lr=args.lr,
        pad_to=args.pad_to, shuffle_tau=bool(args.shuffle_tau),
        seed=args.seed, train_l1=train_l1,
        bc_l1_before=before_bc, bc_l1_after=after_bc,
        dagger_l1_before=before_dagger, dagger_l1_after=after_dagger,
        per_dim_l1=per_dim.tolist(),
        xyz_l1=float(per_dim[0:3].mean()), rot_l1=float(per_dim[3:6].mean()),
        fingers_l1=float(per_dim[6:24].mean()),
        init=str(args.init) if args.init else None,
        dagger_data=str(args.dagger_data) if args.dagger_data else None,
        elapsed_s=time.monotonic() - started))
    print(json.dumps(dict(variant=args.variant, rows=int(len(x)),
                          dagger_rows=dagger_rows, train_l1=train_l1,
                          bc_before=before_bc, bc_after=after_bc,
                          dagger_before=before_dagger, dagger_after=after_dagger,
                          xyz_l1=float(per_dim[0:3].mean()),
                          rot_l1=float(per_dim[3:6].mean()),
                          fingers_l1=float(per_dim[6:24].mean())), indent=2))


if __name__ == '__main__':
    main()
