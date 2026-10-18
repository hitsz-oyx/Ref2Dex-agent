#!/usr/bin/env python3
"""Matched identification: does a live oracle tau substitute for hidden state?

Trains identical MLPs on the same teacher rows with different executor
observations:

* full  : official 300-D observation (upper reference)
* p1    : robot core only (DOF q/dq + previous action)
* p2    : robot core + fingertip kinematics + hand pose (no object/contact)
* p*_tau: partial observation plus the 8-step, 11-point oracle plan
* p*_shuf: same inputs but tau permuted across rows (control)

Splits are grouped by teacher rollout seed, so validation uses unseen object
orientations and trajectories.  Reported errors are native action L1.
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
    path.write_text(json.dumps(value, indent=2))


def make_model(dim, hidden, seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(dim, hidden), torch.nn.ELU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ELU(),
        torch.nn.Linear(hidden, 24))


def normalize_fit(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def train_eval(x_train, y_train, x_val, y_val, seed, hidden, steps, device):
    mean, std = normalize_fit(x_train)
    xt = torch.tensor((x_train - mean) / std, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    xv = torch.tensor((x_val - mean) / std, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)
    model = make_model(x_train.shape[1], hidden, seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.smooth_l1_loss(model(xt), yt, beta=0.1)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_pred = model(xt)
        val_pred = model(xv)
    train_l1 = (train_pred - yt).abs().cpu().numpy()
    val_l1 = (val_pred - yv).abs().cpu().numpy()
    return train_l1, val_l1


def phase_metrics(l1, ticks):
    phases = {
        'approach': ticks < 20,
        'contact': (ticks >= 20) & (ticks < 45),
        'lift': ticks >= 45,
    }
    out = {}
    for name, mask in phases.items():
        if mask.any():
            out[name] = dict(rows=int(mask.sum()),
                             all=float(l1[mask].mean()),
                             xyz=float(l1[mask][:, 0:3].mean()),
                             rot=float(l1[mask][:, 3:6].mean()),
                             fingers=float(l1[mask][:, 6:24].mean()))
    return out


def group_metrics(l1, action_std):
    return dict(
        all=float(l1.mean()),
        xyz=float(l1[:, 0:3].mean()),
        rot=float(l1[:, 3:6].mean()),
        fingers=float(l1[:, 6:24].mean()),
        normalized=float((l1 / np.maximum(action_std, 0.1)).mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--steps', type=int, default=800)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--folds', type=int, default=4)
    p.add_argument('--success-only', action='store_true')
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()
    device = f'cuda:{args.gpu}'
    with np.load(args.data / 'identification_data.npz', allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    seed_values = np.unique(data['seed'])
    if args.success_only:
        keep = []
        for seed in seed_values:
            mask = data['seed'] == seed
            if data['success'][mask][-1] > 0.5:
                keep.append(seed)
        seed_values = np.asarray(keep, dtype=data['seed'].dtype)
        rows = np.isin(data['seed'], seed_values)
        data = {key: value[rows] for key, value in data.items()}
    if len(seed_values) < args.folds:
        raise ValueError('not enough teacher rollouts for grouped folds')
    tau = data['tau'].reshape(data['tau'].shape[0], -1)
    rng = np.random.default_rng(20261018)
    tau_shuffled = tau[rng.permutation(len(tau))]
    variants = {
        'full': data['full'],
        'p1': data['p1'],
        'p1_tau': np.concatenate([data['p1'], tau], axis=1),
        'p1_shuffle': np.concatenate([data['p1'], tau_shuffled], axis=1),
        'p2': data['p2'],
        'p2_tau': np.concatenate([data['p2'], tau], axis=1),
        'p2_shuffle': np.concatenate([data['p2'], tau_shuffled], axis=1),
        'p3': data['p3'],
        'p3_tau': np.concatenate([data['p3'], tau], axis=1),
        'p3_shuffle': np.concatenate([data['p3'], tau_shuffled], axis=1),
    }
    if 'tau_rot' in data:
        tau_rot = data['tau_rot'].reshape(data['tau_rot'].shape[0], -1)
        tau_rot_shuffled = tau_rot[
            np.random.default_rng(20261019).permutation(len(tau_rot))]
        variants.update({
            'p1_rot': np.concatenate([data['p1'], tau_rot], axis=1),
            'p1_tau_rot': np.concatenate([data['p1'], tau, tau_rot], axis=1),
            'p1_tau_rot_shuffle': np.concatenate(
                [data['p1'], tau, tau_rot_shuffled], axis=1),
            'p2_rot': np.concatenate([data['p2'], tau_rot], axis=1),
            'p2_tau_rot': np.concatenate([data['p2'], tau, tau_rot], axis=1),
            'p2_tau_rot_shuffle': np.concatenate(
                [data['p2'], tau, tau_rot_shuffled], axis=1),
        })
    y = data['action'].astype(np.float32)
    action_std = np.maximum(y.std(axis=0), 1e-6)
    fold_of_seed = {int(seed): i % args.folds
                    for i, seed in enumerate(seed_values)}
    results = {name: [] for name in variants}
    started = time.monotonic()
    for fold in range(args.folds):
        val_mask = np.isin(data['seed'], [s for s, f in fold_of_seed.items() if f == fold])
        if not val_mask.any() or val_mask.all():
            raise ValueError(f'empty fold {fold}')
        train_mask = ~val_mask
        for name, x in variants.items():
            for seed in args.seeds:
                train_l1, val_l1 = train_eval(
                    x[train_mask], y[train_mask], x[val_mask], y[val_mask],
                    seed, args.hidden, args.steps, device)
                results[name].append(dict(
                    fold=fold, model_seed=seed,
                    train=group_metrics(train_l1, action_std),
                    val=group_metrics(val_l1, action_std),
                    val_phase=phase_metrics(val_l1, data['tick'][val_mask])))
    baseline_l1 = np.abs(y - y.mean(axis=0, keepdims=True))
    aggregate = {}
    for name, rows in results.items():
        aggregate[name] = {}
        for split in ('train', 'val'):
            for metric in ('all', 'xyz', 'rot', 'fingers', 'normalized'):
                values = np.asarray([row[split][metric] for row in rows])
                aggregate[name][f'{split}_{metric}'] = dict(
                    mean=float(values.mean()), std=float(values.std()))
    for name, rows in results.items():
        phase_agg = {}
        for phase in ('approach', 'contact', 'lift'):
            values = [row['val_phase'][phase]['all'] for row in rows if phase in row['val_phase']]
            if values:
                phase_agg[phase] = dict(mean=float(np.mean(values)),
                                        std=float(np.std(values)))
        aggregate[name]['val_phase_all'] = phase_agg
    write_json(args.output / 'results.json', dict(
        schema='ref2dex.unidex-partial-obs-identification.v1',
        data=str(args.data.resolve()), rows=int(len(y)),
        teacher_seeds=[int(s) for s in seed_values],
        successful_only=bool(args.success_only),
        hidden=args.hidden, steps=args.steps, model_seeds=args.seeds,
        folds=args.folds,
        input_dims={name: int(x.shape[1]) for name, x in variants.items()},
        baseline=group_metrics(baseline_l1, action_std),
        aggregate=aggregate, per_run=results,
        elapsed_s=time.monotonic() - started))
    table = []
    for name in variants:
        row = dict(model=name, dim=int(variants[name].shape[1]))
        row.update({key: value['mean'] for key, value in aggregate[name].items()
                    if key.startswith('val_') and 'mean' in value})
        table.append(row)
    print(json.dumps(dict(rows=int(len(y)), seeds=[int(s) for s in seed_values],
                          baseline=group_metrics(baseline_l1, action_std),
                          table=table), indent=2))


if __name__ == '__main__':
    main()
