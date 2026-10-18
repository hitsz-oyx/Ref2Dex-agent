#!/usr/bin/env python3
"""Train a zero-initialized residual adapter on frozen oracle-tau students.

The base policy is untouched; a small residual corrects the late-phase action
using teacher actions collected at the student's own replayed states.  With a
zero-initialized last layer and a small gain the deployed policy starts as the
base and changes only where the residual data supports it.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from wam_rl_policy.student_inputs import student_features


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2,
                               default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


def build_base(input_dim, hidden):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden), torch.nn.ELU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ELU(),
        torch.nn.Linear(hidden, 24))


def build_residual(input_dim, hidden):
    net = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden), torch.nn.ELU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ELU(),
        torch.nn.Linear(hidden, 24))
    torch.nn.init.zeros_(net[-1].weight)
    torch.nn.init.zeros_(net[-1].bias)
    return net


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--dagger-data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gain', type=float, default=.05)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--min-tick', type=int, default=0,
                   help='optionally restrict to late-packet ticks')
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()
    if args.output == Path('/tmp') or Path('/tmp') in args.output.parents:
        raise ValueError('checkpoint must remain in the project workspace')
    device = f'cuda:{args.gpu}'
    base_ckpt = torch.load(args.base, map_location=device, weights_only=False)
    if base_ckpt['variant'] != 'p2_tau':
        raise ValueError('residual adapter requires a p2_tau base')
    input_dim = int(base_ckpt['input_dim'])
    mean = torch.tensor(base_ckpt['mean'], dtype=torch.float32, device=device)
    std = torch.tensor(base_ckpt['std'], dtype=torch.float32, device=device)
    with np.load(args.dagger_data / 'dagger_data.npz', allow_pickle=False) as source:
        packet = {key: source[key] for key in source.files}
    if args.min_tick:
        keep = packet['tick'] >= args.min_tick
        packet = {key: value[keep] for key, value in packet.items()}
    features = student_features(packet['p2'], 'p2_tau', tau=packet['tau'],
                                pad_to=base_ckpt.get('pad_to', 0))
    if features.shape[1] != input_dim:
        raise ValueError('base input dimension mismatch')
    x = torch.tensor((features - base_ckpt['mean']) / base_ckpt['std'],
                     dtype=torch.float32, device=device)
    y = torch.tensor(packet['action'], dtype=torch.float32, device=device)
    base = build_base(input_dim, int(base_ckpt['hidden'])).to(device)
    base.load_state_dict(base_ckpt['state_dict'])
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    residual = build_residual(input_dim + 24, args.hidden).to(device)
    optimizer = torch.optim.Adam(residual.parameters(), lr=args.lr, weight_decay=1e-4)
    with torch.no_grad():
        base_action = base(x).clamp(-1, 1)
        before = float((base_action - y).abs().mean().item())
    residual_input = torch.cat([x, base_action], dim=1)
    for _ in range(args.steps):
        optimizer.zero_grad()
        action = (base_action + args.gain * torch.tanh(residual(residual_input))).clamp(-1, 1)
        loss = F.smooth_l1_loss(action, y, beta=0.1)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        action = (base_action + args.gain * torch.tanh(residual(residual_input))).clamp(-1, 1)
        after = float((action - y).abs().mean().item())
        per_dim = (action - y).abs().mean(dim=0).cpu().numpy()
        correction = (action - base_action).abs().mean().item()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(base_state_dict=base_ckpt['state_dict'],
                    base_mean=base_ckpt['mean'], base_std=base_ckpt['std'],
                    base_variant=base_ckpt['variant'], hidden=int(base_ckpt['hidden']),
                    input_dim=input_dim, pad_to=int(base_ckpt.get('pad_to', 0)),
                    residual_state_dict=residual.state_dict(), gain=args.gain,
                    rows=int(len(x)), min_tick=args.min_tick),
               args.output / 'residual.pt')
    write_json(args.output / 'metrics.json', dict(
        schema='ref2dex.unidex-holding-residual.v1',
        base=str(args.base.resolve()), dagger_data=str(args.dagger_data.resolve()),
        rows=int(len(x)), min_tick=args.min_tick, gain=args.gain,
        steps=args.steps, lr=args.lr,
        base_l1=before, residual_l1=after, mean_abs_correction=float(correction),
        xyz_l1=float(per_dim[0:3].mean()), rot_l1=float(per_dim[3:6].mean()),
        fingers_l1=float(per_dim[6:24].mean())))
    print(json.dumps(dict(rows=int(len(x)), base_l1=before, residual_l1=after,
                          mean_abs_correction=float(correction),
                          xyz_l1=float(per_dim[0:3].mean()),
                          rot_l1=float(per_dim[3:6].mean()),
                          fingers_l1=float(per_dim[6:24].mean())), indent=2))


if __name__ == '__main__':
    main()
