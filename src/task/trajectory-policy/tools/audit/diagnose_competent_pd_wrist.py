"""Train-only affine PD fit from observed wrist velocity and next palm motion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists() or ROOT / 'outputs/trajectory-policy' not in out.parents:
        raise ValueError('fresh task output required')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    torch.set_num_threads(2)
    started = time.monotonic()
    with np.load(args.source / 'trajectory.npz', allow_pickle=False) as stream:
        data = {key: stream[key].copy() for key in ('hand', 'q', 'dq', 'pd_targets')}
    ticks = np.arange(519)
    inputs = {}
    for name, envs in [('train', np.arange(16)), ('validation', np.arange(16, 24))]:
        # No actual future robot q. The palm is the measured hand trajectory's
        # link0, already confirmed to match the translation joints within5.5um.
        motion = data['hand'][ticks + 1][:, envs, 0] - data['hand'][ticks][:, envs, 0]
        second = data['hand'][ticks + 2][:, envs, 0] - data['hand'][ticks][:, envs, 0]
        velocity = data['dq'][ticks][:, envs, :3]
        target = data['pd_targets'][ticks][:, envs, :3] - data['q'][ticks][:, envs, :3]
        full = np.concatenate((motion / .01, velocity, (second - 2 * motion) / .01), -1).reshape(-1, 9)
        current = velocity.reshape(-1, 3)
        inputs[name] = full, current, target.reshape(-1, 3)
    reports, coefficients = {}, {}
    for key, slot in [('motion', 0), ('state_only', 1)]:
        x = torch.as_tensor(inputs['train'][slot], dtype=torch.float64, device='cuda')
        y = torch.as_tensor(inputs['train'][2], dtype=torch.float64, device='cuda')
        mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
        x = torch.cat(((x - mean) / scale, torch.ones(len(x), 1, device='cuda')), -1)
        gram = x.T @ x
        # One fixed tiny ridge; no hyperparameter selection or test access.
        gram += torch.eye(gram.shape[0], device='cuda') * 1e-3
        weight = torch.linalg.solve(gram, x.T @ y)
        coefficients[key] = dict(mean=mean.cpu().tolist(), scale=scale.cpu().tolist(), weight=weight.cpu().tolist())
        reports[key] = {}
        for split in inputs:
            vx = torch.as_tensor(inputs[split][slot], dtype=torch.float64, device='cuda')
            vx = torch.cat(((vx - mean) / scale, torch.ones(len(vx), 1, device='cuda')), -1)
            error = np.abs((vx @ weight).cpu().numpy() - inputs[split][2])
            startup = np.repeat(ticks < 64, 16 if split == 'train' else 8)
            reports[key][split] = dict(xyz_p95_m=float(np.quantile(error.max(-1), .95)),
                startup_xyz_p95_m=float(np.quantile(error[startup].max(-1), .95)),
                mae_m=float(error.mean()))
    out.mkdir(parents=True)
    result = dict(reports=reports, coefficients=coefficients, elapsed_s=time.monotonic() - started,
        source_sha256=hashlib.sha256((args.source / 'trajectory.npz').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='train/validation wrist conditioning diagnostic only; no test, physics or high-level claim')
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    main()
