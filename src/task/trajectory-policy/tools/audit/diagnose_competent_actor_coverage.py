"""GPU nearest-source H/G coverage diagnostic; no fitting or future labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK / 'src'), str(ROOT / 'src/task/consequence-evaluator/src'), str(Path(__file__).parent)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--execution', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    os.environ['TMPDIR'] = str(ROOT / 'tmp/trajectory-policy/competent-pd')
    import torch
    from trajectory_policy.competent_actor import actor_windows, load_actor
    from audit_competent_actor import independent_input
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    fit = json.loads((args.fit / 'manifest.json').read_text())
    with np.load(Path(fit['source']) / 'trajectory.npz', allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    with np.load(args.execution / 'trajectory.npz', allow_pickle=False) as f:
        execution = {k: f[k].copy() for k in f.files}
    actor = load_actor(torch.load(args.fit / 'joint.pt', map_location='cuda', weights_only=False), 'cuda')
    x = actor_windows(data, list(range(16)))[0]
    validation = actor_windows(data, list(range(16, 24)))[0]
    reference = (torch.as_tensor(x, device='cuda') - actor.input_mean) / actor.input_scale
    reference_norm = reference.square().sum(-1)
    reports = {}

    def nearest(inputs):
        values, indices = [], []
        with torch.no_grad():
            for begin in range(0, len(inputs), 256):
                query = (torch.as_tensor(inputs[begin:begin + 256], device='cuda') - actor.input_mean) / actor.input_scale
                distance = (query.square().sum(-1, keepdim=True) + reference_norm[None] - 2 * query @ reference.T).clamp_min(0) / 340
                best, index = distance.min(-1)
                values.append(best.sqrt().cpu().numpy())
                indices.append(index.cpu().numpy())
        return np.concatenate(values), np.concatenate(indices)

    distance, _ = nearest(validation)
    threshold = float(np.quantile(distance, .95))
    reports['validation'] = dict(median=float(np.median(distance)), p95=threshold)
    saved = {}
    roles = json.loads((args.execution / 'manifest.json').read_text())['roles']
    for role in ('teacher', 'gt_tau_R', 'tau_only', 'joint'):
        ids = np.flatnonzero(np.array(roles) == role)
        ticks, envs = np.repeat(np.arange(519), len(ids)), np.tile(ids, 519)
        inputs = independent_input(execution, ticks, envs)
        distance, index = nearest(inputs)
        saved[role + '_distance'] = distance.reshape(519, len(ids))
        saved[role + '_source_tick'] = (index // 16).reshape(519, len(ids))
        reports[role] = {}
        for phase, mask in [('first8', ticks < 8), ('approach', (ticks >= 8) & (ticks < 48)),
                            ('contact', (ticks >= 48) & (ticks < 64)), ('held_late', ticks >= 64)]:
            reports[role][phase] = dict(median=float(np.median(distance[mask])),
                p95=float(np.quantile(distance[mask], .95)), beyond_val_p95=float((distance[mask] > threshold).mean()))
    out = args.output.resolve()
    if out.exists() or ROOT / 'outputs/trajectory-policy' not in out.parents:
        raise ValueError('fresh task output required')
    out.mkdir(parents=True)
    np.savez_compressed(out / 'coverage.npz', **saved)
    result = dict(reports=reports, no_refit=True, metric='nearest train H/G normalized RMS using train-only scales',
        scope='observational coverage only; distance is not causal attribution or a success gate',
        source_sha256=hashlib.sha256((Path(fit['source']) / 'trajectory.npz').read_bytes()).hexdigest(),
        execution_sha256=hashlib.sha256((args.execution / 'trajectory.npz').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    main()
