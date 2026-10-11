"""Frozen source-state replay separating trajectory prediction from plan padding."""
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
sys.path[:0] = [str(TASK / 'src'), str(ROOT / 'src/task/consequence-evaluator/src')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    os.environ['TMPDIR'] = str(ROOT / 'tmp/trajectory-policy/competent-pd')
    import torch
    from trajectory_policy.competent_actor import actor_windows, load_actor, world_plan, shifted_plan
    from trajectory_policy.retargeter_data import features
    from trajectory_policy.pd_retargeter import load_model
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    manifest = json.loads((args.fit / 'manifest.json').read_text())
    source_dir = Path(manifest['source'])
    with np.load(source_dir / 'trajectory.npz', allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    r = load_model(torch.load(manifest['retargeter'], map_location='cuda', weights_only=False), 'cuda')
    ids = np.arange(16, 24)  # validation only; no new training or test selection
    x, state, truth_tau, target, ticks = actor_windows(data, ids)
    actors = {name: load_actor(torch.load(args.fit / (name + '.pt'), map_location='cuda', weights_only=False), 'cuda')
              for name in ('tau_only', 'joint')}
    predicted = {}
    with torch.no_grad():
        for name, model in actors.items():
            predicted[name] = np.concatenate([model(torch.as_tensor(x[i:i + 512], device='cuda')).cpu().numpy()
                                             for i in range(0, len(x), 512)]).reshape(519, 8, 24, 33)
    taus = dict(gt_fresh=truth_tau)
    for name in ('gt', 'tau_only', 'joint'):
        for period in ([8] if name == 'gt' else [1, 8]):
            windows = []
            for tick in range(519):
                query = tick - tick % period
                if name == 'gt':
                    plan = np.moveaxis(data['hand'][query + np.arange(1, 25)][:, ids], 0, 1)
                else:
                    plan = world_plan(predicted[name][query], data['hand'][query, ids], data['obj'][query, ids, :3, :3])
                future = shifted_plan(plan, tick - query)
                _, tau = features(*(data[k][tick, ids] for k in ('q', 'dq', 'hand', 'obj', 'velocity')), future)
                windows.append(tau)
            taus[name + '_period' + str(period)] = np.concatenate(windows)
    reports, outputs = {}, {}
    with torch.no_grad():
        for name, tau in taus.items():
            outputs[name] = np.concatenate([r(torch.as_tensor(state[i:i + 512], device='cuda'),
                                             torch.as_tensor(tau[i:i + 512], device='cuda')).cpu().numpy()
                                            for i in range(0, len(state), 512)])
            reports[name] = {}
            for phase, mask in [('first8', ticks < 8), ('approach', (ticks >= 8) & (ticks < 48)),
                                ('contact', (ticks >= 48) & (ticks < 64)), ('all', ticks >= 0)]:
                error = outputs[name][mask] - target[mask]
                reports[name][phase] = dict(pd_mse=float(np.mean((error / r.target_scale.cpu().numpy()) ** 2)),
                    xyz_p95_m=float(np.quantile(np.abs(error[:, :3]).max(-1), .95)),
                    rotation_p95_rad=float(np.quantile(np.abs(error[:, 3:6]).max(-1), .95)),
                    finger_p95_rad=float(np.quantile(np.abs(error[:, 6:]).max(-1), .95)),
                    pd_xyz_shift_from_fresh_R_rms_m=float(np.sqrt(np.mean((outputs[name][mask, :3] - outputs['gt_fresh'][mask, :3]) ** 2))),
                    first8_hand_coordinate_rms_m=float(np.sqrt(np.mean((tau[mask, :8] - truth_tau[mask, :8]) ** 2))))
    out = args.output.resolve()
    if out.exists() or ROOT / 'outputs/trajectory-policy' not in out.parents:
        raise ValueError('fresh task output required')
    out.mkdir(parents=True)
    result = dict(reports=reports, no_refit=True, validation_envs=ids.tolist(),
        scope='frozen source-state diagnostic; no closed-loop success attribution',
        source_sha256=hashlib.sha256((source_dir / 'trajectory.npz').read_bytes()).hexdigest(),
        diagnostic_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    main()
