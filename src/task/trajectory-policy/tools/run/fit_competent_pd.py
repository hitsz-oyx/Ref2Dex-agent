"""Fixed competent-PD BC fit with matched state-only control and whole-env split."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK / 'src')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists() or ROOT / 'outputs/trajectory-policy' not in out.parents:
        raise ValueError('fresh task-owned output required')
    source_manifest = json.loads((args.source / 'manifest.json').read_text())
    source_result = json.loads((args.source / 'result.json').read_text())
    if (source_manifest.get('status') != 'COMPLETED' or not source_manifest.get('training_allowed')
            or not source_result.get('teacher_readiness')):
        raise ValueError('qualified fresh training-eligible teacher required')
    if not json.loads((args.source / 'audit.json').read_text()).get('pass'):
        raise ValueError('teacher PD data audit required before fitting')
    for path, digest in source_manifest['input_sha256'].items():
        if sha(path) != digest:
            raise ValueError('teacher inputs changed: ' + path)
    if sha(args.source / 'trajectory.npz') != source_manifest['trajectory_sha256']:
        raise ValueError('teacher trajectory identity changed')
    gpu = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
        '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    util, used = map(int, gpu.split(','))
    pids = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).split()
    if pids or util > 10 or used > 512:
        raise RuntimeError('GPU not idle: ' + gpu)
    scratch = ROOT / 'tmp/trajectory-policy/competent-pd'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(scratch),
                      PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
    import torch
    from trajectory_policy.pd_retargeter import PDRetargeter, SCHEMA, windows, decode_target, command
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(181)
    with np.load(args.source / 'trajectory.npz', allow_pickle=False) as stream:
        data = {key: stream[key].copy() for key in ('q', 'dq', 'hand', 'obj', 'velocity',
                                                 'pd_targets', 'surface_gap', 'support_gap', 'table_footprint')}
    groups = dict(train=list(range(16)), validation=list(range(16, 24)), test=list(range(24, 32)))
    arrays = {key: windows(data, envs) for key, envs in groups.items()}
    if any(not np.isfinite(v).all() for pieces in arrays.values() for v in pieces):
        raise ValueError('finite continuous source required')
    tensors = {key: [torch.as_tensor(value, device='cuda') for value in pieces[:3]]
               for key, pieces in arrays.items()}
    x, tau, target = tensors['train']
    base = PDRetargeter().cuda()
    with torch.no_grad():
        floor = torch.tensor([.001] * 3 + [.01] * 9, device='cuda')
        for name, values, minimum in (('state', x, .001), ('tau', tau, .001), ('target', target, floor)):
            getattr(base, name + '_mean').copy_(values.mean(0))
            getattr(base, name + '_scale').copy_(torch.maximum(values.std(0, unbiased=False),
                                                             torch.as_tensor(minimum, device='cuda')))
    models = dict(tau=base, state_only=copy.deepcopy(base))
    models['state_only'].use_tau = False
    optimizers = {key: torch.optim.Adam(model.parameters(), lr=3e-4) for key, model in models.items()}
    controller = source_manifest['native_controller']
    offset = torch.tensor(controller['offset'], device='cuda')
    scale = torch.tensor(controller['scale'], device='cuda')
    files = [Path(__file__), TASK / 'src/trajectory_policy/pd_retargeter.py',
        TASK / 'src/trajectory_policy/retargeter_data.py',
        *[args.source / name for name in ('manifest.json', 'result.json', 'audit.json', 'trajectory.npz')]]
    hashes = {str(path.resolve()): sha(path) for path in files}
    out.mkdir(parents=True)
    run = dict(schema=SCHEMA, status='RUNNING', input_sha256=hashes, physical_gpu=args.gpu,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        seed=181, updates=3000, batch_size=512, width=256, learning_rate=.0003,
        groups=groups, source=str(args.source.resolve()), native_controller=controller,
        train_only_normalization=True, target='actual PD[t]; wrist relative to measured q[t], fingers absolute',
        checkpoint_selection='own lowest validation normalized MSE every250 updates',
        scope='offline PD inverse and trajectory-use Decision, no high-level or Cm claim', gpu_before=gpu)
    write(out / 'manifest.json', run)
    started = time.monotonic()
    best = {key: (float('inf'), None, None) for key in models}
    metrics = []

    def predict(model, split, shuffled=False):
        s, t, y = tensors[split]
        if shuffled:
            rows = len(groups[split])
            t = t.reshape(-1, rows, 24, 33).roll(1, 1).reshape_as(t)
        model.eval()
        with torch.no_grad():
            outputs = torch.cat([model(s[start:start + 512], t[start:start + 512])
                                 for start in range(0, len(s), 512)])
            loss = ((outputs - y) / model.target_scale).square().mean()
        return outputs, float(loss)

    def report(model, split, shuffled=False):
        prediction, loss = predict(model, split, shuffled)
        truth = tensors[split][2]
        error = (prediction - truth).abs().cpu().numpy()
        tick = arrays[split][3]
        env = np.tile(groups[split], len(tick) // len(groups[split]))
        current = torch.as_tensor(data['q'][tick, env], device='cuda')
        pd = decode_target(prediction, current)
        intended, applied, clipped = command(pd, current, offset, scale)
        result = dict(normalized_mse=loss, physical_mae=float(error.mean()), clipping_rate=float(clipped.float().mean()))
        held = (data['surface_gap'][tick, env] <= .01) & (data['obj'][tick, env, 2, 3] - data['obj'][0, env, 2, 3] >= .03)
        for name, mask in dict(all=np.ones(len(tick), bool), startup=tick < 64, held=held).items():
            result[name] = dict(samples=int(mask.sum()))
            if mask.any():
                result[name].update(xyz_p95_m=float(np.quantile(error[mask, :3].max(-1), .95)),
                    rotation_p95_rad=float(np.quantile(error[mask, 3:6].max(-1), .95)),
                    finger_p95_rad=float(np.quantile(error[mask, 6:].max(-1), .95)))
        return result, prediction.cpu().numpy()

    def deadline(signum, frame):
        raise TimeoutError('fixed PD fit exceeded300s')

    handler = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(300)
    try:
        for step in range(1, 3001):
            ids = torch.randint(len(x), (512,), device='cuda')
            for key, model in models.items():
                model.train()
                prediction = model(x[ids], tau[ids])
                loss = ((prediction - target[ids]) / model.target_scale).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite PD objective')
                optimizers[key].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizers[key].step()
            if step == 1 or step % 250 == 0:
                row = dict(update=step, elapsed_s=time.monotonic() - started,
                    eta_s=(time.monotonic() - started) / step * (3000 - step))
                for key, model in models.items():
                    _, val = predict(model, 'validation')
                    row[key] = val
                    if val < best[key][0]:
                        best[key] = (val, step, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                row['gpu'] = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                    '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
                active = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                    '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).split()
                if set(map(int, active)) - {os.getpid()}:
                    raise RuntimeError('foreign GPU process')
                metrics.append(row)
                print(json.dumps(row), flush=True)
                with (out / 'monitor.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
        reports, predictions, checkpoint_sha = {}, {}, {}
        for key, model in models.items():
            model.load_state_dict(best[key][2])
            reports[key], predictions[key] = {}, {}
            for split in arrays:
                reports[key][split], predictions[key][split] = report(model, split)
            torch.save(dict(schema=SCHEMA, width=256, use_tau=model.use_tau, model=best[key][2],
                selected_update=best[key][1], input_sha256=hashes), out / (key + '.pt'))
            checkpoint_sha[key] = sha(out / (key + '.pt'))
        reports['tau_shuffle'], _ = report(models['tau'], 'validation', True)
        probe_tau = tensors['validation'][1][:64].detach().clone().requires_grad_(True)
        models['tau'](tensors['validation'][0][:64], probe_tau).square().mean().backward()
        gradient = float(probe_tau.grad.abs().sum())
        r = reports['tau']['validation']
        precision = all(r[part]['xyz_p95_m'] <= .005 and r[part]['rotation_p95_rad'] <= .05
                        and r[part]['finger_p95_rad'] <= .05 for part in ('all', 'startup'))
        passed = (precision and r['clipping_rate'] < .01 and gradient > 0
                  and r['normalized_mse'] < reports['state_only']['validation']['normalized_mse'])
        np.savez_compressed(out / 'predictions.npz', **{key + '_' + split: value
            for key, parts in predictions.items() for split, value in parts.items()})
        result = dict(status='PROMISING' if passed else 'UNCLEAR', offline_only=True,
            physical_screen_pass=passed, precision_pass=precision, reports=reports,
            selected_updates={key: best[key][1] for key in best}, checkpoint_sha256=checkpoint_sha,
            tau_gradient_l1=gradient, windows={key: len(value[0]) for key, value in arrays.items()},
            elapsed_s=time.monotonic() - started, torch_peak_mib=torch.cuda.max_memory_allocated() / 2**20)
        write(out / 'result.json', result)
        if any(sha(path) != value for path, value in hashes.items()):
            raise ValueError('input drift')
        run.update(status='COMPLETED', elapsed_s=result['elapsed_s'])
        write(out / 'manifest.json', run)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        run.update(status='FAILED', error=repr(error))
        write(out / 'manifest.json', run)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, handler)


if __name__ == '__main__':
    main()
