"""Frozen fit diagnosis: training precision versus whole-wave distribution shift."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from audit_learned_retargeter import ROOT, TASK, reconstruct, report, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or ROOT/'outputs/trajectory-policy' not in args.output.parents:
        raise ValueError('fresh task-owned diagnostic output required')
    before = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    utilization, memory = map(int, before.split(','))
    if utilization > 10 or memory > 512:
        raise ValueError('diagnostic GPU not idle: '+before)
    audit = json.loads((args.audit/'manifest.json').read_text())
    if audit['audit'] != 'PASS' or audit['status'] != 'COMPLETED':
        raise ValueError('completed independent audit required')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE='1')
    import torch
    sys.path.insert(0, str(TASK/'src'))
    from trajectory_policy.learned_retargeter import load_retargeter
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    manifest = dict(status='RUNNING', git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        physical_gpu=args.gpu, gpu_before=before)
    try:
        fit = json.loads((args.fit/'manifest.json').read_text())
        hashes = dict(fit['input_sha256'])
        for path in (args.fit/'manifest.json', args.fit/'result.json', args.audit/'manifest.json', Path(__file__), Path(__file__).with_name('audit_learned_retargeter.py')):
            hashes[str(path.resolve())] = sha(path)
        for name in ('tau', 'state_only'):
            hashes[str((args.fit/(name+'-best.pt')).resolve())] = fit['checkpoint_sha256'][name]
        if any(sha(path) != digest for path, digest in hashes.items()):
            raise ValueError('frozen fit/source drift')
        sources, ticks = [], []
        for metadata in fit['metadata']:
            with np.load(Path(metadata['source'])/'low.npz', allow_pickle=False) as f:
                data = {key: f[key] for key in ('q', 'dq', 'hand', 'obj', 'velocity', 'applied', 'episode_start', 'episode_tick')}
            state, tau, action, indices = reconstruct(data, [0, 542, 1084, 1626])
            sources.append((state.reshape(-1, 87), tau.reshape(-1, 24, 33), action.reshape(-1, 8, 12)))
            ticks.append(np.repeat(data['episode_tick'][indices], 16))
        state, tau, truth = [np.concatenate([part[k] for part in sources]) for k in range(3)]
        ticks = np.concatenate(ticks)
        if len(state) != 66304:
            raise ValueError('fixed training sample count differs')
        diagnostics = {}
        for name in ('tau', 'state_only'):
            model = load_retargeter(torch.load(args.fit/(name+'-best.pt'), map_location='cpu', weights_only=False), 'cuda').eval()
            outputs = []
            with torch.no_grad():
                for begin in range(0, len(state), 512):
                    outputs.append(model(torch.as_tensor(state[begin:begin+512], device='cuda'),
                        torch.as_tensor(tau[begin:begin+512], device='cuda')).cpu().numpy())
            prediction = np.concatenate(outputs)
            if not np.isfinite(prediction).all():
                raise ValueError('nonfinite frozen train predictions')
            scale = model.action_scale.cpu().numpy()
            train = report(prediction, truth, scale, fit['native_controller'], ticks < 64)
            validation = audit['metrics'][name]
            diagnostics[name] = dict(train=train, validation=validation,
                validation_train_l1_ratio=validation['normalized_l1']/train['normalized_l1'],
                train_normalized_l1_by_horizon=(np.abs(prediction-truth)/scale).mean(axis=(0, 2)).tolist(),
                train_first_native_mae_by_coordinate=np.abs(prediction[:, 0].clip(-1, 1)-truth[:, 0]).mean(axis=0).tolist())
            processes = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).strip()
            for process in processes.splitlines():
                pid, used = map(int, process.split(','))
                if pid != os.getpid() and used > 512:
                    raise RuntimeError('foreign GPU compute detected')
        if any(sha(path) != digest for path, digest in hashes.items()):
            raise ValueError('source drift during diagnostic')
        write(args.output/'result.json', diagnostics)
        manifest.update(status='COMPLETED', elapsed_s=time.monotonic()-started, input_sha256=hashes,
            train_samples=len(state), torch_peak_bytes=torch.cuda.max_memory_allocated(),
            claim='Frozen in-sample precision diagnostic; no updates, physical execution, hyperparameter selection or formal generalization claim')
        write(args.output/'manifest.json', manifest)
        print(json.dumps(diagnostics, indent=2), flush=True)
    except BaseException as error:
        manifest.update(status='FAILED', elapsed_s=time.monotonic()-started, error=repr(error))
        write(args.output/'manifest.json', manifest)
        raise


if __name__ == '__main__':
    def deadline(signum, frame):
        raise TimeoutError('frozen retargeter precision diagnostic exceeded60s')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(60)
    main()
