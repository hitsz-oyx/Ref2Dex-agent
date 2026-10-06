#!/usr/bin/env python3
"""Wait for verified full627 caches, launch three matched arms and enforce group deadline."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

TASK = Path(__file__).resolve().parents[2]


def record(path, value):
    part = path.with_suffix('.json.part')
    part.write_text(json.dumps(value, indent=2) + '\n')
    part.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--config', type=Path, default=TASK / 'configs/wm30_k24.json')
    p.add_argument('--prepare-seconds', type=int, default=10800)
    args = p.parse_args()
    root = args.data.resolve()
    state_path = root / 'group_status.json'
    config = json.loads(args.config.read_text())
    start = time.time()
    identity = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                    scripts={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in
                             [TASK / 'tools/run/train_oakink2_wm30.py', TASK / 'src/oakink_wm/data.py', TASK / 'src/oakink_wm/model.py',
                              TASK / 'tools/audit/audit_wm30_splits.py', TASK / 'tools/run/evaluate_oakink2_wm30.py']})
    if state_path.exists(): raise FileExistsError('preserve existing group launch record')
    record(state_path, dict(status='WAITING_FOR_FULL_CORPUS', pid=os.getpid(), identity=identity, waiting_since=start))
    while not (root / 'processed/manifest.json').exists():
        if time.time() - start > args.prepare_seconds:
            record(state_path, dict(status='PREPARATION_TIMEOUT', identity=identity)); return 1
        try:
            download = json.loads((root / 'download_manifest.json').read_text())
            if download['status'] == 'FAILED':
                record(state_path, dict(status='DOWNLOAD_FAILED', error=download.get('error'), identity=identity)); return 1
        except (FileNotFoundError, json.JSONDecodeError): pass
        time.sleep(10)
    manifest = json.loads((root / 'processed/manifest.json').read_text())
    download = json.loads((root / 'download_manifest.json').read_text())
    if manifest['status'] != 'COMPLETED' or len(manifest['sequences']) != 627 or download['status'] != 'COMPLETED':
        raise ValueError('Full corpus manifest mismatch')
    if hashlib.sha256(args.config.read_bytes()).hexdigest() != identity['config_sha256']:
        raise ValueError('queued config drift')
    for name, digest in identity['scripts'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest: raise ValueError('queued code drift')
    subprocess.run([sys.executable, str(TASK / 'tools/audit/audit_wm30_splits.py'), '--data', str(root)], check=True)
    deadline = time.time() + config['group_seconds']
    while True:
        smi = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
        usage = {int(line.split(',')[0]): int(line.split(',')[1]) for line in smi.splitlines()}
        conflicts = {gpu: usage.get(gpu, -1) for gpu in (0, 1, 2) if usage.get(gpu, -1) > 512}
        if not conflicts:
            break
        if time.time() >= deadline:
            record(state_path, dict(status='GPU_CONFLICT', usage=usage, identity=identity)); return 1
        record(state_path, dict(status='WAITING_FOR_GPUS', conflicts=conflicts, usage=usage,
                                identity=identity, deadline=deadline))
        time.sleep(60)
    procs, logs = {}, []
    try:
        for gpu, arm in enumerate(('history', 'action', 'shuffle')):
            output = root / ('train-' + arm)
            output.mkdir(exist_ok=False)
            log = (output / 'console.log').open('w')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TMPDIR=str((Path.cwd() / 'tmp').resolve()))
            procs[arm] = subprocess.Popen([sys.executable, str(TASK / 'tools/run/train_oakink2_wm30.py'),
                            '--data', str(root), '--config', str(args.config.resolve()), '--output', str(output),
                            '--arm', arm, '--deadline', str(deadline)], stdout=log, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
        record(state_path, dict(status='TRAINING', deadline=deadline, identity=identity,
                               processes={k: p.pid for k, p in procs.items()}, updates=config['updates']))
        while any(proc.poll() is None for proc in procs.values()):
            failed = any(proc.poll() not in (None, 0) for proc in procs.values())
            if failed or time.time() >= deadline:
                for proc in procs.values():
                    if proc.poll() is None: os.kill(proc.pid, signal.SIGUSR1)
                if time.time() > deadline + 120:
                    for proc in procs.values():
                        if proc.poll() is None: os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(10)
        codes = {k: proc.returncode for k, proc in procs.items()}
        results = {k: json.loads((root / ('train-'+k) / 'progress.json').read_text()) for k in procs}
        steps = {k: v.get('step') for k, v in results.items()}
        status = 'FAILED' if any(c != 0 for c in codes.values()) else ('BUDGET_STOP' if any(v.get('status') == 'BUDGET_STOP' for v in results.values()) else 'COMPLETED')
        if status == 'COMPLETED' and time.time() + 120 < deadline:
            record(state_path, dict(status='TEST_EVALUATION', identity=identity, deadline=deadline))
            for arm in procs:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', TMPDIR=str((Path.cwd() / 'tmp').resolve()))
                subprocess.run([sys.executable, str(TASK / 'tools/run/evaluate_oakink2_wm30.py'),
                                '--data', str(root), '--checkpoint', str(root / ('train-'+arm) / 'final.pt'),
                                '--output', str(root / ('train-'+arm) / 'test_result.json')],
                               env=env, check=True, timeout=max(1, deadline-time.time()))
        record(state_path, dict(status=status,
                               identity=identity, exit_codes=codes, steps=steps,
                               matched_updates=len(set(steps.values())) == 1, deadline=deadline))
        return 0 if all(c == 0 for c in codes.values()) else 1
    except BaseException:
        for proc in procs.values():
            if proc.poll() is None: os.kill(proc.pid, signal.SIGUSR1)
        for proc in procs.values():
            if proc.poll() is None:
                try: proc.wait(timeout=120)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGTERM)
        raise
    finally:
        for log in logs: log.close()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        if '--data' in sys.argv:
            location = Path(sys.argv[sys.argv.index('--data')+1]) / 'group_status.json'
            try: previous = json.loads(location.read_text())
            except (FileNotFoundError, json.JSONDecodeError): previous = {}
            previous.update(status='FAILED', error=str(exc))
            record(location, previous)
        raise
