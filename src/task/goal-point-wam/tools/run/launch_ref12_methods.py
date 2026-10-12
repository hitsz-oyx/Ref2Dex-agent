#!/usr/bin/env python3
"""Own-process supervisor for remaining-card method Probes; unrelated DDP is untouched."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
REPO = TASK.parents[2]
HELPER = TASK.parent / 'cm-pointflow-effect-pretrain/tools/run/launch_pointworld_action_ddp.py'
spec = importlib.util.spec_from_file_location('owned_process_helpers', HELPER)
helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)


def run(args):
    plan = json.loads(args.plan.read_text())
    root = Path(plan['output']).resolve(); root.mkdir(parents=True, exist_ok=False)
    gpus = [entry['gpu'] for entry in plan['runs']]
    if len(set(gpus)) != len(gpus) or not set(gpus).issubset({4, 5, 6, 7}):
        raise ValueError('only remaining GPUs4--7, one task/card')
    if helpers.gpu_processes(gpus):
        raise RuntimeError('assigned card is busy; never preempt it')
    started = time.time(); deadline = started + (900 if args.smoke else plan['wall_seconds'])
    if deadline > started + 5400:
        raise ValueError('90min maximum additional wall budget')
    sources = [Path(__file__), Path(__file__).with_name('train_ref12_method.py'), Path(__file__).with_name('ref12_methods.py'),
               TASK / 'tools/run/train_three_domain.py', Path(plan['stats']), Path(plan['data']) / 'processed/manifest.json', Path(plan['indices'])]
    sources += list((TASK / 'src/goal_point_wam').glob('*.py'))
    sources += [Path(entry['config']) for entry in plan['runs']]
    hashes = {str(p.resolve()): helpers.digest(p) for p in sources}
    stopped = [False]
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    children = []; handles = []
    state = dict(status='TRAINING', pid=os.getpid(), started_at=started, deadline=deadline, smoke=args.smoke,
                 gpus=gpus, sources=hashes, runs=[], git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    try:
        for entry in plan['runs']:
            output = root / entry['method']; handle = (root / (entry['method'] + '.log')).open('w'); handles.append(handle)
            command = [sys.executable, str(Path(__file__).with_name('train_ref12_method.py')),
                       '--config', entry['config'], '--output', str(output), '--deadline', str(deadline)]
            for name in ('parent', 'data', 'stats', 'indices'):
                command += ['--' + name, plan[name]]
            if args.smoke:
                command += ['--smoke']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(entry['gpu']), TMPDIR=str(REPO / 'tmp'),
                       TRITON_CACHE_DIR=str(REPO / 'tmp/triton-ref12-methods'), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
            child = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child); state['runs'].append(dict(**entry, pid=child.pid, output=str(output), command=command))
        storage_tick = 0; requested_at = None
        while any(child.poll() is None for child in children):
            foreign = [pid for pid in helpers.gpu_processes(gpus) if not any(helpers.belongs_to(pid, child.pid) for child in children)]
            drift = any(not Path(p).exists() or helpers.digest(Path(p)) != h for p, h in hashes.items())
            if time.time() - storage_tick >= 120:
                value = subprocess.check_output(['du', '-sb', str(REPO / 'outputs'), str(REPO / 'tmp')], text=True)
                state['artifact_bytes'] = sum(int(line.split()[0]) for line in value.splitlines()); storage_tick = time.time()
            failed = any(child.poll() not in (None, 0) for child in children)
            reason = ('foreign_gpu_process' if foreign else 'source_drift' if drift else 'child_failed' if failed else
                      'storage_cap' if state.get('artifact_bytes', 0) >= 298_000_000_000 else
                      'user_stop' if stopped[0] else 'deadline' if time.time() >= deadline else None)
            if reason and requested_at is None:
                state['stop_reason'] = reason; requested_at = time.time()
                for child in children:
                    if child.poll() is None and helpers.belongs_to(child.pid, os.getpid()):
                        child.send_signal(signal.SIGUSR1)
            if requested_at and time.time() - requested_at > 180:
                for child in children:
                    if child.poll() is None and helpers.belongs_to(child.pid, os.getpid()):
                        child.terminate()
            for entry, child in zip(state['runs'], children):
                entry['exit_code'] = child.poll()
                progress = Path(entry['output']) / 'progress.json'
                if progress.exists():
                    entry['progress'] = json.loads(progress.read_text())
            helpers.record(root / 'group_status.json', state); time.sleep(10 if args.smoke else 30)
        state.update(status='STOPPED' if requested_at else 'FAILED' if any(c.returncode for c in children) else 'COMPLETED', ended_at=time.time())
        for entry, child in zip(state['runs'], children):
            entry['exit_code'] = child.returncode
        helpers.record(root / 'group_status.json', state)
    finally:
        for child in children:
            if child.poll() is None and helpers.belongs_to(child.pid, os.getpid()):
                child.terminate()
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--smoke', action='store_true')
    run(p.parse_args())
