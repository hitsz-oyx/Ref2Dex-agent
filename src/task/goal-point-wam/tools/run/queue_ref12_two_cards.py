#!/usr/bin/env python3
"""Run two saved method waves exclusively on GPUs6/7; retain original deadline."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from train_three_domain import atomic_json


def run(args):
    plans = [json.loads(path.read_text()) for path in args.plans]
    if any(set(entry['gpu'] for entry in plan['runs']) != {6, 7} for plan in plans):
        raise ValueError('two-card queue uses exclusively GPUs6/7')
    deadline = plans[0]['deadline']
    if any(plan['deadline'] != deadline for plan in plans):
        raise ValueError('all waves preserve the original absolute deadline')
    stop = [False]; child = None
    for sig in (signal.SIGUSR1, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    state = dict(status='RUNNING', pid=os.getpid(), gpus=[6, 7], deadline=deadline, phases=[])
    try:
        for path, plan in zip(args.plans, plans):
            if stop[0] or time.time() >= deadline:
                state['status'] = 'STOPPED'; break
            command = [sys.executable, str(Path(__file__).with_name('launch_ref12_methods.py')), '--plan', str(path)]
            child = subprocess.Popen(command, start_new_session=True)
            state['phases'].append(dict(plan=str(path), output=plan['output'], pid=child.pid))
            atomic_json(args.status, state)
            signaled = False
            while child.poll() is None:
                if (stop[0] or time.time() >= deadline) and not signaled:
                    child.send_signal(signal.SIGUSR1); signaled = True
                time.sleep(10)
            if child.returncode:
                state['status'] = 'FAILED'; break
            results = [json.loads((Path(plan['output']) / entry['method'] / 'result.json').read_text()) for entry in plan['runs']]
            if any(value['status'] != 'COMPLETED' for value in results):
                state['status'] = 'STOPPED'; break
        else:
            state['status'] = 'COMPLETED'
        state['ended_at'] = time.time(); atomic_json(args.status, state)
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGUSR1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plans', nargs='+', type=Path, required=True)
    p.add_argument('--status', type=Path, required=True)
    run(p.parse_args())
