"""Registered steady-object correction probe on frozen references, no training."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
TASK = ROOT/'src/task/wam-rl-policy-v3'
PYTHON = '/home2/wyy/miniconda3/envs/graspenv/bin/python'


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wait-seconds', type=int, default=0)
    a = p.parse_args();out = a.output.resolve()
    if ROOT/'outputs/wam-rl-policy-v3' not in out.parents or not 0 <= a.wait_seconds <= 3600:
        raise ValueError('bounded task output and <=3600s resource wait required')
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = out/'execution.json'
    if ledger_path.exists():raise ValueError('do not overwrite existing execution')
    ledger = dict(status='WAITING_GPU', git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                  maximum_controls=6000, maximum_simulation_wall_s=1800, maximum_new_bytes=2**30,
                  references='frozen v53 constant endpoint and r4 straight 100mm at 30/100/250mm/s',
                  policy_updates=0, commands=[], wait_samples=[])
    started = time.monotonic()
    while True:
        gpu = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True)
        rows = [[int(y.strip()) for y in x.split(',')] for x in gpu.strip().splitlines()]
        ledger['wait_samples'].append(dict(elapsed_s=time.monotonic()-started, gpu=rows))
        available = [i for i, memory, utilization in rows if i in (4,5,6,7) and memory <= 64 and utilization <= 5]
        if available:
            selected = available[0];break
        ledger['status'] = 'BLOCKED_GPU' if time.monotonic()-started >= a.wait_seconds else 'WAITING_GPU'
        write(ledger_path, ledger)
        print(json.dumps(dict(status=ledger['status'], gpu=rows)), flush=True)
        if ledger['status'] == 'BLOCKED_GPU':return
        time.sleep(min(60, max(.01, a.wait_seconds-(time.monotonic()-started))))
    if shutil.disk_usage(ROOT).free < 22*2**30:raise RuntimeError('disk guard')
    ledger.update(status='RUNNING', gpu=selected);write(ledger_path, ledger)
    frozen = ROOT/'outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r4'
    endpoint = ROOT/'outputs/wam-rl-policy-v3/tracking-error-decomposition-20261023-r1/static-reference.npz'
    jobs = [('static', endpoint, 'on', 3)]
    for index in (12,13,14,15,16,17):
        meta = json.loads((frozen/('case%03d'%index)/'result.json').read_text())
        assert meta['shape']=='straight' and meta['scale_m']==.10 and meta['stage']=='B'
        jobs.append(('dynamic-%03d'%index, frozen/('case%03d'%index)/'reference.npz', 'on' if meta['feedforward'] else 'off', 1))
    sim_start = time.monotonic();controls = 0
    for name, reference, ff, repeats in jobs:
        remaining = int(1800-(time.monotonic()-sim_start))
        if remaining < 60:raise TimeoutError('registered simulation wall exhausted')
        command = [PYTHON,'-u',str(TASK/'tools/run/run_object_tracking.py'),
                   '--output',str(out/name),'--gpu',str(selected),'--seconds',str(min(900,remaining)),
                   '--max-controls',str(min(2500,6000-controls)),'--reference-npz',str(reference),
                   '--stages','B','--feedforward',ff,'--repeats',str(repeats),
                   '--legacy-unchecked','--object-steady-correction']
        ledger['commands'].append(command);write(ledger_path,ledger)
        with (out/(name+'.log')).open('x') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            ledger.update(status='FAILED', failed_job=name, returncode=result.returncode)
            write(ledger_path,ledger);raise RuntimeError('no automatic retry: '+name)
        manifest = json.loads((out/name/'manifest.json').read_text())
        controls += manifest['controls'];ledger['controls'] = controls
        print(json.dumps(dict(job=name, controls=manifest['controls'],status=manifest['status'])),flush=True)
        if any(r.get('abort',{} ) and r['abort'].get('kind') in ('limit','collision') for r in manifest['results']):
            ledger.update(status='STOPPED_RISK', failed_job=name,
                          rollback='correction remains opt-in/off by default; do not run more cases')
            write(ledger_path,ledger);return
        if manifest['status']!='COMPLETED':
            ledger.update(status='INCOMPLETE',failed_job=name);write(ledger_path,ledger);return
    ledger.update(status='COMPLETED', simulation_elapsed_s=time.monotonic()-sim_start)
    write(ledger_path,ledger)


if __name__=='__main__':main()
