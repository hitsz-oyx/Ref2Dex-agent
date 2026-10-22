"""Launch only a measured, budget-approved paired probe, detached from the client."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[5]
PYTHON='/home2/wyy/miniconda3/envs/graspenv/bin/python'
RUNNER=Path(__file__).with_name('run_bc_ppo_probe.py')


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--budget',type=int,required=True);p.add_argument('--worker',action='store_true')
    a=p.parse_args();out=a.run.resolve();preflight=json.loads((out/'preflight-summary.json').read_text())
    if not preflight['numerical_pass'] or a.budget!=preflight['approved_controls_per_arm']:raise ValueError('measured preflight and explicit budget required')
    if not a.worker:
        with (out/'paired-supervisor.log').open('x') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--run',str(out),'--budget',str(a.budget),'--worker'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(out/'paired-supervisor-process.json',dict(pid=child.pid,started=time.time(),budget=a.budget,gpu=7));print(child.pid);return
    stages=[('bc',False,a.budget),('ppo',True,a.budget),('assessment-bc',False,3600),('assessment-ppo',False,3600)]
    manifest=dict(status='RUNNING',pid=os.getpid(),stages=[],started=time.time(),gpu=7)
    write(out/'paired-supervisor.json',manifest)
    hard_deadline=datetime.fromisoformat('2026-10-23T07:50:00+08:00').timestamp()
    for name,learning,budget in stages:
        registration=out/('assessment-registration.json' if name.startswith('assessment') else 'paired-registration.json')
        checkpoint=out/'ppo/residual-final.pt' if name=='assessment-ppo' else ROOT/'outputs/wam-rl-policy-v3/drift-intent-residual-20261022-r1/bc/bc.pt'
        cmd=[PYTHON,'-u',str(RUNNER),'--registration',str(registration),'--worker','0','--gpu','7','--output',str(out/name),'--residual-checkpoint',str(checkpoint),'--budget',str(budget),'--runtime-source',str(out/'preflight-ppo/runtime-code')]
        if learning:cmd+=['--ppo']
        with (out/(name+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,TMPDIR=str(ROOT/'tmp'),PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)
        receipt=dict(stage=name,pid=child.pid,command=cmd,started=time.time());manifest['stages'].append(receipt);write(out/'paired-supervisor.json',manifest)
        while child.poll() is None:
            if time.time()>hard_deadline:
                child.send_signal(signal.SIGINT);child.wait(timeout=90);manifest.update(status='DEADLINE_STOP',stage=name);write(out/'paired-supervisor.json',manifest);return 1
            time.sleep(30)
        receipt.update(returncode=child.returncode,finished=time.time());write(out/'paired-supervisor.json',manifest)
        if child.returncode:
            manifest.update(status='FAILED',stage=name);write(out/'paired-supervisor.json',manifest);return child.returncode
    summary=[PYTHON,str(Path(__file__).with_name('summarize_bc_ppo_probe.py')),'--bc',str(out/'assessment-bc'),'--ppo',str(out/'assessment-ppo'),'--training',str(out/'ppo'),'--output',str(out/'summary-final.json')]
    rc=subprocess.run(summary,cwd=ROOT).returncode
    manifest.update(status='COMPLETED' if rc==0 else 'FAILED',finished=time.time());write(out/'paired-supervisor.json',manifest)
    return rc


if __name__=='__main__':raise SystemExit(main())
