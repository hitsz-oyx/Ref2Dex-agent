"""Run bounded v5 real-data CUDA checks while preserving the authorized WAM job."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def gpu():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
    return [dict(zip(('index','utilization','memory_mib'),map(int,r.split(',')))) for r in rows.splitlines()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('main-run','code-root','base-code-root','recipe','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seconds',type=int,default=300);p.add_argument('--git-commit',required=True)
    a=p.parse_args()
    if not 60<=a.seconds<=540:raise ValueError('bounded60..540 second pause required')
    state=json.loads((a.main_run/'process.json').read_text())
    supervisor,worker=state['supervisor_pid'],state['torchrun_pid']
    if state['status']!='RUNNING':raise ValueError('main job must be running')
    for pid in (supervisor,worker):
        if str(a.main_run).encode() not in Path(f'/proc/{pid}/cmdline').read_bytes():raise ValueError('main process ownership changed')
    group=os.getpgid(worker)
    if group!=worker:raise ValueError('main torchrun group must be isolated')
    parents={int(r.split()[0]):int(r.split()[1]) for r in subprocess.check_output(['ps','-eo','pid,ppid'],text=True).splitlines()[1:]}
    owned={supervisor}
    for _ in range(20):owned.update(pid for pid,parent in parents.items() if parent in owned)
    pids=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
    if any(int(pid) not in owned for pid in pids):raise ValueError('unknown GPU0 owner; preserve existing tasks')
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='PREPARING',supervisor_pid=supervisor,torchrun_pid=worker,main_run=str(a.main_run),
        git_commit=a.git_commit,before_gpu=gpu(),before_progress=json.loads((a.main_run/'progress.json').read_text()),cases=[])
    def save():(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    watchdog_code='''import os,signal,sys,time,json
from pathlib import Path
d=json.loads(sys.argv[1]);time.sleep(d['seconds'])
for pid in (d['worker'],d['supervisor']):
 p=Path('/proc')/str(pid)/'cmdline'
 if p.exists() and d['run'].encode() in p.read_bytes():
  if pid==d['worker']:os.killpg(pid,signal.SIGCONT)
  else:os.kill(pid,signal.SIGCONT)
'''
    watchdog=subprocess.Popen([sys.executable,'-c',watchdog_code,json.dumps(dict(seconds=a.seconds+30,
        worker=worker,supervisor=supervisor,run=str(a.main_run)))],start_new_session=True)
    started=time.monotonic();child=None;paused=False;stop=threading.Event()
    def monitor():
        with (a.output/'gpu.jsonl').open('w') as f:
            while not stop.is_set():f.write(json.dumps(dict(elapsed_s=time.monotonic()-started,gpu=gpu()))+'\n');f.flush();stop.wait(10)
    thread=threading.Thread(target=monitor,daemon=True)
    def interrupted(*_):raise RuntimeError('v5 smoke orchestrator interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        os.kill(supervisor,signal.SIGSTOP);paused=True
        for _ in range(5):
            os.killpg(group,signal.SIGSTOP);time.sleep(2)
            if all(r['utilization']<=5 for r in gpu()):break
            os.killpg(group,signal.SIGCONT);time.sleep(1)
        else:raise RuntimeError('queued work did not quiesce')
        thread.start();report['status']='TESTING';save()
        task=a.code_root/'src/task/ptv3-shared-wam-v5'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',REF2DEX_V5_BASE_CODE_ROOT=str(a.base_code_root),
            OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        for batch,steps in ((2,4),(64,8),(128,8)):
            remaining=a.seconds-(time.monotonic()-started)
            if remaining<30:break
            output=a.output/f'batch{batch}'
            command=[sys.executable,str(task/'tools/run/train_pretrain.py'),'--config',str(task/'configs/pretrain_joint21_surface2048_scale1024.json'),
                '--manifest',str(a.recipe/'manifest.json'),'--stats',str(a.recipe/'stats.json'),'--output',str(output),
                '--smoke','--steps',str(steps),'--batch-size',str(batch),'--git-commit',a.git_commit]
            log=(a.output/f'batch{batch}.log').open('w')
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
            try:code=child.wait(timeout=remaining)
            finally:log.close()
            entry=dict(batch_size=batch,exit_code=code)
            if (output/'result.json').exists():entry.update(json.loads((output/'result.json').read_text()))
            report['cases'].append(entry);save();print(json.dumps(entry),flush=True)
            child=None
            if code and batch==2:raise RuntimeError('small real-data v5 smoke failed; see owned log')
        report['status']='COMPLETED' if all(c['exit_code']==0 for c in report['cases']) else 'PARTIAL'
    except BaseException as e:report.update(status='FAILED',error=repr(e))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if paused:os.killpg(group,signal.SIGCONT);os.kill(supervisor,signal.SIGCONT)
        stop.set()
        if thread.is_alive():thread.join(timeout=3)
        watchdog.terminate();watchdog.wait(timeout=5)
        report['pause_seconds']=time.monotonic()-started;report['after_gpu']=gpu();save()
        print(json.dumps(dict(status=report['status'],main_resumed=True,pause_seconds=report['pause_seconds'])),flush=True)
    if report['status']=='FAILED':raise SystemExit(1)


if __name__=='__main__':main()
