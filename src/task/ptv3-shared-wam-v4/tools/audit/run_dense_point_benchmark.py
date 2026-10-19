"""Bound the explicitly authorized main-job pause and restore it on exit."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def gpu():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used',
                                 '--format=csv,noheader,nounits'],text=True)
    return [dict(zip(('index','utilization','memory_mib'),map(int,line.split(',')))) for line in raw.splitlines()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main-run',type=Path,required=True);p.add_argument('--code-root',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--benchmark-script',type=Path,required=True);p.add_argument('--seconds',type=int,default=540)
    p.add_argument('--full-training-followup',action='store_true')
    a=p.parse_args()
    if not 60<=a.seconds<=540:raise ValueError('pause benchmark budget60..540 seconds')
    state=json.loads((a.main_run/'process.json').read_text())
    if state['status']!='RUNNING':raise ValueError('identified main job is not running')
    supervisor,worker=state['supervisor_pid'],state['torchrun_pid']
    for pid in [supervisor,worker]:
        if str(a.main_run).encode() not in Path(f'/proc/{pid}/cmdline').read_bytes():
            raise ValueError('main PID does not own expected run')
    group=os.getpgid(worker)
    if group!=worker:raise ValueError('expected isolated torchrun process group')
    table=subprocess.check_output(['ps','-eo','pid,ppid'],text=True).splitlines()[1:]
    parents={int(row.split()[0]):int(row.split()[1]) for row in table}
    owned={supervisor}
    for _ in range(20):
        owned.update(pid for pid,parent in parents.items() if parent in owned)
    for row in subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines():
        if int(row.strip()) not in owned:raise ValueError('unknown GPU0 task; preserve it')
    if not (a.main_run/'latest.pt').resolve(strict=True).is_file():raise ValueError('published checkpoint required')
    parent=(a.main_run/'latest.pt').resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='PREPARING',main_run=str(a.main_run),supervisor_pid=supervisor,torchrun_pgid=group,
        checkpoint=str(parent),before_gpu=gpu(),before_progress=json.loads((a.main_run/'progress.json').read_text()),
        started_at_utc=datetime.now(timezone.utc).isoformat(),cases=[])
    def save():
        (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    watchdog_code='''import os,signal,sys,time,json
from pathlib import Path
d=json.loads(sys.argv[1]);time.sleep(d['seconds'])
if Path('/proc/'+str(d['worker'])+'/cmdline').exists() and d['run'].encode() in Path('/proc/'+str(d['worker'])+'/cmdline').read_bytes():
 os.killpg(d['group'],signal.SIGCONT)
if Path('/proc/'+str(d['supervisor'])+'/cmdline').exists() and d['run'].encode() in Path('/proc/'+str(d['supervisor'])+'/cmdline').read_bytes():
 os.kill(d['supervisor'],signal.SIGCONT)
print('watchdog resumed verified main job',flush=True)
'''
    watchdog=subprocess.Popen([sys.executable,'-c',watchdog_code,json.dumps(dict(seconds=min(580,a.seconds+30),
        worker=worker,group=group,supervisor=supervisor,run=str(a.main_run)))],
        stdout=(a.output/'watchdog.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
    paused=False;child=None;started=time.monotonic();monitor_stop=threading.Event();monitor=None
    def monitor_gpu():
        with (a.output/'gpu.jsonl').open('w') as stream:
            while not monitor_stop.is_set():
                stream.write(json.dumps(dict(elapsed_s=time.monotonic()-started,gpu=gpu()))+'\n');stream.flush()
                monitor_stop.wait(10)
    def stop_requested(signum,frame):raise RuntimeError('benchmark orchestrator interrupted')
    signal.signal(signal.SIGTERM,stop_requested);signal.signal(signal.SIGINT,stop_requested)
    try:
        os.kill(supervisor,signal.SIGSTOP);paused=True
        for attempt in range(5):
            os.killpg(group,signal.SIGSTOP);time.sleep(2)
            sample=gpu();report.setdefault('pause_attempts',[]).append(sample);save()
            print(json.dumps(dict(stage='pause',attempt=attempt,gpu=sample)),flush=True)
            if all(row['utilization']<=5 for row in sample):break
            os.killpg(group,signal.SIGCONT);time.sleep(1)
        else:raise RuntimeError('could not quiesce queued CUDA/NCCL work; main job preserved')
        report['status']='BENCHMARKING';save()
        monitor=threading.Thread(target=monitor_gpu,daemon=True);monitor.start()
        cases=[('baseline-10mm','baseline',.01,128),('geometry-10mm','geometry',.01,128),
               ('geometry-5mm','geometry',.005,128),('vertices-5mm-c128','full',.005,128),
               ('vertices-5mm-c2048','full',.005,2048),('baseline-5mm','baseline',.005,128)]
        if a.full_training_followup:
            cases=[('vertices-train-5mm-c128-b64','full',.005,128),
                   ('vertices-train-5mm-c2048-b64','full',.005,2048),
                   ('baseline-5mm-b64','baseline',.005,128)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',REF2DEX_BENCH_CODE_ROOT=str(a.code_root))
        position=0
        while position<len(cases):
            name,mode,voxel,queries=cases[position];position+=1
            batch_size=64 if name.endswith('-b64') else 128
            remaining=a.seconds-(time.monotonic()-started)
            if remaining<25:
                report['budget_exhausted_before']=name;break
            output=a.output/(name+'.json')
            command=[sys.executable,str(a.benchmark_script),'--fixture',str(a.fixture),'--checkpoint',str(parent),
                '--output',str(output),'--case',name,'--mode',mode,'--voxel',str(voxel),
                '--queries',str(queries),'--batch',str(batch_size),'--hands','1','--warmup','3','--steps','8']
            if name=='baseline-10mm':command+=['--check-equivalence']
            if a.full_training_followup:command+=['--skip-batch-inference']
            print(json.dumps(dict(stage='case-start',case=name,remaining_s=remaining)),flush=True)
            log=(a.output/(name+'.log')).open('w')
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:code=child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=10)
                raise TimeoutError('benchmark pause budget reached')
            finally:log.close()
            if code:raise RuntimeError(f'{name} exited {code}; see owned log')
            entry=json.loads(output.read_text());report['cases'].append(entry);save()
            print(json.dumps(dict(stage='case-done',case=name,status=entry['status'],
                training=entry.get('training'),inference=entry.get('inference_batch1'))),flush=True)
            child=None
            if a.full_training_followup and entry['status']=='OOM' and batch_size==128:
                cases.append((name+'-b64',mode,voxel,queries))
                if not any(case[0]=='baseline-5mm-b64' for case in cases):
                    cases.append(('baseline-5mm-b64','baseline',.005,128))
        report['status']='COMPLETED'
    except BaseException as error:
        report.update(status='FAILED',error=repr(error))
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if paused:
            os.killpg(group,signal.SIGCONT);os.kill(supervisor,signal.SIGCONT)
        report['main_resumed_at_utc']=datetime.now(timezone.utc).isoformat()
        report['pause_seconds']=time.monotonic()-started
        monitor_stop.set()
        if monitor is not None:monitor.join(timeout=5)
        watchdog.terminate();watchdog.wait(timeout=10)
        report['after_gpu']=gpu();save()
        print(json.dumps(dict(stage='main-resumed',status=report['status'],pause_seconds=report['pause_seconds'],
                             error=report.get('error'))),flush=True)
    if report['status']!='COMPLETED':raise SystemExit(1)


if __name__=='__main__':main()
