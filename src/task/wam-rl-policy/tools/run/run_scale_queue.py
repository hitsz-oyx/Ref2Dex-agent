"""Bounded owned evaluation queue; each job names immutable input readiness files."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queue',type=Path,required=True);a=p.parse_args()
    spec=json.loads(a.queue.read_text());out=Path(spec['output']);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();child=None;stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    state=dict(status='RUNNING',pid=os.getpid(),jobs=[],max_seconds=spec['max_seconds'])
    def save():
        state['elapsed_s']=time.monotonic()-started;part=out/'process.json.part';part.write_text(json.dumps(state,indent=2)+'\n');part.replace(out/'process.json')
    def boundary():
        if stop[0] or (out/'STOP').exists():raise InterruptedError('Evaluation queue STOP')
        if time.monotonic()-started>spec['max_seconds']:raise TimeoutError('Queue wall budget')
    try:
        for job in spec['jobs']:
            row=dict(name=job['name'],status='WAITING',command=job['command']);state['jobs'].append(row);save()
            while True:
                boundary();ready=all(Path(path).exists() for path in job.get('ready_files',[]))
                for path in job.get('completed_manifests',[]):
                    f=Path(path)
                    if not f.exists():ready=False;continue
                    status=json.loads(f.read_text())['status']
                    if status=='FAILED':raise RuntimeError('Dependency failed: '+path)
                    ready&=status=='COMPLETED'
                if ready:break
                time.sleep(15)
            row['status']='RUNNING';save()
            with (out/(job['name']+'.log')).open('x') as log:
                child=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                row['pid']=child.pid;save()
                while child.poll() is None:boundary();time.sleep(10)
                row.update(status='COMPLETED' if child.returncode==0 else 'FAILED',exit_code=child.returncode);save()
                if child.returncode:raise RuntimeError('Evaluation job failed: '+job['name'])
            child=None
        state['status']='COMPLETED'
    except BaseException as error:state.update(status='FAILED',error=repr(error));raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        save()


if __name__=='__main__':main()
