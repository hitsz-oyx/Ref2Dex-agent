"""Wait for this task's bounded parent queue, then qualify its endpoint once."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from consequence_evaluator.contracts import is_within


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--seed',type=int,default=290)
    a=p.parse_args()
    parent,output=a.parent.resolve(),a.output.resolve()
    if any(not is_within(x,ROOT/'outputs/consequence-evaluator') for x in (parent,output)) or output.exists():
        p.error('owned parent queue and fresh qualification output required')
    initial=json.loads((parent/'run_manifest.json').read_text())
    pid=initial['pid'];cmdline=Path('/proc')/str(pid)/'cmdline'
    queue_path=str(TASK/'tools/run/queue_baseline_rebuild.py')
    if cmdline.exists():
        arguments=cmdline.read_bytes().decode().split('\0')
        parent_cwd=(cmdline.parent/'cwd').resolve()
        if not any(str((parent_cwd/arg).resolve())==queue_path for arg in arguments if arg.endswith('.py')):
            raise ValueError('parent PID is not this task queue')
    qualifier=TASK/'tools/run/qualify_parent.py'
    frozen=hashlib.sha256(qualifier.read_bytes()).hexdigest()
    status_path=output.with_name(output.name+'-queue.json')
    if status_path.exists():
        raise ValueError('fresh qualification queue record required')
    record=dict(status='WAITING',pid=os.getpid(),parent_pid=pid,parent=str(parent),
                output=str(output),physical_gpu=a.gpu,seed=a.seed,
                qualifier_sha256=frozen,wait_budget_s=3600,evaluation_budget_s=900,
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    started=time.monotonic();child=None
    def save():
        record['elapsed_s']=time.monotonic()-started
        status_path.write_text(json.dumps(record,indent=2)+'\n')
    def stopped(signum,frame):
        raise KeyboardInterrupt('qualification queue stopped')
    signal.signal(signal.SIGTERM,stopped);save()
    try:
        while True:
            state=json.loads((parent/'run_manifest.json').read_text())
            if state['status']=='PARENT_TRAINED_EVALUATION_PENDING':break
            if state['status'] in ('FAILED','TIMED_OUT'):
                raise RuntimeError('parent queue failed: '+str(state.get('error')))
            if not cmdline.exists():raise RuntimeError('parent disappeared without successful terminal record')
            if time.monotonic()-started>3600:raise TimeoutError('fixed parent wait deadline')
            if hashlib.sha256(qualifier.read_bytes()).hexdigest()!=frozen:raise RuntimeError('qualification code drift')
            time.sleep(10)
        trained_run=Path(state.get('trained_run_dir', str(parent/'parent_s1'))).resolve()
        if not is_within(trained_run,parent):
            raise ValueError('trained endpoint must belong to the owned queue')
        command=[sys.executable,str(qualifier),'--run-dir',str(trained_run),
                 '--output',str(output),'--gpu',str(a.gpu),'--seed',str(a.seed),'--seconds','900']
        record.update(status='RUNNING',command=command);save()
        with output.with_name(output.name+'.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record['child_pid']=child.pid;save()
            child.wait(timeout=930)
        if child.returncode:raise RuntimeError('qualification failed: '+str(child.returncode))
        result=json.loads((output/'qualification.json').read_text())
        record.update(status='COMPLETED',qualified_episodes=result['qualified_episodes'],
                      data_readiness_pass=result['data_readiness_pass'])
    except BaseException as error:
        record.update(status='FAILED',error=repr(error));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        save()


if __name__=='__main__':main()
