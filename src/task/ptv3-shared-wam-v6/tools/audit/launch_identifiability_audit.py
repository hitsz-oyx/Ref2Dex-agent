"""Resume inference only in available owner-authorized GPU0..3 headroom."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def write(path,value):
    tmp=path.with_suffix('.json.part');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args()
    original=json.loads(a.receipt.read_text());run=a.receipt.parent;command=original['command'];out=Path(command[command.index('--output')+1]);frozen=Path(command[3]).parents[5]
    # Import only a narrow, noncredential environment allowlist.
    env={k:os.environ[k] for k in ('PATH','HOME','LD_LIBRARY_PATH','CONDA_PREFIX') if k in os.environ}
    env.update(TMPDIR=str(run),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',REF2DEX_RUN_ROOT=str(a.run_root),REF2DEX_V6_V5_CODE_ROOT=str(frozen),REF2DEX_V5_BASE_CODE_ROOT=str(frozen),XDG_CACHE_HOME=str(run/'cache'),CUDA_CACHE_PATH=str(run/'cuda-cache'))
    start=time.monotonic();state=dict(status='WAITING',supervisor_pid=os.getpid(),started_at=time.time(),scope='inference only;GPU0..3;never controls training',attempts=[])
    while time.monotonic()-start<1800:
        if (out/'STOP').exists():state['status']='STOPPED';break
        raw=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
        telemetry=[]
        for line in raw.splitlines():
            i,m,u=[int(v.strip()) for v in line.split(',')];telemetry.append(dict(gpu=i,free_mib=m,utilization=u))
        state['telemetry']=telemetry;state['elapsed_s']=time.monotonic()-start
        candidates=[r for r in telemetry if r['free_mib']>=3584 and r['utilization']<=50]
        if not candidates:
            state['status']='WAITING';write(out/'resource-controller.json',state);time.sleep(10);continue
        pick=max(candidates,key=lambda r:r['free_mib']);number=len(state['attempts'])+1
        record=dict(attempt=number,physical_gpu=pick['gpu'],started_at=time.time(),command=command);state['attempts'].append(record)
        state.update(status='RUNNING',physical_gpu=pick['gpu']);write(out/'resource-controller.json',state)
        logpath=run/('evaluate-resume-%02d.log'%number)
        with logpath.open('x') as log:
            child=subprocess.Popen(command,env=dict(env,CUDA_VISIBLE_DEVICES=str(pick['gpu'])),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            record['pid']=child.pid;write(out/'resource-controller.json',state)
            while child.poll() is None:
                if time.monotonic()-start>1800 or (out/'STOP').exists():
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30);break
                time.sleep(5)
        record.update(returncode=child.returncode,ended_at=time.time(),log=str(logpath));write(out/'resource-controller.json',state)
        if child.returncode==0:state['status']='COMPLETED';break
        text=logpath.read_text()
        if not any(x in text for x in ('device headroom','out of memory')):
            state.update(status='FAILED',error='unexpected inference error;inspect '+str(logpath));break
        state['status']='WAITING';write(out/'resource-controller.json',state);time.sleep(10)
    else:state['status']='WAITING_BUDGET_EXHAUSTED'
    state['elapsed_s']=time.monotonic()-start;write(out/'resource-controller.json',state)


if __name__=='__main__':main()
