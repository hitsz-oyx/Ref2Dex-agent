"""Wait for one declared checkpoint, then evaluate the fixed native Shadow panel."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--code-root',type=Path,required=True)
    p.add_argument('--step',type=int,default=2000);p.add_argument('--gpu',type=int,default=6)
    p.add_argument('--detach',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[5];run=a.run.resolve();code=a.code_root.resolve()
    if root/'outputs/shadow-finetune' not in run.parents or a.step!=2000:
        raise ValueError('This bounded watcher admits only the declared step2000 panel')
    if a.detach:
        with (run/'milestone2000-watch.log').open('x') as f:
            cmd=[sys.executable,str(Path(__file__).resolve())]+[x for x in sys.argv[1:] if x!='--detach']
            child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(watcher_pid=child.pid,run=str(run))));return
    checkpoint=run/'step2000.pt';started=time.monotonic();child=None
    status=dict(status='WAITING',pid=os.getpid(),step=2000,gpu=a.gpu,max_wait_seconds=3600)
    def write():
        part=run/'milestone2000-watch.json.part';part.write_text(json.dumps(status,indent=2)+'\n');part.replace(run/'milestone2000-watch.json')
    def signal_stop(number,frame):raise KeyboardInterrupt('User signal '+str(number))
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal_stop)
    write()
    try:
        while True:
            if time.monotonic()-started>3600:raise TimeoutError('Bounded checkpoint/GPU readiness wait')
            process=json.loads((run/'process.json').read_text())
            if process['status'] in ('FAILED','STOPPED'):raise RuntimeError('Training is not running')
            busy=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if checkpoint.is_file() and not busy:break
            time.sleep(10)
        manifest=json.loads((run/'manifest.json').read_text())
        cmd=[sys.executable,str(code/'src/task/shadow-finetune/tools/run/evaluate.py'),
            '--checkpoint',str(checkpoint),'--cache',manifest['data'],
            '--object-manifest',str(root/'outputs/shadow-finetune/pilot-panel-20261015-r1.json'),
            '--gpu',str(a.gpu),'--envs','16','--seed','216','--output',str(run/'native-step2000')]
        with (run/'native-step2000.log').open('x') as f:
            child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                env=dict(os.environ,REF2DEX_RUN_ROOT=str(root),TMPDIR=str(root/'tmp'),PYTHONDONTWRITEBYTECODE='1'))
        status.update(status='RUNNING',evaluation_pid=child.pid,command=cmd);write()
        evaluation_started=time.monotonic()
        while child.poll() is None:
            if time.monotonic()-evaluation_started>1200:raise TimeoutError('Native milestone evaluation deadline')
            time.sleep(5)
        if child.returncode:raise RuntimeError('Native evaluation failed; inspect fixed log')
        result=json.loads((run/'native-step2000/comparison.json').read_text())
        status.update(status='COMPLETED',result=result)
    except BaseException as error:
        status.update(status='FAILED',error=repr(error))
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        status['elapsed_s']=time.monotonic()-started;write()
    if status['status']!='COMPLETED':raise SystemExit(1)


if __name__=='__main__':main()
