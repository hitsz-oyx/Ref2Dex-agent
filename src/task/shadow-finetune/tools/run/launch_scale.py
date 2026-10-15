"""One-GPU, one-hour owned supervisor for the frozen small/full Probe."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('REF2DEX_RUN_ROOT',TASK.parents[2])).resolve()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--gpu',type=int,default=6)
    p.add_argument('--detach',action='store_true');a=p.parse_args();run=a.run.resolve()
    if ROOT/'outputs/shadow-finetune' not in run.parents or (run/'process.json').exists():
        raise ValueError('prepared Task output without previous execution required')
    if a.detach:
        with (run/'supervisor.log').open('x') as f:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--run',str(run),'--gpu',str(a.gpu)],
                                   stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(supervisor_pid=child.pid,run=str(run),gpu=a.gpu)));return
    def gpu_pids(gpu):
        text=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True)
        return [int(x) for x in text.split() if x.isdigit()]
    if gpu_pids(a.gpu):raise RuntimeError('selected GPU occupied')
    commands=json.loads((run/'commands.json').read_text());controls=json.loads((run/'controls.json').read_text())
    if controls['settings']['updates']>2000:raise ValueError('bounded Probe only')
    child=None;stop=[];started=time.monotonic()
    status=dict(status='RUNNING',supervisor_pid=os.getpid(),gpu=a.gpu,stages=[],max_seconds=3600,
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TASK,text=True).strip(),
                code_root=str(TASK.parents[2]),authorization='user permits one temporarily idle card for small Shadow tests')
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda number,frame:stop.append(number))
    frozen=list((TASK/'src').rglob('*.py'))+list((TASK/'tools').rglob('*.py'))
    stamps={str(f):[f.stat().st_size,f.stat().st_mtime_ns] for f in frozen}
    def write():
        partial=run/'process.json.part';partial.write_text(json.dumps(status,indent=2)+'\n');partial.replace(run/'process.json')
    def execute(name,command):
        nonlocal child
        cmd=list(command)
        if name!='summary':
            cmd[1]=str(TASK/'tools/run'/Path(cmd[1]).name)
            cmd[cmd.index('--gpu')+1]=str(a.gpu)
        stage=dict(name=name,command=cmd,status='RUNNING');status['stages'].append(stage);write()
        with (run/(name+'.log')).open('x') as f:
            child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                    env=dict(os.environ,REF2DEX_RUN_ROOT=str(ROOT),PYTHONDONTWRITEBYTECODE='1',
                             TMPDIR=str(ROOT/'tmp'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
        stage['pid']=child.pid;write()
        while child.poll() is None:
            if stop:raise RuntimeError('user signal')
            if time.monotonic()-started>3600:raise TimeoutError('one-hour complete comparison budget')
            if shutil.disk_usage(run).free<20*2**30:raise RuntimeError('20GiB free floor')
            if int(subprocess.check_output(['du','-sb',str(run)],text=True).split()[0])>14*10**9:
                raise RuntimeError('14GB artifact budget')
            for path,stamp in stamps.items():
                if [Path(path).stat().st_size,Path(path).stat().st_mtime_ns]!=stamp:
                    raise RuntimeError('frozen Task code changed')
            for pid in gpu_pids(a.gpu):
                try:group=os.getpgid(pid)
                except ProcessLookupError:continue
                if group!=child.pid:raise RuntimeError('foreign selected-GPU process')
            if a.gpu==6 and gpu_pids(7):raise RuntimeError('GPU7 reserve occupied; release own temporary card')
            time.sleep(5)
        stage.update(status='COMPLETED' if child.returncode==0 else 'FAILED',exit_code=child.returncode);write()
        if child.returncode:raise RuntimeError('stage failed: '+name)
    write()
    try:
        for arm in ('small','full'):
            execute('train-'+arm,commands[arm]['train'])
            execute('native-final-'+arm,commands[arm]['evaluate_final'])
        execute('summary',[sys.executable,str(TASK/'tools/audit/summarize_scale.py'),'--run',str(run)])
        status['status']='COMPLETED'
    except BaseException as error:status.update(status='FAILED',error=repr(error))
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        status['elapsed_s']=time.monotonic()-started;write()
    if status['status']!='COMPLETED':raise SystemExit(1)


if __name__=='__main__':main()
