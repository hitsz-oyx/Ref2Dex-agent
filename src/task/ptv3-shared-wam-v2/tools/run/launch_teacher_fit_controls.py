"""Two owned local GPUs, four bounded teacher-input optimization controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[5]
TASK=ROOT/'src/task/ptv3-shared-wam-v2'


def write(path,data):
    temp=path.with_suffix('.json.part');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve()
    if out.exists() or out.parent!=ROOT/'outputs/ptv3-shared-wam-v2':raise ValueError('fresh task output required')
    if shutil.disk_usage(ROOT).free<20*2**30:raise RuntimeError('20GiB free disk reserve')
    out.mkdir();started=time.monotonic()
    status=dict(status='RUNNING',pid=os.getpid(),run_id=out.name,local_gpus=[0,1],reserved_remote_gpus=4,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        limits=dict(per_arm_seconds=1200,total_wall_seconds=2500,total_bytes=5_000_000_000),runs=[])
    write(out/'process.json',status)
    def worker(gpu,arm,schedule):
        name=arm+'-'+schedule;folder=out/name
        cmd=[sys.executable,str(TASK/'tools/run/probe_full_teacher.py'),'--cache',str(a.cache.resolve()),'--output',str(folder),
            '--card',str(TASK/'docs/experiments/probes/P-20261015-teacher-fit-generalization.md'),
            '--experiment-id','P-20261015-teacher-fit-generalization','--arm',arm,'--gpu',str(gpu),'--updates','24000',
            '--input-normalization','teacher_RMS','--lr-schedule',schedule,'--artifact-budget-bytes','1000000000']
        with (out/(name+'.log')).open('w') as log:
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(out/(name+'-launch.json'),dict(pid=process.pid,gpu=gpu,cmd=cmd))
            try:code=process.wait(timeout=1250)
            except BaseException:
                if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise
        if code:raise RuntimeError('worker failed '+name+'; see retained log')
        return dict(arm=arm,schedule=schedule,output=str(folder),result=json.loads((folder/'result.json').read_text()))
    try:
        for schedule in ['constant','cosine_tail']:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(worker,gpu,arm,schedule) for gpu,arm in [(0,'full_relative'),(1,'current_relative')]]
                status['runs'].extend(future.result() for future in futures)
            status.update(elapsed_s=time.monotonic()-started,phase=schedule);write(out/'process.json',status)
            if sum(f.stat().st_size for f in out.rglob('*') if f.is_file())>5_000_000_000:raise RuntimeError('aggregate output budget')
        status.update(status='COMPLETED',elapsed_s=time.monotonic()-started);write(out/'process.json',status)
        write(out/'result.json',dict(status='COMPLETED',runs=status['runs']))
    except BaseException as error:
        status.update(status='FAILED',error=repr(error),elapsed_s=time.monotonic()-started);write(out/'process.json',status);raise


if __name__=='__main__':main()
