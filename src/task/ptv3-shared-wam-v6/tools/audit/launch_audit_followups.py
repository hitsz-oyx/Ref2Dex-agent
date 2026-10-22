"""Detached bounded followups after owned inference workers release their GPU.

Never controls training or unknown processes. Idle GPU checks cover only the
explicitly authorized0..3; an occupied preferred device is waited on, not killed.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def write(path,value):
    tmp=path.with_suffix('.json.part');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    if a.gpu not in range(4):raise ValueError('GPU0..3 only')
    out=a.output.resolve();root=Path.cwd();started=time.monotonic();python='/home2/wyy/miniconda3/envs/graspenv/bin/python'
    receipt=out/('mesh-controller%d.json'%a.shard)
    while time.monotonic()-started<7200:
        if (out/'STOP').exists() or shutil.disk_usage(out).free<20*2**30:raise RuntimeError('owned STOP/disk boundary')
        if (out/('complete-worker%d.json'%a.shard)).exists():
            result=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
            used,util=map(int,result.split(','))
            if used<256 and util<5:break
        write(receipt,dict(status='WAITING_FOR_OWNED_WORKER_AND_IDLE_GPU',gpu=a.gpu,elapsed_s=time.monotonic()-started))
        time.sleep(30)
    else:raise TimeoutError('bounded2h wait for own inference completion/idle preferred GPU')
    env={k:os.environ[k] for k in ('PATH','HOME','LD_LIBRARY_PATH','CONDA_PREFIX') if k in os.environ};env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TMPDIR=str(root/'tmp/goal-chain-audit-20261022-r1'))
    cmd=[python,'-u','src/task/ptv3-shared-wam-v6/tools/audit/audit_large_mesh_contact.py','--output',str(out),'--phase','predict','--shard',str(a.shard)]
    with open(out/('mesh-endpoints-worker%d.log'%a.shard),'a') as f:child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    write(receipt,dict(status='RUNNING',pid=child.pid,command=cmd,physical_gpu=a.gpu,started=datetime.datetime.now().astimezone().isoformat(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),max_seconds=3600))
    code=child.wait(timeout=3700)
    write(receipt,dict(status='COMPLETED' if code==0 else 'FAILED',returncode=code,pid=child.pid,command=cmd,physical_gpu=a.gpu))


if __name__=='__main__':main()
