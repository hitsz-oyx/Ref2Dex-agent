"""Bounded owned-worker supervisor for C0 and conditional overnight C1."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
spec=importlib.util.spec_from_file_location('owned_gpu_helpers',ROOT/'src/task/cm-pointflow-effect-pretrain/tools/run/launch_pointworld_action_ddp.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)


def write(path,value):
    part=path.with_suffix('.json.part');part.write_text(json.dumps(value,indent=2)+'\n');part.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    out=a.output.resolve();gpus=list(map(int,a.gpus.split(',')))
    if len(set(gpus))!=len(gpus) or not 1<=len(gpus)<=4 or ROOT/'outputs/goal-point-wam' not in out.parents:
        raise ValueError('owned output and1..4 distinct local GPUs required')
    if out.exists():raise FileExistsError('fresh supervisor output required')
    if helpers.gpu_processes(gpus):raise RuntimeError('assigned GPU occupied')
    out.mkdir();scratch=ROOT/'tmp/wam-effect-consistency';scratch.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus,TMPDIR=str(scratch),TRITON_CACHE_DIR=str(scratch/'triton'),
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    deadline=time.time()+(600 if a.smoke else 28800)
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node='+str(len(gpus)),
             str(TASK/'tools/run/train_effect_consistency.py'),'--output',str(out/'train'),'--deadline',str(deadline)]
    if a.smoke:command.append('--smoke')
    sources=[TASK/'tools/run/train_effect_consistency.py',TASK/'src/goal_point_wam/effect_consistency.py',
             TASK/'src/goal_point_wam/model.py',ROOT/'src/task/trajectory-policy/src/trajectory_policy/wam_effect_data.py']
    versions={str(p):(p.stat().st_mtime_ns,p.stat().st_size) for p in sources}
    stop=[False]
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    with (out/'console.log').open('w') as log:
        proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status=dict(status='TRAINING',pid=os.getpid(),torchrun_pid=proc.pid,gpus=gpus,deadline=deadline,
                    command=command,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        sent=None;checked=0;monitored=0
        while proc.poll() is None:
            foreign=[pid for pid in helpers.gpu_processes(gpus) if not helpers.belongs_to(pid,proc.pid)]
            drift=any(not Path(p).exists() or (Path(p).stat().st_mtime_ns,Path(p).stat().st_size)!=v for p,v in versions.items())
            if time.time()-checked>=300:
                status['artifact_bytes']=sum(int(line.split()[0]) for line in subprocess.check_output(['du','-sb',str(ROOT/'outputs'),str(ROOT/'tmp')],text=True).splitlines())
                status['owned_bytes']=int(subprocess.check_output(['du','-sb',str(out)],text=True).split()[0])
                checked=time.time()
            if time.time()-monitored>=60:
                status['gpu_snapshot']=subprocess.check_output(['nvidia-smi','-i',a.gpus,'--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
                monitored=time.time()
            reason=('foreign_gpu_process' if foreign else 'source_changed' if drift else 'storage_cap' if status.get('artifact_bytes',0)>=400_000_000_000
                    or status.get('owned_bytes',0)>=8_000_000_000 else 'user_stop' if stop[0] else 'deadline' if time.time()>=deadline else None)
            if reason and sent is None:
                status['stop_reason']=reason;sent=time.time()
                ranks=out/'train/ranks.json'
                if ranks.exists():
                    for worker in json.loads(ranks.read_text()):
                        if helpers.belongs_to(worker['pid'],proc.pid):
                            try:os.kill(worker['pid'],signal.SIGUSR1)
                            except ProcessLookupError:pass
                else:proc.terminate()
            if sent and time.time()-sent>180:
                # This process group was created here and contains only this run.
                os.killpg(proc.pid,signal.SIGTERM)
            status['updated_at']=time.time();write(out/'group_status.json',status);time.sleep(10)
        status.update(exit_code=proc.returncode,status='COMPLETED' if proc.returncode==0 else 'FAILED',ended_at=time.time())
        result=out/'train/result.json'
        if result.exists():
            status['training_result']=json.loads(result.read_text());status['status']=status['training_result']['status']
        write(out/'group_status.json',status)
    if proc.returncode:raise SystemExit(proc.returncode)


if __name__=='__main__':main()
