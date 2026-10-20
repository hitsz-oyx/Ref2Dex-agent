"""One synchronized Shadow PPO across up to six co-located Inspire GPU ranks."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'
sys.path.insert(0,str(TASK/'src'))
from wam_rl_policy_v3.inspire_partitions import partition_inspire
MODEL='/home2/wyy/miniconda3/envs/graspenv/bin/python'
PHYSICS='/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python'


def write(path,value):
    part=path.with_suffix('.partial');part.write_text(json.dumps(value,indent=2)+'\n');part.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('output','cache','bc'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3,4,5])
    p.add_argument('--envs',type=int,default=256);p.add_argument('--updates',type=int,default=10000)
    p.add_argument('--seconds',type=int,default=172800);p.add_argument('--seed',type=int,default=246)
    p.add_argument('--reset-probe',action='store_true')
    p.add_argument('--wam-executor-mode',choices=['full_fp32','hand_fp32','hand_shared_tf32'],default='full_fp32')
    p.add_argument('--wam-batch',type=int,default=128);p.add_argument('--performance-migration-reason')
    p.add_argument('--state-transport',choices=['json','packed'],default='json')
    a=p.parse_args();out=a.output.resolve();a.cache=a.cache.resolve();a.bc=a.bc.resolve()
    if out.exists() or ROOT/'outputs/wam-rl-policy-v3' not in out.parents:raise ValueError('fresh Task run required')
    if not 2<=len(set(a.gpus))==len(a.gpus)<=6 or any(g not in range(8) for g in a.gpus):raise ValueError('two to six distinct GPUs required')
    if not 2<=a.envs<=256 or not 1<=a.updates<=10000 or not 60<=a.seconds<=172800:raise ValueError('bounded mixed training required')
    meta=json.loads((a.cache/'manifest.json').read_text());partitions=partition_inspire(meta['sequences'],len(a.gpus),a.seed)
    if any(len(r['object_names'])>a.envs for r in partitions):raise ValueError('every object needs an environment')
    for gpu in a.gpus:
        rows=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
        for row in rows.splitlines():
            pid,mem=[int(x.strip()) for x in row.split(',')]
            owner=subprocess.check_output(['ps','-o','user=','-p',str(pid)],text=True).strip()
            if owner!='monica' or mem>1024:raise RuntimeError('unexpected GPU occupant '+row)
    if shutil.disk_usage(ROOT).free<24*2**30:raise RuntimeError('24GiB startup reserve')
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));out.mkdir(parents=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    frozen=out/'runtime-task';shutil.copytree(TASK,frozen,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1',0));port=reservation.getsockname()[1]
    private=ROOT/'tmp'/('imix-'+str(os.getpid()));private.mkdir(mode=0o700)
    started=time.monotonic();children=[];streams=[];sockets=[];stop_deadline=None;last=-30
    state=dict(status='RUNNING',pid=os.getpid(),git_commit=commit,run_id=out.name,gpus=a.gpus,world_size=len(a.gpus),
        environments_per_rank=a.envs,environments=len(a.gpus)*a.envs,objects=sum(len(r['object_names']) for r in partitions),
        train_sequences=sum(len(r['sequence_names']) for r in partitions),updates=a.updates,seed=a.seed,resume=str(a.resume),
        max_seconds=a.seconds,max_bytes=3*2**30,rendezvous_port=port,children=[],partitions=partitions,
        training='one synchronized official Shadow PPO; co-located frozen WAM and pure native physics; no hand-ref actor',
        inference=dict(mode=a.wam_executor_mode,batch=a.wam_batch,state_transport=a.state_transport),performance_migration_reason=a.performance_migration_reason,
        started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    write(out/'process.json',state)
    def launch(command,name,rank,gpu):
        stream=(out/(name+'.log')).open('x');streams.append(stream)
        env=dict(os.environ,REF2DEX_RUN_ROOT=str(ROOT),REF2DEX_CODE_COMMIT=commit,PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(proc);state['children'].append(dict(pid=proc.pid,name=name,rank=rank,gpu=gpu,command=command))
        write(out/'process.json',state);return proc
    def guard():
        nonlocal last,stop_deadline
        elapsed=time.monotonic()-started
        if elapsed>a.seconds or shutil.disk_usage(ROOT).free<20*2**30:raise RuntimeError('mixed time/storage floor')
        if (out/'STOP').exists():
            if stop_deadline is None:
                stop_deadline=time.monotonic()+90;state['status']='STOPPING';write(out/'process.json',state)
            if time.monotonic()>stop_deadline:raise TimeoutError('graceful stop deadline; periodic checkpoints retained')
        if elapsed-last>=30:
            last=elapsed;size=sum(f.stat().st_size for f in out.rglob('*') if f.is_file() and not f.is_symlink())
            if size>state['max_bytes']:raise RuntimeError('3GiB mixed output budget')
            gpu=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,a.gpus)),'--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
            with (out/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(elapsed_s=elapsed,gpus=gpu,bytes=size))+'\n')
            print(json.dumps(dict(status=state['status'],elapsed_s=elapsed,gpus=gpu)),flush=True)
    def stop(*_):
        (out/'STOP').write_text('owned supervisor signal\n')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        servers=[]
        for rank,gpu in enumerate(a.gpus):
            folder=out/('rank-%d'%rank);folder.mkdir();manifest=folder/'objects.json';write(manifest,partitions[rank])
            sock=private/('s%d'%rank);sockets.append(sock)
            command=[MODEL,'-u',str(frozen/'tools/run/inspire_shadow_ppo.py'),'--cache',str(a.cache),'--bc',str(a.bc),
                '--output',str(folder/'policy'),'--socket',str(sock),'--gpu',str(gpu),'--updates',str(a.updates),'--seconds',str(a.seconds),
                '--seed',str(a.seed),'--rank',str(rank),'--world-size',str(len(a.gpus)),'--rendezvous','tcp://127.0.0.1:'+str(port),
                '--stop-file',str(out/'STOP'),'--wam-executor-mode',a.wam_executor_mode,'--wam-batch',str(a.wam_batch),
                '--state-transport',a.state_transport]
            if a.performance_migration_reason:command+=['--performance-migration-reason',a.performance_migration_reason]
            if a.resume:command+=['--resume',str(a.resume.resolve())]
            if a.reset_probe:command+=['--reset-probe']
            servers.append(launch(command,'policy-%d'%rank,rank,gpu))
        deadline=time.monotonic()+120
        while not all(s.exists() for s in sockets):
            guard()
            if any(c.poll() is not None for c in servers) or time.monotonic()>deadline:raise RuntimeError('policy listen failed')
            time.sleep(1)
        for rank,gpu in enumerate(a.gpus):
            folder=out/('rank-%d'%rank)
            command=[PHYSICS,'-u',str(frozen/'tools/run/inspire_physics_host.py'),'--cache',str(a.cache),
                '--output',str(folder/'physics'),'--socket',str(sockets[rank]),'--server-pid',str(servers[rank].pid),
                '--gpu',str(gpu),'--envs',str(a.envs),'--seconds',str(a.seconds),'--seed',str(a.seed+rank),
                '--object-manifest',str(folder/'objects.json'),'--state-transport',a.state_transport]
            launch(command,'physics-%d'%rank,rank,gpu)
        while True:
            guard();codes=[c.poll() for c in children]
            if any(c not in (None,0) for c in codes):raise RuntimeError('mixed rank peer failed '+str(codes))
            if all(c==0 for c in codes):break
            time.sleep(2)
        state['status']='STOPPED' if stop_deadline else 'COMPLETED'
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));raise
    finally:
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for stream in streams:stream.close()
        for sock in sockets:sock.unlink(missing_ok=True)
        state.update(elapsed_s=time.monotonic()-started,exit_codes=[c.returncode for c in children]);write(out/'process.json',state)


if __name__=='__main__':main()
