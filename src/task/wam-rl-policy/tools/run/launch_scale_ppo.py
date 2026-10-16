"""Bounded synchronous multi-GPU Shadow PPO gang with owned process supervision."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panels',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',default='1,2,3,4');p.add_argument('--envs',type=int,default=64)
    p.add_argument('--updates',type=int,default=200);p.add_argument('--seconds',type=int,default=43200)
    p.add_argument('--bc-kind',choices=('wam','mlp'),default='wam');p.add_argument('--resume',type=Path)
    p.add_argument('--seed',type=int,default=245);p.add_argument('--policy-seed',type=int,default=246)
    p.add_argument('--exploration-seed',type=int,default=247)
    p.add_argument('--initial-log-std',type=float,default=-2.5)
    p.add_argument('--outside-support-zero',action='store_true')
    p.add_argument('--rollout-steps',type=int,default=199)
    p.add_argument('--observation-kind',choices=('native','normalized','bc-hidden'),default='native');a=p.parse_args()
    out=a.output.resolve();gpus=[int(x) for x in a.gpus.split(',')]
    if out.exists() or ROOT/'outputs/wam-rl-policy' not in out.parents:raise ValueError('Fresh Task output required')
    if len(gpus)!=len(set(gpus)) or not 1<=len(gpus)<=4:raise ValueError('1..4 unique GPUs required')
    for gpu in gpus:
        if subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied: '+str(gpu))
    if not os.environ.get('REF2DEX_REMOTE'):
        available=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        idle=[int(line.split(',')[0]) for line in available.splitlines() if int(line.split(',')[1])<100]
        if len(set(idle)-set(gpus))<2:raise RuntimeError('Keep two local GPUs idle')
    panel=json.loads(a.panels.read_text());manifests=panel['train'][:len(gpus)]
    if len(manifests)!=len(gpus):raise ValueError('One TRAIN manifest per rank required')
    out.mkdir(parents=True)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    children=[];logs=[];start=time.monotonic();stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    process=dict(status='RUNNING',pid=os.getpid(),git_commit=os.environ.get('REF2DEX_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        gpus=gpus,budget_seconds=a.seconds,global_envs=a.envs*len(gpus),objects=sum(len(json.loads(Path(m).read_text())['objects']) for m in manifests),
        world_size=len(gpus),rendezvous='tcp://127.0.0.1:%d'%port,ranks=[])
    def save():
        part=out/'process.json.part';part.write_text(json.dumps(process,indent=2)+'\n');part.replace(out/'process.json')
    try:
        for rank,(gpu,manifest) in enumerate(zip(gpus,manifests)):
            target=out/('rank%02d'%rank);(target/'tmp').mkdir(parents=True)
            command=[sys.executable,'-u',str(TASK/'tools/run/shadow_ppo.py'),'--worker','--checkpoint',str(a.checkpoint.resolve()),
                '--cache',str(a.cache.resolve()),'--object-manifest',manifest,'--output',str(target),'--mode','train','--gpu',str(gpu),
                '--envs',str(a.envs),'--updates',str(a.updates),'--seconds',str(a.seconds),'--bc-kind',a.bc_kind,
                '--experiment-id','P-20261016-shadow-scale-continuous','--rank',str(rank),'--world-size',str(len(gpus)),
                '--rendezvous',process['rendezvous'],'--seed',str(a.seed+rank),'--policy-seed',str(a.policy_seed),
                '--exploration-seed',str(a.exploration_seed),'--initial-log-std',str(a.initial_log_std),'--trace-every','50']
            command+=['--observation-kind',a.observation_kind]
            command+=['--rollout-steps',str(a.rollout_steps)]
            if a.resume:command+=['--rl-checkpoint',str((a.resume/('rank%02d'%rank)/'latest.pt').resolve())]
            if a.outside_support_zero:command+=['--outside-support-zero']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TMPDIR=str(target/'tmp'),TORCH_EXTENSIONS_DIR=os.environ.get('TORCH_EXTENSIONS_DIR',str(ROOT/'tmp/shadow-isaac-extensions')),
                MAX_JOBS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
            env['LD_LIBRARY_PATH']=str(Path(sys.executable).resolve().parents[1]/'lib')+':'+env.get('LD_LIBRARY_PATH','')
            env['PATH']=str(Path(sys.executable).resolve().parent)+os.pathsep+env.get('PATH','')
            log=(target/'worker.log').open('x');logs.append(log)
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(child)
            process['ranks'].append(dict(rank=rank,gpu=gpu,pid=child.pid,command=command));save()
        while any(c.poll() is None for c in children):
            elapsed=time.monotonic()-start
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Rank failed; terminating owned gang')
            if stop[0] or (out/'STOP').exists():raise InterruptedError('Owned gang STOP')
            if elapsed>a.seconds+120:raise TimeoutError('Gang wall budget')
            resource=subprocess.check_output(['nvidia-smi','-i',a.gpus,'--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True)
            with (out/'resources.jsonl').open('a') as stream:stream.write(json.dumps(dict(elapsed_s=elapsed,gpu=resource))+'\n')
            time.sleep(10)
        if any(c.returncode!=0 for c in children):raise RuntimeError('A rank exited unsuccessfully before final poll')
        process['status']='COMPLETED'
    except BaseException as error:
        process.update(status='FAILED',error=repr(error));raise
    finally:
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        for child in children:
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        process.update(elapsed_s=time.monotonic()-start,exit_codes=[c.returncode for c in children]);save()
        for log in logs:log.close()


if __name__=='__main__':main()
