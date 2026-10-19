"""Own and monitor both pinned-runtime peers for Inspire's Shadow PPO adapter."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[5]
MODEL='/home2/wyy/miniconda3/envs/graspenv/bin/python'
PHYSICS='/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--bc',type=Path);g.add_argument('--current-parent',type=Path)
    p.add_argument('--envs',type=int,default=256);p.add_argument('--motions',type=int,default=0);p.add_argument('--updates',type=int,default=4)
    p.add_argument('--seconds',type=int,default=1800);p.add_argument('--reset-probe',action='store_true');p.add_argument('--resume',type=Path)
    p.add_argument('--shared-run',type=Path,help='Explicit project-owned BC coordinator process.json allowed to share GPUs')
    p.add_argument('--evaluate',action='store_true');p.add_argument('--evaluation-controls',type=int,default=300)
    p.add_argument('--model-gpu',type=int,default=6);p.add_argument('--physics-gpu',type=int,default=7)
    p.add_argument('--wam-executor-mode',choices=['full_fp32','hand_fp32','hand_shared_tf32'],default='full_fp32')
    p.add_argument('--wam-batch',type=int,default=128);p.add_argument('--performance-migration-reason')
    p.add_argument('--state-transport',choices=['json','packed'],default='json')
    a=p.parse_args();out=a.output.resolve();a.cache=a.cache.resolve()
    if out.exists() or ROOT/'outputs/wam-rl-policy-v2' not in out.parents or not 2<=a.envs<=256 or not 0<a.seconds<=172800:
        raise ValueError('fresh bounded Task run required')
    allowed={}
    if a.shared_run:
        m=json.loads(a.shared_run.read_text())
        if m['status']!='RUNNING':raise ValueError('shared-run not active')
        for child in m['children']:
            cmd=Path('/proc')/str(child['pid'])/'cmdline'
            if cmd.exists() and 'inspire_full_bc.py' in cmd.read_text():allowed[child['pid']]=True
    if a.model_gpu==a.physics_gpu or not all(0<=g<8 for g in (a.model_gpu,a.physics_gpu)):raise ValueError('two distinct local GPUs required')
    for gpu in (a.model_gpu,a.physics_gpu):
        rows=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
        for row in rows.splitlines():
            pid,mem=[int(x.strip()) for x in row.split(',')];owner=subprocess.check_output(['ps','-o','user=','-p',str(pid)],text=True).strip()
            if not ((owner=='monica' and mem<=1024) or pid in allowed):raise RuntimeError('unexpected GPU occupant: '+row)
    if shutil.disk_usage(ROOT).free<24*2**30:raise RuntimeError('storage reserve')
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));out.mkdir(parents=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    frozen=out/'runtime-task';shutil.copytree(ROOT/'src/task/wam-rl-policy-v2',frozen,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    env=dict(os.environ,REF2DEX_RUN_ROOT=str(ROOT),REF2DEX_CODE_COMMIT=commit,PYTHONDONTWRITEBYTECODE='1')
    sockdir=ROOT/'tmp'/('isppo-'+str(os.getpid()));sockdir.mkdir(mode=0o700);sock=sockdir/'s'
    state=dict(status='RUNNING',pid=os.getpid(),git_commit=commit,run_id=out.name,envs=a.envs,gpus=[a.model_gpu,a.physics_gpu],
        max_seconds=a.seconds,max_bytes=3*1024**3,children=[],started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    state['inference']=dict(mode=a.wam_executor_mode,batch=a.wam_batch,state_transport=a.state_transport)
    children=[];handles=[];started=time.monotonic();last=-30
    def save():
        path=out/'process.json';part=path.with_suffix('.part');part.write_text(json.dumps(state,indent=2)+'\n');part.replace(path)
    def launch(args,name):
        stream=(out/(name+'.log')).open('x');handles.append(stream)
        proc=subprocess.Popen(args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(proc);state['children'].append(dict(pid=proc.pid,name=name,command=args));save();return proc
    def guard():
        nonlocal last
        elapsed=time.monotonic()-started
        if elapsed>a.seconds or shutil.disk_usage(out).free<20*2**30:raise RuntimeError('time/storage floor')
        if (out/'STOP').exists():raise InterruptedError('owned STOP requested')
        if elapsed-last>=30:
            last=elapsed;size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())
            if size>state['max_bytes']:raise RuntimeError('bounded checkpoints/artifacts exceeded')
            gpu=subprocess.check_output(['nvidia-smi','-i',str(a.model_gpu)+','+str(a.physics_gpu),'--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
            with (out/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(elapsed_s=elapsed,gpu=gpu,bytes=size))+'\n')
            print(json.dumps(dict(elapsed_s=elapsed,gpu=gpu)),flush=True)
    def interrupt(*_):raise InterruptedError('supervisor interrupted')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        args=[MODEL,str(frozen/'tools/run/inspire_shadow_ppo.py'),'--cache',str(a.cache),'--output',str(out/'policy'),
            '--socket',str(sock),'--gpu',str(a.model_gpu),'--updates',str(a.updates),'--seconds',str(a.seconds),
            '--bc' if a.bc else '--current-parent',str((a.bc or a.current_parent).resolve()),
            '--wam-executor-mode',a.wam_executor_mode,'--wam-batch',str(a.wam_batch),'--state-transport',a.state_transport]
        if a.performance_migration_reason:args.extend(['--performance-migration-reason',a.performance_migration_reason])
        if a.reset_probe:args.append('--reset-probe')
        if a.resume:args.extend(['--resume',str(a.resume.resolve())])
        if a.evaluate:args.extend(['--evaluate','--evaluation-controls',str(a.evaluation_controls)])
        server=launch(args,'policy');deadline=time.monotonic()+120
        while not sock.exists():
            guard()
            if server.poll() is not None or time.monotonic()>deadline:raise RuntimeError('policy server failed before listen')
            time.sleep(1)
        args=[PHYSICS,str(frozen/'tools/run/inspire_physics_host.py'),'--cache',str(a.cache),'--output',str(out/'physics'),
            '--socket',str(sock),'--server-pid',str(server.pid),'--envs',str(a.envs),'--gpu',str(a.physics_gpu),'--seconds',str(a.seconds),
            '--motions',str(a.motions),'--state-transport',a.state_transport]
        native=launch(args,'physics')
        while True:
            guard();codes=[p.poll() for p in children]
            if any(code not in (None,0) for code in codes):raise RuntimeError('peer failed '+str(codes))
            if all(code==0 for code in codes):break
            time.sleep(2)
        state['status']='COMPLETED'
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));raise
    finally:
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for stream in handles:stream.close()
        state.update(elapsed_s=time.monotonic()-started,finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'));save()
        sock.unlink(missing_ok=True)


if __name__=='__main__':main()
