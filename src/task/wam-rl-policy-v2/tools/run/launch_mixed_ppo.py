"""Bounded six-rank mixed-object PPO supervisor; one synchronized policy."""
import argparse,json,os,shutil,signal,socket,subprocess,sys,time
from pathlib import Path
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from wam_rl_policy_v2.mixed_objects import partition_objects,prepare_assets


def write(path,value):
    part=path.with_suffix('.partial');part.write_text(json.dumps(value,indent=2)+'\n');part.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True);p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3,4,5])
    p.add_argument('--objects-per-rank',type=int,default=16);p.add_argument('--envs',type=int,default=160)
    p.add_argument('--updates',type=int,default=10000);p.add_argument('--ticks',type=int,default=8)
    p.add_argument('--seconds',type=int,default=7200);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--save-every',type=int,default=200)
    p.add_argument('--port',type=int,default=0);p.add_argument('--resume',type=Path)
    p.add_argument('--checkpoint-migration-reason')
    p.add_argument('--execution-protocol',choices=['corrected','legacy'],default='corrected')
    p.add_argument('--goal-type',choices=['position_only','se3_pose'],default='position_only')
    p.add_argument('--termination-policy',choices=['wam_workspace','common_workspace'],default='wam_workspace')
    a=p.parse_args();a.output=a.output.resolve();a.policy=a.policy.resolve()
    if a.resume:a.resume=a.resume.resolve()
    if len(set(a.gpus))!=len(a.gpus) or not 1<=len(a.gpus)<=6:raise ValueError('One to six distinct GPUs')
    if not 1<=a.objects_per_rank<=a.envs<=256 or a.envs%a.objects_per_rank:raise ValueError('Equal object replicas required')
    if not 1<=a.updates<=10000 or not 1<=a.ticks<=180 or not 60<=a.seconds<=172800:raise ValueError('Bounded training required')
    if a.output.exists() or ROOT/'outputs/wam-rl-policy-v2' not in a.output.parents:raise ValueError('Fresh v2 output required')
    # Check before any worker can join another run's store; OS selects a port by default.
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as reservation:
        reservation.bind(('127.0.0.1',a.port))
        a.port=reservation.getsockname()[1]
    records=json.loads((ROOT/'outputs/shadow-finetune/policy-data-full-20261015-r1/manifest.json').read_text())['records']
    partitions=partition_objects(records,len(a.gpus),a.objects_per_rank,a.seed)
    a.output.mkdir(parents=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    frozen=a.output/'runtime-code'
    shutil.copytree(TASK,frozen,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
    status=dict(status='RUNNING',pid=os.getpid(),git_commit=commit,work_version=commit,run_id=a.output.name,
        gpus=a.gpus,world_size=len(a.gpus),rendezvous_port=a.port,batch_pipeline=os.environ.get('REF2DEX_BATCH_PIPELINE','legacy'),objects=len(a.gpus)*a.objects_per_rank,
        environments=len(a.gpus)*a.envs,updates=a.updates,ticks=a.ticks,seconds=a.seconds,
        policy=str(a.policy),resume=str(a.resume),python=sys.executable,
        execution_protocol=a.execution_protocol,goal_type=a.goal_type,termination_policy=a.termination_policy,
        checkpoint_migration_reason=a.checkpoint_migration_reason,
        training='pinned official PPO run/update/storage; synchronized gradients/KL/advantage statistics; frozen WAM; TRAIN only')
    write(a.output/'process.json',status)
    children=[];streams=[];started=time.monotonic();stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    try:
        for rank,(gpu,rows) in enumerate(zip(a.gpus,partitions)):
            manifest=a.output/('objects-rank-%d.json'%rank)
            assets=a.output/('assets-rank-%d'%rank);size=prepare_assets(rows,assets)
            write(manifest,dict(objects=rows,asset_root=str(assets),linked_assets_bytes=size))
            cmd=[sys.executable,'-u',str(frozen/'tools/run/closed_loop_scheduled_shadow.py'),
                '--output',str(a.output/('rank-%d'%rank)),'--policy',str(a.policy),
                '--object-manifest',str(manifest),'--gpu',str(gpu),'--rank',str(rank),
                '--world-size',str(len(a.gpus)),'--rendezvous','tcp://127.0.0.1:'+str(a.port),
                '--envs',str(a.envs),'--ppo-updates',str(a.updates),'--ticks',str(a.ticks),
                '--seconds',str(a.seconds),'--seed',str(a.seed),'--official-ppo','--save-every',str(a.save_every)]
            if a.resume:cmd+=['--rl-checkpoint',str(a.resume)]
            cmd+=['--execution-protocol',a.execution_protocol,'--goal-type',a.goal_type,'--termination-policy',a.termination_policy]
            if a.checkpoint_migration_reason:cmd+=['--checkpoint-migration-reason',a.checkpoint_migration_reason]
            env=dict(os.environ,REF2DEX_RUN_ROOT=str(ROOT),REF2DEX_CODE_COMMIT=commit,
                OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
                TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/shadow-isaac-extensions'),MAX_JOBS='2')
            stream=(a.output/('rank-%d.log'%rank)).open('x');streams.append(stream)
            children.append(subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
        status['pids']=[c.pid for c in children];write(a.output/'process.json',status)
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('A rank failed; stopping owned worker groups')
            if stopped[0] or (a.output/'STOP').exists():raise InterruptedError('STOP requested')
            if time.monotonic()-started>a.seconds+30:raise TimeoutError('Training wall budget')
            if shutil.disk_usage(a.output).free<20*2**30:raise RuntimeError('20GiB disk reserve')
            resource=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True)
            with (a.output/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(elapsed_s=time.monotonic()-started,gpus=resource.strip().splitlines()))+'\n')
            time.sleep(15)
        if any(c.returncode for c in children):raise RuntimeError('Rank exit failure')
        status['status']='COMPLETED'
    except BaseException as error:
        status.update(status='FAILED',error=repr(error));raise
    finally:
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for stream in streams:stream.close()
        status.update(elapsed_s=time.monotonic()-started,exit_codes=[c.returncode for c in children])
        write(a.output/'process.json',status)

if __name__=='__main__':main()
