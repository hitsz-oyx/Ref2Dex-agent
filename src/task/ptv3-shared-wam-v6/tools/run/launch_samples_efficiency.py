"""Bounded matched A/B/C queue. Run only on GPUs explicitly assigned by owner."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[5]
TASK = ROOT / 'src/task/ptv3-shared-wam-v6'
MODEL = '/home2/wyy/miniconda3/envs/graspenv/bin/python'
PHYSICS = '/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python'
CACHE = ROOT / 'outputs/wam-rl-policy-v2/inspire-shadow-parity-20261019-r1/cache'
ADMISSION = ROOT / 'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json'
BC = ROOT / 'outputs/wam-rl-policy-v3/inspire-scheduled686-bc-20261020-r4/best.pt'


def write(path, value):
    part = path.with_suffix(path.suffix + '.partial')
    part.write_text(json.dumps(value, indent=2) + '\n')
    part.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpus', nargs='+', type=int, default=[2, 3])
    p.add_argument('--seeds', nargs='+', type=int, default=[191, 192, 193])
    p.add_argument('--arms', nargs='+', choices=['A', 'B', 'C'], default=['A', 'B', 'C'])
    p.add_argument('--updates', type=int, default=512)
    p.add_argument('--envs', type=int, default=256)
    p.add_argument('--seconds-per-run', type=int, default=14400)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-ddp-20261021-r1/training/step004371.pt')
    p.add_argument('--stats', type=Path, default=ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/stats.json')
    p.add_argument('--geometry-cache',type=Path,required=True,help='Certified dense2048 cache for v6; native initialization retains original cache')
    p.add_argument('--objects', nargs='+', default=['cubemedium', 'cylindermedium', 'mug'])
    p.add_argument('--eval-seed', type=int, default=194)
    p.add_argument('--engineering-smoke',action='store_true',help='Force asynchronous reset fixture and skip learning-curve evaluation; never scientific evidence')
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists() or ROOT/'outputs/ptv3-shared-wam-v6' not in out.parents:
        raise ValueError('fresh task-local output required')
    if not a.gpus or len(set(a.gpus)) != len(a.gpus) or set(a.gpus)-{0, 1, 2, 3}:
        raise ValueError('only owner-authorized GPUs 0–3; defaults 2/3')
    if not 1 <= a.updates <= 512 or not 2 <= a.envs <= 512 or not 30 <= a.seconds_per_run <= 14400:
        raise ValueError('bounded updates/envs required')
    for gpu in a.gpus:
        text = subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        if int(text.strip().split(',')[1]) > 64:
            raise RuntimeError('GPU no longer idle: '+text)
    out.mkdir(parents=True)
    private = ROOT/'tmp'/out.name
    private.mkdir(parents=True,exist_ok=True)
    objects = out/'objects.json'
    write(objects, dict(object_names=a.objects))
    manifest = dict(status='RUNNING',experiment_id='P-20261022-wam-v6-samples-efficiency-3arm',run_id=out.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        gpus=a.gpus,seeds=a.seeds,arms=a.arms,objects=a.objects,updates=a.updates,envs=a.envs,
        controls_per_seed_arm=a.updates*a.envs*8,seconds_per_run=a.seconds_per_run,
        eval_seed=a.eval_seed,bc=str(BC),checkpoint=str(a.checkpoint),stats=str(a.stats),geometry_cache=str(a.geometry_cache.resolve()),jobs=[])
    write(out/'manifest.json',manifest)
    environment=dict(os.environ,REF2DEX_RUN_ROOT=str(ROOT),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=str(private))

    def pair(folder, arm, seed, gpu, *, resume=None, evaluation=False):
        folder.mkdir()
        sock=private/(folder.name+'.sock')
        policy=[MODEL,'-u',str(TASK/'tools/run/train_samples_efficiency.py'),'--bc',str(BC),
            '--audit-manifest',str(ADMISSION),'--geometry-cache',str(a.geometry_cache.resolve()),'--output',str(folder/'policy'),
            '--socket',str(sock),'--gpu',str(gpu),'--seed',str(seed),'--updates',str(a.updates),
            '--seconds',str(a.seconds_per_run),'--arm',arm,'--checkpoint',str(a.checkpoint),'--stats',str(a.stats)]
        if resume:policy+=['--resume',str(resume)]
        if evaluation:policy+=['--evaluate-only']
        if a.engineering_smoke:policy+=['--reset-probe']
        row=dict(folder=str(folder),arm=arm,seed=seed,gpu=gpu,evaluation=evaluation,policy_command=policy,status='STARTING')
        write(folder/'process.json',row)
        children=[]
        streams=[]
        started=time.monotonic()
        try:
            stream=(folder/'policy.log').open('x');streams.append(stream)
            server=subprocess.Popen(policy,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);children.append(server)
            while not sock.exists():
                if server.poll() is not None or time.monotonic()-started>300:raise RuntimeError('policy did not listen')
                time.sleep(1)
            native=[PHYSICS,'-u',str(TASK/'tools/run/samples_efficiency_physics.py'),'--cache',str(CACHE),
                '--output',str(folder/'native'),'--socket',str(sock),'--server-pid',str(server.pid),
                '--envs',str(a.envs),'--gpu',str(gpu),'--seed',str(seed),'--seconds',str(a.seconds_per_run),
                '--object-manifest',str(objects),'--admission-manifest',str(ADMISSION),
                '--state-transport','packed','--extended-state','--synchronize-setters']
            stream=(folder/'native.log').open('x');streams.append(stream)
            children.append(subprocess.Popen(native,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
            row.update(status='RUNNING',pids=[c.pid for c in children],native_command=native);write(folder/'process.json',row)
            last_resource=-30
            while any(c.poll() is None for c in children):
                if any(c.poll() not in (None,0) for c in children):raise RuntimeError('owned peer failed')
                if time.monotonic()-started>a.seconds_per_run+120:raise TimeoutError('run exceeded bounded budget')
                if (out/'STOP').exists():(folder/'STOP').touch()
                elapsed=time.monotonic()-started
                if elapsed-last_resource>=30:
                    last_resource=elapsed
                    resources=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
                    progress=folder/'policy/progress.json'
                    status=json.loads(progress.read_text()) if progress.exists() else {}
                    with (folder/'resources.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(elapsed_s=elapsed,gpu=resources,iteration=status.get('iteration'),eta_s=status.get('eta_s')))+'\n')
                time.sleep(2)
            if any(c.returncode for c in children):raise RuntimeError('nonzero owned peer exit')
            row['status']='COMPLETED'
        except BaseException as error:
            row.update(status='FAILED',error=repr(error));raise
        finally:
            for child in children:
                if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            for child in children:
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            for stream in streams:stream.close()
            row.update(elapsed_s=time.monotonic()-started,exit_codes=[c.returncode for c in children]);write(folder/'process.json',row)
        return folder/'policy'

    def lane(gpu, jobs):
        results=[]
        for seed,arm in jobs:
            if (out/'STOP').exists():break
            folder=out/('%s-seed%d'%(arm,seed))
            policy=pair(folder,arm,seed,gpu)
            evaluations=[]
            for iteration in ([] if a.engineering_smoke else [0]+list(range(128,a.updates+1,128))):
                checkpoint=policy/('official-iteration-%d.pt'%iteration)
                if not checkpoint.exists():raise RuntimeError('missing fixed evaluation checkpoint '+str(checkpoint))
                evaluation=out/('%s-seed%d-eval%d'%(arm,seed,iteration))
                pair(evaluation,arm,a.eval_seed,gpu,resume=checkpoint,evaluation=True)
                evaluations.append(str(evaluation/'policy/evaluation.json'))
            results.append(dict(arm=arm,seed=seed,policy=str(policy),evaluations=evaluations))
        return results

    # Rotate arm order across seeds to limit confounding by machine time/order.
    jobs=[]
    for index,seed in enumerate(a.seeds):
        order=a.arms[index%len(a.arms):]+a.arms[:index%len(a.arms)]
        jobs.extend((seed,arm) for arm in order)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
            futures=[pool.submit(lane,gpu,jobs[index::len(a.gpus)]) for index,gpu in enumerate(a.gpus)]
            for future in futures:manifest['jobs'].extend(future.result())
        manifest['status']='STOPPED' if (out/'STOP').exists() else 'COMPLETED'
    except BaseException as error:
        (out/'STOP').touch();manifest.update(status='FAILED',error=repr(error));raise
    finally:write(out/'manifest.json',manifest)


if __name__=='__main__':main()
