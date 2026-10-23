"""One-shot stage controller: data gates -> smoke -> half -> expansion.

Faults are terminal. There is no restart path; a later manually authorized run
must use a new output and explicitly preserve the original evidence.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'tools/run')]
from remote_staged_data import read,write,sha,GIB,NAS,paths_environment


def gpu_snapshot():
    return dict(at=time.time(),gpus=subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True),
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
            '--format=csv,noheader,nounits'],text=True))


def environment(out):
    paths_environment(out)
    return dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',REF2DEX_RUN_ROOT=str(ROOT/'tmp/rh'),
        TORCH_EXTENSIONS_DIR=str(out/'cache/extensions'),TRITON_CACHE_DIR=str(out/'cache/triton'),
        CUMM_CACHE_PATH=str(out/'cache/cumm'),
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
        NCCL_ASYNC_ERROR_HANDLING='1')


def launch(out,label,command):
    path=out/(label+'-launch.json')
    if path.exists():raise FileExistsError('one-shot launch already exists: '+label)
    with (out/(label+'.log')).open('x') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=environment(out),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(path,dict(pid=process.pid,pgid=process.pid,command=command,at=time.time(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),automatic_restart=False),immutable=True)
    return process


def stop_owned(process):
    if process.poll() is not None:return
    if os.getpgid(process.pid)!=process.pid:raise RuntimeError('owned process group changed')
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()


def wait_process(out,label,process,seconds,budget=136*GIB):
    start=time.monotonic();last_progress=time.time();last_step=None
    while process.poll() is None:
        snapshot=gpu_snapshot();snapshot.update(label=label,pid=process.pid,free_bytes=shutil.disk_usage(out).free)
        directory=out/label
        progress=directory/'progress.json'
        if progress.exists():
            value=read(progress);snapshot['progress']=value
            if value.get('step')!=last_step:last_step=value.get('step');last_progress=time.time()
        checkpoints=sum(p.stat().st_size for name in ('half-training','full-training') for p in (out/name).glob('step*.pt*'))
        snapshot['checkpoint_bytes']=checkpoints
        write(out/'watchdog-latest.json',snapshot)
        with (out/'watchdog.jsonl').open('a') as stream:stream.write(json.dumps(snapshot)+'\n')
        if snapshot['free_bytes']<30*GIB or checkpoints>budget or time.monotonic()-start>seconds:
            stop_owned(process);raise RuntimeError('disk/artifact/time fault: '+label)
        if 'training' in label and time.time()-last_progress>1800:
            stop_owned(process);raise RuntimeError('training stalled for1800s: '+label)
        if list((out/'remaining').glob('failure-*.json')):
            stop_owned(process);raise RuntimeError('parallel remaining conversion failed')
        if (out/'STOP').exists():
            stop_owned(process);raise InterruptedError('owner STOP')
        time.sleep(30)
    if process.returncode:raise RuntimeError(label+' failed with exit '+str(process.returncode))


def recipe(out,stage):
    panel=read(out/'fixed-panel/manifest.json');checks=out/(stage+'-gates/data-checks.json')
    config=dict(panel['config'])
    for key in ('local_subset_scope','curated_manifest','condition_recipe','condition_recipe_sha256','hard_stop_epoch'):
        config.pop(key,None)
    config.update(execution_campaign='v6.remote-staged-20261023',batch_size=96,updates=40000,
        max_seconds=48*3600,artifact_budget_bytes=136*GIB,checkpoint_interval=500,
        validation_interval=500,save_initial_weights=True)
    folder=out/(stage+'-recipe');folder.mkdir(exist_ok=False)
    write(folder/'config.json',config,immutable=True);write(folder/'stats.json',panel['stats'],immutable=True)
    path=folder/'recipe.json'
    r=dict(schema='ref2dex.remote-staged-training.v1',stage=stage,recipe_path=str(path),
        manifest=str(out/(stage+'-census/manifest.json')),stats=str(folder/'stats.json'),
        panel_manifest=str(out/'fixed-panel/manifest.json'),config=str(folder/'config.json'),
        geometry_manifest=str(NAS/'ptv3-shared-wam-v5-20261019-r1/data/manifest.json'),
        data_checks=str(checks),admission=str(folder/'admission.json'),
        initialization=str(out/'local3000-weights.pt'),initialization_sha256=sha(out/'local3000-weights.pt'),
        local3000_panel=str(out/'local3000-panel.npz'),local3000_panel_sha256=sha(out/'local3000-panel.npz'),
        source_probabilities={str(s):1/9 for s in (0,1,2,4,5,6,8,9,10)},
        sampling='uniform source -> uniform mother -> uniform observed known phase -> uniform window; cap8 per mother per rank batch',
        world_size=4,local_batch=96,global_batch=384,loader_workers_per_rank=2,
        last_step=3000 if stage=='half' else 40000,max_seconds=(6 if stage=='half' else 42)*3600,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        claim_scope='nine-source TRAIN / eight-source DEV; fixed TRAIN panel only; no source8 DEV or execution claim',
        preregistered=dict(primary_metrics=panel['metrics_order'],
            primary='all three EPE lower than same-runtime initialization and all paired cluster CI upper endpoints<0;500-update mean loss blocks monotonically nonincreasing',
            secondary='paired differences vs frozen local3000 predictions,4000 mother-cluster bootstraps seed170',
            fault=['OOM','nonfinite loss/gradient','loader failure','disk<30GiB','checkpoint>136GiB','stall1800s'],
            automatic_restart=False),
        transition='half and full have separate data/recipe/checkpoint/panel identities; source probability1/9 stays fixed, within-source mother probabilities change when remaining mothers are added')
    for field in ('manifest','stats','panel_manifest','config'):r[field+'_sha256']=sha(r[field])
    write(path,r,immutable=True);return path


def train_command(path,out,smoke=False,expand=None):
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4',
        str(TASK/'tools/run/train_staged_pretrain.py'),'--recipe',str(path),'--output',str(out)]
    if smoke:command+=['--smoke']
    if expand:command+=['--expand-from',str(expand)]
    return command


def admit(out,stage,path):
    scope=read(out/'scope-authorization.json')
    if scope.get('allow_nine_train_eight_dev') is not True:raise RuntimeError('source8 DEV scope not authorized')
    smoke=read(out/(stage+'-smoke/result.json'))
    if smoke.get('status')!='SMOKE_PASS' or smoke['updates']!=1:raise RuntimeError('E2E smoke failed')
    checks=read(out/(stage+'-gates/data-checks.json'))
    write(path.parent/'admission.json',dict(status='EIGHT_GATES_PASS_RESTRICTED_SCOPE',training_allowed=True,
        recipe_sha256=sha(path),data_checks_sha256=sha(out/(stage+'-gates/data-checks.json')),
        smoke=smoke,scope_authorization=scope,coverage_limit=read(path)['claim_scope'],
        gates=dict(checks['gates'],e2e='PASS',coverage='PASS_EXPLICIT_RESTRICTED_SCOPE')),immutable=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--adopt-idle-controller',action='store_true');a=p.parse_args();out=a.output.resolve()
    assert NAS in out.parents
    lock=out/'CAMPAIGN_ONE_SHOT.lock'
    if a.adopt_idle_controller:
        original_pid=int(lock.read_text())
        if (Path('/proc')/str(original_pid)).exists():raise RuntimeError('original controller still alive')
        status=read(out/'campaign-status.json')['status']
        if status not in ('WAITING_HALF_CONVERSION','WAITING_REFERENCES') or (out/'half-data-gates-launch.json').exists():
            raise RuntimeError('only an idle pre-training controller can be adopted')
        lock=out/'CAMPAIGN_IDLE_ADOPTION.lock'
    fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o664)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    active=[];started=time.monotonic()
    try:
        while True:
            progress=read(out/'half/progress.json') if (out/'half/progress.json').exists() else {}
            if list((out/'half').glob('failure-*.json')):raise RuntimeError('half conversion failed')
            if progress.get('status')=='CONVERSION_COMPLETED':break
            if time.monotonic()-started>6*3600:raise TimeoutError('half conversion deadline')
            write(out/'campaign-status.json',dict(status='WAITING_HALF_CONVERSION',progress=progress,automatic_restart=False))
            if (out/'STOP').exists():raise InterruptedError('owner STOP')
            time.sleep(30)
        for stage in ('half','full'):
            if stage=='full':
                wait_process(out,'remaining-conversion',remaining,6*3600)
            gate=launch(out,stage+'-data-gates',[sys.executable,'-u',str(TASK/'tools/audit/remote_staged_gates.py'),'--output',str(out),'--stage',stage])
            active.append(gate);wait_process(out,stage+'-data-gates',gate,3600)
            while not all((out/f).exists() for f in ('fixed-panel/manifest.json','local3000-weights.pt','local3000-panel.npz')):
                if time.monotonic()-started>6*3600:raise TimeoutError('reference transfer deadline')
                write(out/'campaign-status.json',dict(status='WAITING_REFERENCES',automatic_restart=False));time.sleep(30)
            path=recipe(out,stage)
            snapshot=gpu_snapshot();write(out/(stage+'-gpu-preflight.json'),snapshot,immutable=True)
            if snapshot['processes'].strip():raise RuntimeError('other GPU processes present at training preflight')
            smoke=launch(out,stage+'-smoke',train_command(path,out/(stage+'-smoke'),smoke=True))
            active.append(smoke);wait_process(out,stage+'-smoke',smoke,1800)
            # Finish every technical check before waiting for a coverage-scope
            # answer. Elapsed time never constitutes approval for the scope.
            while not (out/'scope-authorization.json').exists():
                write(out/'campaign-status.json',dict(status='WAITING_SOURCE8_SCOPE_AUTHORIZATION',data_checks=str(out/(stage+'-gates/data-checks.json')),smoke=str(out/(stage+'-smoke/result.json')),automatic_restart=False))
                if (out/'STOP').exists():raise InterruptedError('owner STOP')
                time.sleep(30)
            admit(out,stage,path)
            expand=None if stage=='half' else max((out/'half-training').glob('step*.pt'),key=lambda p:int(p.stem[4:]))
            training=launch(out,stage+'-training',train_command(path,out/(stage+'-training'),expand=expand))
            active.append(training)
            write(out/'campaign-status.json',dict(status='TRAINING_'+stage.upper(),pid=training.pid,recipe=str(path),automatic_restart=False))
            if stage=='half':
                while not (out/'half-training/progress.json').exists():
                    if training.poll() is not None:raise RuntimeError('half training failed before first delivered update')
                    if time.monotonic()-started>8*3600:raise TimeoutError('first update deadline')
                    time.sleep(15)
                # Only after actual half training starts, run the complement.
                remaining=launch(out,'remaining-conversion',[sys.executable,'-u',str(TASK/'tools/run/remote_staged_data.py'),'--output',str(out),'--mode','remaining'])
                active.append(remaining)
            wait_process(out,stage+'-training',training,((6 if stage=='half' else 42)*3600)+1800)
            result=read(out/(stage+'-training/result.json'))
            if result['status']!='COMPLETED' or result['updates']<=0:raise RuntimeError('training stage did not complete')
            write(out/(stage+'-stage-completed.json'),dict(result=result,recipe_sha256=sha(path),
                final_checkpoints=[str(f) for f in (out/(stage+'-training')).glob('step*.pt')],
                panels=[str(f) for f in (out/(stage+'-training')).glob('panel-step*/summary.json')]),immutable=True)
        write(out/'campaign-status.json',dict(status='COMPLETED',automatic_restart=False))
    except BaseException as exc:
        for process in active:stop_owned(process)
        write(out/'campaign-failure.json',dict(reason=repr(exc),automatic_restart=False,at=time.time()),immutable=True)
        raise


if __name__=='__main__':main()
