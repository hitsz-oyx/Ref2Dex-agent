#!/usr/bin/env python3
"""Fixed HF21 randomized execution and full per-phase audit within one slot."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission,PYTHON


def run(args):
    begin=time.monotonic()
    base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    output=args.output.resolve()
    if output.parent!=base or output.exists() or args.output.is_symlink():
        raise ValueError('unique owned source output required')
    devices=args.gpus
    if len(devices)!=3 or len(set(devices))!=3:
        raise ValueError('fixed three distinct idle GPUs required')
    engineering_path=ROOT/'docs/experiments/probes/P-20261002-structured-opportunity-engineering-completion-r1.json'
    engineering=json.loads(engineering_path.read_text())
    if not engineering['engineering_passed'] or engineering['run_status']!='COMPLETED':
        raise ValueError('complete native and full planner engineering required')
    if any(sha(Path(k))!=v for k,v in engineering['artifact_sha256'].items()):
        raise ValueError('engineering evidence drift')
    old_path=Path(next(k for k in engineering['artifact_sha256'] if k.endswith('opportunity-engineering-r1/run_manifest.json')))
    old=json.loads(old_path.read_text())
    if any(sha(Path(k))!=v for k,v in old['input_sha256'].items()):
        raise ValueError('native generator input drift')
    hashes=dict(old['input_sha256'])
    dependencies=['contact_geometry_consequence.py','native_pd_selector.py','native_pd_consequence.py',
                  'native_pd_policy_controls.py','optimized_contact_actions.py', 'structured_contact_consequence.py',
                  'structured_contact_actions.py','structured_contact_opportunity.py']
    paths=[Path(__file__),engineering_path,
           ROOT/'docs/experiments/probes/P-20261002-structured-contact-opportunity.md',
           ROOT/'scripts/analyze_structured_contact_opportunity.py',
           ROOT/'scripts/audit_structured_opportunity_statistics.py',
           ROOT/'scripts/structured_opportunity_statistics.py']
    paths += [ROOT/'src/task/CmResidual'/name for name in dependencies]
    for path in paths:
        hashes[str(path.resolve())]=sha(path)
    # Admit every device before launching any native process.
    admissions=[gpu_admission(gpu) for gpu in devices]
    output.mkdir(exist_ok=False)
    phases=[dict(seed=seed,gpu_index=devices[(seed-611)%3],run_status='PENDING',
                 directory=str(output/f'seed{seed}')) for seed in range(611,619)]
    manifest=dict(experiment_id='P-20261002-structured-contact-opportunity',family='HF21',
        probe_index_in_family=2,run_status='RUNNING',smoke_only=False,pid=os.getpid(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        input_sha256=hashes,phases=phases,admissions=admissions,
        wall_limit_seconds=3600,source_pipeline_limit_seconds=3200,
        prior_engineering_seconds=400.,output_limit_bytes=8<<30,
        max_owned_gpu_concurrency=3,model_training=False,expert_training=False,
        scope='randomized H10 generated candidates and full audits; utility analysis separate')
    lock=threading.RLock();stop=threading.Event();processes={}
    def save():
        with lock:
            manifest['elapsed_seconds']=time.monotonic()-begin
            manifest['cumulative_seconds']=manifest['elapsed_seconds']+400.
            temporary=output/'run_manifest.json.tmp'
            temporary.write_text(json.dumps(manifest,indent=2)+'\n')
            temporary.replace(output/'run_manifest.json')
    def check():
        if stop.is_set():
            raise RuntimeError('another owned phase failed; no new work')
        if time.monotonic()-begin>3200 or time.monotonic()-begin+400.>3600:
            raise TimeoutError('whole fixed slot wall budget')
        size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if size+engineering['total_output_bytes']>8<<30:
            raise ValueError('whole fixed slot output budget')
        if any(sha(Path(k))!=v for k,v in hashes.items()):
            raise ValueError('fixed source input drift')
    def terminate(process):
        if process.poll() is None:
            try:
                os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:
                process.wait(timeout=5)
                return
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait(timeout=5)
    def execute(command,env,log_path,phase,kind,limit):
        check();started=time.monotonic()
        with log_path.open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT/'third_party/DExplore',env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:
                processes[process.pid]=process
                phase[kind+'_pid']=process.pid
                phase[kind+'_command']=command
                save()
            try:
                while process.poll() is None:
                    if time.monotonic()-started>limit:
                        raise TimeoutError(kind+' phase timeout')
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        check()
                with lock:
                    phase[kind+'_exit_code']=process.returncode
                if process.returncode:
                    raise RuntimeError(kind+' child exit '+str(process.returncode))
            except BaseException:
                terminate(process)
                with lock:
                    phase[kind+'_exit_code']=process.returncode
                raise
            finally:
                with lock:
                    phase[kind+'_elapsed_seconds']=time.monotonic()-started
                    processes.pop(process.pid,None)
                save()
    def lane(admission):
        gpu=admission['index']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',PYTHONDONTWRITEBYTECODE='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',
            TORCH_EXTENSIONS_DIR=str(output/f'cache/gpu{gpu}/torch_extensions'),
            XDG_CACHE_HOME=str(output/f'cache/gpu{gpu}'))
        env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        for phase in [p for p in phases if p['gpu_index']==gpu]:
            try:
                check()
                current=gpu_admission(gpu)
                if current['uuid']!=admission['uuid']:
                    raise ValueError('GPU identity drift')
                seed=phase['seed'];directory=Path(phase['directory'])
                command=list(old['phases'][0]['command'])
                changes={'--output-dir':str(directory),'--panel-seed':str(seed),
                    '--assignment-seed':str(seed+15000),'--seed':str(seed),
                    '--windows-per-stratum':'2','--max-steps':'300','--wall-seconds':'600',
                    '--output':str(directory/'unused.json'),'--output_path':str(directory/'player')}
                for key,value in changes.items():
                    command[command.index(key)+1]=value
                env['PYTHONHASHSEED']=str(seed)
                with lock:
                    phase.update(run_status='RUNNING',admission=current)
                save()
                execute(command,env,output/f'seed{seed}.native.log',phase,'native',640)
                result=json.loads((directory/'results.json').read_text())
                if result['run_status']!='COMPLETED' or sha(directory/'records.pt')!=result['record_sha256']:
                    raise ValueError('terminal native records')
                if sha(directory/'planning.pt')!=result['planning_sha256']:
                    raise ValueError('terminal native planning')
                with lock:
                    phase.update(result=result,exit_code=0)
                # A completed-phase manifest, explicitly distinct from the still-active campaign.
                control=output/'phase-controls'/f'seed{seed}'
                control.mkdir(parents=True,exist_ok=False)
                with lock:
                    single=dict(phase,run_status='COMPLETED')
                phase_manifest=dict(run_status='COMPLETED',child_exit_code=0,input_sha256=hashes,
                    phases=[single],source_campaign=str(output),
                    scope='this native phase is terminal; parent campaign may still be RUNNING')
                (control/'run_manifest.json').write_text(json.dumps(phase_manifest,indent=2)+'\n')
                audit_path=output/f'seed{seed}.audit.json'
                audit_command=[PYTHON,'-u',str(ROOT/'scripts/audit_structured_contact_opportunity.py'),
                    '--run',str(control),'--output',str(audit_path),'--gpu',str(gpu)]
                execute(audit_command,env,output/f'seed{seed}.audit.log',phase,'audit',450)
                audit=json.loads(audit_path.read_text())
                if audit['run_status']!='COMPLETED' or len(audit['phases'])!=1:
                    raise ValueError('terminal independent full audit')
                with lock:
                    phase.update(run_status='COMPLETED',audit=str(audit_path),audit_sha256=sha(audit_path))
                check();save()
                print(json.dumps(dict(seed=seed,gpu=gpu,rows=result['rows'],
                    source_seconds=phase['native_elapsed_seconds'],audit_seconds=phase['audit_elapsed_seconds'],
                    complete_phases=sum(p['run_status']=='COMPLETED' for p in phases),
                    cumulative_seconds=time.monotonic()-begin+400.)),flush=True)
            except BaseException as exc:
                with lock:
                    phase.update(run_status='FAILED',error=repr(exc))
                stop.set();save()
                raise
    save()
    pool=ThreadPoolExecutor(max_workers=3)
    try:
        futures=[pool.submit(lane,admission) for admission in admissions]
        for future in as_completed(futures):
            future.result()
        if not all(p['run_status']=='COMPLETED' for p in phases):
            raise ValueError('incomplete fixed panel')
        check()
        manifest.update(run_status='COMPLETED',child_exit_code=0,input_hashes_unchanged=True,
            cumulative_source_gpu_seconds=sum(p['native_elapsed_seconds'] for p in phases),
            cumulative_audit_gpu_seconds=sum(p['audit_elapsed_seconds'] for p in phases))
    except BaseException as exc:
        stop.set()
        with lock:
            active=list(processes.values())
        for process in active:
            terminate(process)
        with lock:
            manifest.update(run_status='FAILED',error=repr(exc))
        raise
    finally:
        pool.shutdown(wait=True)
        save()
    print(json.dumps(dict(run_status='COMPLETED',phases=len(phases),
        cumulative_seconds=manifest['cumulative_seconds'],
        source_gpu_seconds=manifest['cumulative_source_gpu_seconds'],
        audit_gpu_seconds=manifest['cumulative_audit_gpu_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs=3,default=[4,5,6])
    run(p.parse_args())
