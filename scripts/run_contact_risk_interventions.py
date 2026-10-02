#!/usr/bin/env python3
"""Fixed conditional-risk execution panel and independent full GPU audits."""
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
    if len(devices)!=(1 if args.engineering else 3) or len(set(devices))!=len(devices):
        raise ValueError('one engineering or three scientific idle GPUs')
    static_path=ROOT/'docs/experiments/probes/P-20261002-contact-risk-guard-engineering-r2.json'
    static=json.loads(static_path.read_text())
    if not static['engineering_passed'] or not static['practical_binding_coverage']:
        raise ValueError('fixed contact-specific action binding required')
    template_path=ROOT/'src/task/CmResidual/research/contact_consequence/output/P-20261002-structured-opportunity-engineering-r1/run_manifest.json'
    template=json.loads(template_path.read_text())
    # Inherited environment, asset and six self-trained expert identities remain read-only.
    hashes=dict(template['input_sha256'])
    prior_engineering=0.;engineering_bytes=0
    if not args.engineering:
        if args.engineering_run is None:raise ValueError('accepted new native engineering required')
        engineering_path=args.engineering_run.resolve()/'run_manifest.json'
        engineering=json.loads(engineering_path.read_text())
        if engineering['run_status']!='COMPLETED' or not engineering['smoke_only'] or engineering['child_exit_code']!=0:
            raise ValueError('terminal native risk engineering')
        if any(sha(Path(k))!=v for k,v in engineering['input_sha256'].items()):raise ValueError('engineering input drift')
        phase=engineering['phases'][0]
        if phase['seed']!=649 or phase['native_exit_code']!=0 or phase['audit_exit_code']!=0:
            raise ValueError('excluded fixed native engineering')
        import torch
        torch.set_num_threads(2)
        record=torch.load(Path(phase['directory'])/'records.pt',map_location='cpu',weights_only=False)
        from analyze_contact_risk_interventions import arrays
        data=arrays(record)
        if not ((data['assignment']==2)&data['actual_pd_difference']).any():
            raise ValueError('native engineering lacks actual Cm-vs-direct PD action effect; stop without scientific seeds')
        prior_engineering=engineering['elapsed_seconds']
        engineering_bytes=sum(p.stat().st_size for p in args.engineering_run.rglob('*') if p.is_file())
        hashes[str(engineering_path)]=sha(engineering_path)
        for path in (Path(phase['audit']),Path(phase['directory'])/'records.pt',Path(phase['directory'])/'planning.pt'):
            hashes[str(path.resolve())]=sha(path)
    dependencies=['contact_geometry_consequence.py','native_pd_selector.py','native_pd_consequence.py',
        'native_pd_policy_controls.py','optimized_contact_actions.py','structured_contact_consequence.py',
        'structured_contact_actions.py','support_preserving_consequence.py','contact_risk_guard.py','contact_risk_interventions.py']
    checkpoint=ROOT/'src/task/CmResidual/research/contact_consequence/output/P-20261002-support-preserving-contact-fit-r2/support_preserving_contact_consequence.pt'
    if sha(checkpoint)!=static['checkpoint_sha256']:raise ValueError('frozen adapted model')
    paths=[Path(__file__),static_path,checkpoint,template_path,
        ROOT/'docs/experiments/probes/P-20261002-contact-risk-interventions.md',
        ROOT/'docs/decisions/D-20261002-contact-risk-interventions.md',
        ROOT/'scripts/collect_contact_risk_interventions.py',ROOT/'scripts/audit_contact_risk_interventions.py',
        ROOT/'scripts/analyze_contact_risk_interventions.py',ROOT/'scripts/contact_risk_intervention_statistics.py',
        ROOT/'scripts/check_contact_risk_intervention_statistics.py',ROOT/'scripts/audit_contact_risk_intervention_statistics.py',
        ROOT/'scripts/analyze_structured_contact_opportunity.py',ROOT/'scripts/analyze_orientation_feedback_opportunity.py']
    paths += [ROOT/'src/task/CmResidual'/name for name in dependencies]
    for path in paths:hashes[str(path.resolve())]=sha(path)
    if any(sha(Path(k))!=v for k,v in hashes.items()):raise ValueError('inherited identity drift')
    # Admit every device before launching any native process.
    admissions=[gpu_admission(gpu) for gpu in devices]
    output.mkdir(exist_ok=False)
    phases=[dict(seed=seed,gpu_index=devices[(seed-(649 if args.engineering else 651))%len(devices)],run_status='PENDING',
                 directory=str(output/f'seed{seed}')) for seed in ([649] if args.engineering else range(651,659))]
    manifest=dict(experiment_id='P-20261002-contact-risk-interventions',family='HF23',
        probe_index_in_family=1,run_status='RUNNING',smoke_only=args.engineering,pid=os.getpid(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        input_sha256=hashes,phases=phases,admissions=admissions,
        wall_limit_seconds=600 if args.engineering else 3600,
        prior_engineering_seconds=prior_engineering+60.+16.45606490969658,
        prior_native_engineering_seconds=prior_engineering,prior_static_engineering_seconds=16.45606490969658,
        preparation_budget_seconds=60.,analysis_reserve_seconds=120.,output_limit_bytes=8<<30,
        max_owned_gpu_concurrency=len(devices),model_training=False,expert_training=False,
        scope='randomized H10 generated candidates and full audits; utility analysis separate')
    lock=threading.RLock();stop=threading.Event();processes={}
    def save():
        with lock:
            manifest['elapsed_seconds']=time.monotonic()-begin
            manifest['cumulative_seconds']=manifest['elapsed_seconds']+manifest['prior_engineering_seconds']
            temporary=output/'run_manifest.json.tmp'
            temporary.write_text(json.dumps(manifest,indent=2)+'\n')
            temporary.replace(output/'run_manifest.json')
    def check():
        if stop.is_set():
            raise RuntimeError('another owned phase failed; no new work')
        if (args.engineering and time.monotonic()-begin>600) or (not args.engineering and
            time.monotonic()-begin+manifest['prior_engineering_seconds']+120.>3600):
            raise TimeoutError('whole fixed slot wall budget')
        size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if size+engineering_bytes>8<<30:
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
                command=list(template['phases'][0]['native_command'])
                command[2]=str(ROOT/'scripts/collect_contact_risk_interventions.py')
                changes={'--output-dir':str(directory),'--panel-seed':str(seed),
                    '--assignment-seed':str(seed+16000),'--seed':str(seed),
                    '--windows-per-stratum':'1' if args.engineering else '2',
                    '--max-steps':'160' if args.engineering else '300','--wall-seconds':'300' if args.engineering else '600',
                    '--generator-checkpoint':str(checkpoint),
                    '--output':str(directory/'unused.json'),'--output_path':str(directory/'player')}
                for key,value in changes.items():
                    command[command.index(key)+1]=value
                env['PYTHONHASHSEED']=str(seed)
                with lock:
                    phase.update(run_status='RUNNING',admission=current)
                save()
                execute(command,env,output/f'seed{seed}.native.log',phase,'native',330 if args.engineering else 640)
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
                audit_command=[PYTHON,'-u',str(ROOT/'scripts/audit_contact_risk_interventions.py'),
                    '--run',str(control),'--output',str(audit_path),'--gpu',str(gpu)]
                execute(audit_command,env,output/f'seed{seed}.audit.log',phase,'audit',240 if args.engineering else 450)
                audit=json.loads(audit_path.read_text())
                if audit['run_status']!='COMPLETED' or len(audit['phases'])!=1:
                    raise ValueError('terminal independent full audit')
                with lock:
                    phase.update(run_status='COMPLETED',audit=str(audit_path),audit_sha256=sha(audit_path))
                check();save()
                print(json.dumps(dict(seed=seed,gpu=gpu,rows=result['rows'],
                    source_seconds=phase['native_elapsed_seconds'],audit_seconds=phase['audit_elapsed_seconds'],
                    complete_phases=sum(p['run_status']=='COMPLETED' for p in phases),
                    cumulative_seconds=time.monotonic()-begin+manifest['prior_engineering_seconds'])),flush=True)
            except BaseException as exc:
                with lock:
                    phase.update(run_status='FAILED',error=repr(exc))
                stop.set();save()
                raise
    save()
    pool=ThreadPoolExecutor(max_workers=len(devices))
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
    p.add_argument('--gpus',type=int,nargs='+',default=[4,5,6])
    p.add_argument('--engineering',action='store_true')
    p.add_argument('--engineering-run',type=Path)
    run(p.parse_args())
