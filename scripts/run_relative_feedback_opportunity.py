#!/usr/bin/env python3
"""Fixed six-seed native candidate opportunity source with full independent audits."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import PYTHON,gpu_admission,sha


def run(args):
    begin=time.monotonic();base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    output=args.output.resolve();engineering=base/'P-20261002-contact-relative-feedback-native-engineering-r2'
    if output.parent!=base or output.exists() or args.output.is_symlink():raise ValueError('unique owned source')
    previous=json.loads((engineering/'run_manifest.json').read_text())
    if previous['run_status']!='COMPLETED' or not previous['engineering_passed'] or not previous['smoke_only']:
        raise ValueError('accepted terminal engineering required')
    for pid in [previous['pid']]+[p['pid'] for p in previous['phases']]:
        if Path('/proc').joinpath(str(pid)).exists():raise ValueError('engineering still live')
    hashes=dict(previous['input_sha256'])
    for path in (engineering/'run_manifest.json',engineering/'audit.json',engineering/'seed669/records.pt',Path(__file__),
        ROOT/'docs/experiments/probes/P-20261002-relative-feedback-opportunity.md',
        ROOT/'scripts/analyze_relative_feedback_opportunity.py',ROOT/'scripts/audit_relative_feedback_opportunity_statistics.py'):
        hashes[str(path.resolve())]=sha(path)
    if any(sha(Path(k))!=v for k,v in hashes.items()):raise ValueError('fixed input drift')
    admission=gpu_admission(args.gpu);output.mkdir(exist_ok=False)
    prior=previous['cumulative_seconds']+60.
    prior_bytes=previous['output_bytes']+sum(p['bytes'] for p in previous['prior_attempts'])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',PYTHONDONTWRITEBYTECODE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_EXTENSIONS_DIR=str(output/'cache/torch_extensions'),XDG_CACHE_HOME=str(output/'cache'))
    env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
    manifest=dict(experiment_id='P-20261002-relative-feedback-opportunity',family='HF24',probe_index_in_family=1,
        run_status='RUNNING',smoke_only=False,pid=os.getpid(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        input_sha256=hashes,admission=admission,prior_budget_seconds=prior,engineering_bytes=prior_bytes,
        wall_limit_seconds=3600,output_limit_bytes=2<<30,analysis_reserve_seconds=180,phases=[])
    def save():
        manifest['elapsed_seconds']=time.monotonic()-begin;manifest['cumulative_seconds']=manifest['elapsed_seconds']+prior
        (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if time.monotonic()-begin+prior+180>3600:raise TimeoutError('whole opportunity source budget')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())+prior_bytes>2<<30:raise ValueError('source output budget')
        if any(sha(Path(k))!=v for k,v in hashes.items()):raise ValueError('fixed source inputs drift')
    def execute(phase,kind,command,limit):
        check();start=time.monotonic();process=None
        try:
            with (output/f"seed{phase['seed']}.{kind}.log").open('x') as log:
                process=subprocess.Popen(command,cwd=ROOT/'third_party/DExplore',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                phase[kind+'_pid']=process.pid;phase[kind+'_command']=command;save()
                while process.poll() is None:
                    if time.monotonic()-start>limit:raise TimeoutError(kind+' phase budget')
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:check();save()
                phase[kind+'_exit_code']=process.returncode
                if process.returncode:raise RuntimeError(kind+' exit '+str(process.returncode))
        except BaseException:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
            if process is not None:phase[kind+'_exit_code']=process.returncode
            raise
        finally:phase[kind+'_elapsed_seconds']=time.monotonic()-start;save()
    save()
    try:
        for seed in range(671,677):
            check();gpu_admission(args.gpu);directory=output/f'seed{seed}'
            command=list(previous['phases'][0]['command']);command.remove('--smoke-only')
            for key,value in {'--output-dir':str(directory),'--panel-seed':str(seed),'--assignment-seed':str(seed+17000),
                '--seed':str(seed),'--windows-per-stratum':'2','--max-steps':'300','--wall-seconds':'240',
                '--output':str(directory/'unused.json'),'--output_path':str(directory/'player')}.items():command[command.index(key)+1]=value
            phase=dict(seed=seed,directory=str(directory),run_status='RUNNING');manifest['phases'].append(phase);env['PYTHONHASHSEED']=str(seed);save()
            execute(phase,'native',command,300)
            result=json.loads((directory/'results.json').read_text())
            if result['smoke_only'] or result['run_status']!='COMPLETED' or result['saved_observation_expert_replay_max_error']>2e-5:
                raise ValueError('terminal unbiased science source')
            if sha(directory/'records.pt')!=result['record_sha256']:raise ValueError('record drift')
            phase['result']=result;gpu_admission(args.gpu)
            audit=output/f'seed{seed}.audit.json'
            execute(phase,'audit',[PYTHON,'-u',str(ROOT/'scripts/audit_contact_relative_feedback.py'),
                '--record',str(directory/'records.pt'),'--output',str(audit),'--gpu',str(args.gpu)],90)
            proof=json.loads(audit.read_text())
            if not proof['engineering_passed'] or proof['smoke_only']:raise ValueError('full science actuation and physics audit')
            phase.update(run_status='COMPLETED',audit=str(audit),audit_sha256=sha(audit));check();save()
            print(json.dumps(dict(seed=seed,rows=result['rows'],completed=len(manifest['phases']),
                cumulative_seconds=manifest['cumulative_seconds'])),flush=True)
        manifest.update(run_status='COMPLETED',child_exit_code=0,
            output_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())+prior_bytes)
    except BaseException as e:
        if manifest['phases'] and manifest['phases'][-1]['run_status']=='RUNNING':manifest['phases'][-1]['run_status']='FAILED'
        manifest.update(run_status='FAILED',error=repr(e));raise
    finally:save();print(json.dumps({k:v for k,v in manifest.items() if k not in ('input_sha256','phases','admission')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=5);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
