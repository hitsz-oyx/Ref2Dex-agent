#!/usr/bin/env python3
"""Single owned native seed followed by full physical/planner audit."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission,PYTHON
BASE=ROOT/'src/task/CmResidual/research/contact_consequence/output'


def run(args):
    started=time.monotonic()
    if args.output.parent.resolve()!=BASE.resolve() or args.output.exists():raise ValueError('unique owned native output')
    old_path=BASE/'P-20261002-optimized-contact-native-engineering-r2/run_manifest.json'
    old=json.loads(old_path.read_text())
    if old['run_status']!='COMPLETED' or old['child_exit_code']!=0:raise ValueError('valid native substrate required')
    fit=BASE/'P-20261002-structured-contact-fit-r1/structured_contact_consequence.pt'
    fit_audit=ROOT/'docs/experiments/probes/P-20261002-structured-contact-fit-audit-r1.json'
    gradient=BASE/'P-20261002-structured-action-engineering-r1/results.json'
    info=json.loads(fit_audit.read_text());gen=json.loads(gradient.read_text())
    if not info['audit_passed'] or info['label']!='PROMISING' or info['checkpoint_sha256']!=sha(fit) or not gen['engineering_passed']:
        raise ValueError('audited information and generation engineering required')
    hashes=dict(old['input_sha256']);hashes.update(gen['input_sha256'])
    for path in (Path(__file__),fit,fit_audit,gradient,old_path,
        ROOT/'scripts/collect_structured_contact_source.py',ROOT/'scripts/audit_structured_contact_source.py',
        ROOT/'docs/experiments/probes/P-20261002-structured-native-engineering.md'):
        hashes[str(path.resolve())]=sha(path)
    if any(sha(Path(k))!=v for k,v in hashes.items()):raise ValueError('native engineering input drift')
    admission=gpu_admission(args.gpu);args.output.mkdir()
    directory=args.output/'seed610'
    command=list(old['phases'][0]['command']);command[command.index(str(ROOT/'scripts/collect_optimized_contact_source.py'))]=str(ROOT/'scripts/collect_structured_contact_source.py')
    changes={'--output-dir':str(directory),'--panel-seed':'610','--assignment-seed':'15610','--seed':'610',
        '--windows-per-stratum':'1','--max-steps':'180','--wall-seconds':'420',
        '--generator-checkpoint':str(fit),'--output':str(directory/'unused.json'),'--output_path':str(directory/'player')}
    for key,value in changes.items():command[command.index(key)+1]=value
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        PYTHONHASHSEED='610',PYTHONDONTWRITEBYTECODE='1',LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',
        TORCH_EXTENSIONS_DIR=str(args.output/'cache/torch_extensions'),XDG_CACHE_HOME=str(args.output/'cache'),
        CUBLAS_WORKSPACE_CONFIG=':4096:8')
    env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
    phase=dict(seed=610,run_status='RUNNING',directory=str(directory.resolve()))
    manifest=dict(run_status='RUNNING',smoke_only=True,pid=os.getpid(),input_sha256=hashes,phases=[phase],gpu=admission,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        wall_limit_seconds=900,output_limit_bytes=128<<20,scope='excluded native engineering; no utility claim')
    def save():
        manifest['elapsed_seconds']=time.monotonic()-started
        (args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if time.monotonic()-started>900:raise TimeoutError('engineering wall bound')
        if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())>128<<20:raise ValueError('engineering storage bound')
        if any(sha(Path(k))!=v for k,v in hashes.items()):raise ValueError('engineering input drift')
    def execute(cmd,kind,limit):
        check();begin=time.monotonic()
        with (args.output/(kind+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT/'third_party/DExplore',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            phase[kind+'_pid']=child.pid;phase[kind+'_command']=cmd;save()
            try:
                while child.poll() is None:
                    if time.monotonic()-begin>limit:raise TimeoutError(kind+' bounded timeout')
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:check()
                phase[kind+'_exit_code']=child.returncode
                if child.returncode:raise RuntimeError(kind+' child exit '+str(child.returncode))
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
                phase[kind+'_exit_code']=child.returncode;raise
            finally:
                phase[kind+'_elapsed_seconds']=time.monotonic()-begin;save()
    save()
    try:
        execute(command,'native',480)
        result=json.loads((directory/'results.json').read_text())
        if result['run_status']!='COMPLETED' or sha(directory/'records.pt')!=result['record_sha256'] or sha(directory/'planning.pt')!=result['planning_sha256']:
            raise ValueError('complete native artifacts required')
        phase.update(run_status='COMPLETED',result=result,exit_code=0)
        control=args.output/'phase-control';control.mkdir()
        (control/'run_manifest.json').write_text(json.dumps(dict(run_status='COMPLETED',child_exit_code=0,
            input_sha256=hashes,phases=[dict(phase)],scope='native phase terminal, parent engineering audit pending'),indent=2)+'\n')
        audit=args.output/'audit.json'
        execute([PYTHON,'-u',str(ROOT/'scripts/audit_structured_contact_source.py'),'--run',str(control),'--output',str(audit),'--gpu',str(args.gpu)],'audit',390)
        report=json.loads(audit.read_text())
        if report['run_status']!='COMPLETED' or len(report['phases'])!=1:raise ValueError('full native audit required')
        check();phase.update(audit=str(audit),audit_sha256=sha(audit))
        manifest.update(run_status='COMPLETED',child_exit_code=0,engineering_passed=True,input_hashes_unchanged=True)
        save();print(json.dumps(dict(run_status='COMPLETED',engineering_passed=True,rows=result['rows'],elapsed_seconds=manifest['elapsed_seconds'])),flush=True)
    except BaseException as error:
        phase.update(run_status='FAILED',error=repr(error));manifest.update(run_status='FAILED',child_exit_code=1,error=repr(error));save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=1)
    args=p.parse_args();args.output=args.output.resolve();run(args)
