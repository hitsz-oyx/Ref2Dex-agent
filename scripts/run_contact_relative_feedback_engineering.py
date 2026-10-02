#!/usr/bin/env python3
"""Bounded, excluded native feedback smoke followed by independent GPU audit."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import PYTHON,sha,gpu_admission


def run(args):
    begin=time.monotonic();base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    output=args.output.resolve()
    if output.parent!=base or output.exists() or args.output.is_symlink():raise ValueError('unique owned engineering directory')
    static_path=ROOT/'docs/experiments/probes/P-20261002-contact-relative-feedback-static-r1.json'
    static=json.loads(static_path.read_text())
    if not static['engineering_passed']:raise ValueError('static mapping required')
    template_path=base/'P-20261002-structured-opportunity-engineering-r1/run_manifest.json'
    template=json.loads(template_path.read_text());command=list(template['phases'][0]['native_command'])
    command[2]=str(ROOT/'scripts/collect_contact_relative_feedback.py')
    position=command.index('--generator-checkpoint');del command[position:position+2]
    seed=669;directory=output/'seed669'
    for key,value in {'--output-dir':str(directory),'--panel-seed':str(seed),'--assignment-seed':str(seed+17000),
        '--seed':str(seed),'--windows-per-stratum':'1','--max-steps':'60','--wall-seconds':'180',
        '--output':str(directory/'unused.json'),'--output_path':str(directory/'player')}.items():
        command[command.index(key)+1]=value
    command.append('--smoke-only')
    paths=[Path(__file__),ROOT/'scripts/collect_contact_relative_feedback.py',ROOT/'scripts/audit_contact_relative_feedback.py',
        ROOT/'scripts/check_contact_relative_feedback.py',ROOT/'scripts/collect_contact_consequences.py',
        ROOT/'scripts/run_paired_evaluator_resolution.py',ROOT/'scripts/audit_contact_risk_interventions.py',
        ROOT/'src/task/CmResidual/contact_relative_feedback.py',ROOT/'src/task/CmResidual/executable_contact_options.py',
        ROOT/'src/task/CmResidual/weight_normalized_contact.py',ROOT/'src/task/CmResidual/paired_evaluation.py',
        ROOT/'src/task/CmResidual/configs/hf02_temporal_canonical_route.json',static_path,
        ROOT/'docs/decisions/D-20261002-contact-relative-feedback.md',template_path,
        ROOT/'third_party/DExplore/dexplore/env/tasks/base_dexplore_task.py',ROOT/'third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py',
        ROOT/'third_party/DExplore/dexplore/evaluate.py',ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf',
        ROOT/'third_party/DExplore/dexplore/data/assets/mjcf/objects/airplane/airplane.obj',
        ROOT/'third_party/DExplore/dexplore/data/assets/mjcf/objects/table/table.obj']
    for key in ('--cfg_env','--cfg_train','--checkpoint'):paths.append(Path(command[command.index(key)+1]))
    route=json.loads((ROOT/'src/task/CmResidual/configs/hf02_temporal_canonical_route.json').read_text())
    for spec in route['experts'].values():
        path=(ROOT/spec['checkpoint']).resolve()
        if sha(path)!=spec['sha256']:raise ValueError('expert identity drift')
        paths.append(path)
    motions=Path(command[command.index('--motion_file')+1])
    for spec in route['motions']:
        path=motions/spec['name']/'interaction_hand_inspire.pt'
        if sha(path)!=spec['interaction_hand_sha256']:raise ValueError('motion identity drift')
        paths.append(path)
    hashes={str(p.resolve()):sha(p) for p in paths}
    admission=gpu_admission(args.gpu);output.mkdir(exist_ok=False)
    prior=60.+float(static['elapsed_seconds'])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED=str(seed),
        CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_EXTENSIONS_DIR=str(output/'cache/torch_extensions'),XDG_CACHE_HOME=str(output/'cache'))
    env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
    manifest=dict(run_status='RUNNING',smoke_only=True,run_id=output.name,pid=os.getpid(),seed=seed,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),input_sha256=hashes,
        admission=admission,wall_limit_seconds=600,output_limit_bytes=128<<20,prior_budget_seconds=prior,phases=[])
    def save():
        manifest['elapsed_seconds']=time.monotonic()-begin;manifest['cumulative_seconds']=manifest['elapsed_seconds']+prior
        (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if time.monotonic()-begin+prior>600:raise TimeoutError('whole engineering budget')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>128<<20:raise ValueError('engineering output budget')
        if any(sha(Path(k))!=v for k,v in hashes.items()):raise ValueError('engineering inputs drift')
    def execute(kind,cmd,limit):
        check();started=time.monotonic();phase=dict(kind=kind,command=cmd,run_status='RUNNING')
        manifest['phases'].append(phase);process=None
        try:
            with (output/(kind+'.log')).open('x') as log:
                process=subprocess.Popen(cmd,cwd=ROOT/'third_party/DExplore',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                phase['pid']=process.pid;save()
                while process.poll() is None:
                    if time.monotonic()-started>limit:raise TimeoutError(kind+' phase budget')
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:check();save()
                phase['exit_code']=process.returncode
                if process.returncode:raise RuntimeError(kind+' exit '+str(process.returncode))
            phase['run_status']='COMPLETED'
        except BaseException:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
            if process is not None:phase['exit_code']=process.returncode
            phase['run_status']='FAILED';raise
        finally:phase['elapsed_seconds']=time.monotonic()-started;save()
    save()
    try:
        execute('native',command,220)
        result=json.loads((directory/'results.json').read_text())
        if not result['smoke_only'] or result['rows']<16 or min(result['arms'])<1:raise ValueError('engineering branch coverage incomplete')
        if result['saved_observation_expert_replay_max_error']>2e-5 or sha(directory/'records.pt')!=result['record_sha256']:
            raise ValueError('terminal expert replay or records drift')
        gpu_admission(args.gpu)
        audit=output/'audit.json'
        execute('audit',[PYTHON,'-u',str(ROOT/'scripts/audit_contact_relative_feedback.py'),
            '--record',str(directory/'records.pt'),'--output',str(audit),'--gpu',str(args.gpu)],180)
        evidence=json.loads(audit.read_text())
        if not evidence['engineering_passed'] or any(evidence['actuation'][i]['changed_windows']<1 for i in (2,3)):
            raise ValueError('both actual feedback branches must change native PD')
        check();manifest.update(run_status='COMPLETED',child_exit_code=0,engineering_passed=True,
            result=result,audit_sha256=sha(audit),record_sha256=sha(directory/'records.pt'),actuation=evidence['actuation'],
            output_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
    except BaseException as e:manifest.update(run_status='FAILED',error=repr(e));raise
    finally:save();print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=5);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
