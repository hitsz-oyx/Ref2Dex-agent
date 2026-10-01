#!/usr/bin/env python3
"""Fresh IID randomized acquisition after freezing direct-response checkpoints."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import admission,sha,PYTHON,REFERENCE

PANELS=((286,494),(286,495),(287,494),(287,495))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models',type=Path,required=True);args=parser.parse_args()
    output=args.output.resolve();models=args.models.resolve()
    if output.exists() or ROOT not in output.parents:raise ValueError('unique independent output required')
    fit=json.loads((models/'run_manifest.json').read_text())
    if fit['run_status']!='COMPLETED' or fit['stage']!='FIT':raise ValueError('fit incomplete')
    hashes=dict(fit['input_sha256']);hashes.update(fit['output_sha256'])
    hashes[str(models/'run_manifest.json')]=sha(models/'run_manifest.json')
    if any(sha(Path(p))!=value for p,value in hashes.items()):raise ValueError('model/input/source drift before acquisition')
    original=json.loads((REFERENCE/'run_manifest.json').read_text())
    templates={p['training_seed']:p for p in original['phases'] if not p['repeat'] and p['evaluation_seed']==288}
    output.mkdir(parents=True);begin=time.monotonic()
    manifest=dict(experiment_id='P-20261001-direct-randomized-response',stage='TEST',run_id=output.name,
        run_status='RUNNING',pid=os.getpid(),panels=PANELS,phases=[],frozen_models=str(models),input_sha256=hashes,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        wall_limit_seconds=3600,storage_limit_bytes=2*(1<<30),assignment_mode='iid',no_training=True)
    def save():
        manifest['wall_seconds']=time.monotonic()-begin
        (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if time.monotonic()-begin>3300:raise TimeoutError('whole test budget')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>2*(1<<30):raise RuntimeError('storage budget')
        if any(sha(Path(p))!=value for p,value in hashes.items()):raise ValueError('frozen input drift')
    def execute(name,command,timeout):
        check();gpu=admission(4)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_EXTENSIONS_DIR=str(output/'cache/torch_extensions'),XDG_CACHE_HOME=str(output/'cache'))
        env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        phase=dict(name=name,command=command,run_status='RUNNING',admission=gpu);manifest['phases'].append(phase);save()
        process=None;start=time.monotonic();print(json.dumps(dict(name=name,status='STARTED')),flush=True)
        try:
            with (output/(name+'.log')).open('x') as log:
                process=subprocess.Popen(command,cwd=ROOT/'third_party/DExplore',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                phase.update(pid=process.pid,pgid=process.pid);save();code=process.wait(timeout=timeout)
            if code:raise RuntimeError(f'{name} exit{code}; inspect retained log')
            phase['run_status']='COMPLETED'
        except BaseException as error:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
            phase.update(run_status='FAILED',error=repr(error));raise
        finally:phase['wall_seconds']=time.monotonic()-start;save()
        print(json.dumps(dict(name=name,status='COMPLETED',wall_seconds=phase['wall_seconds'])),flush=True)
    save()
    try:
        coverage=[]
        for training,evaluation in PANELS:
            name=f't{training}_s{evaluation}';directory=output/name
            command=list(templates[training]['command']);command[2]=str(ROOT/'scripts/run_randomized_effect_environment.py')
            position=command.index('--arm');del command[position:position+2]
            for flag,value in (('--run-dir',directory),('--eval-seed',evaluation),('--seed',evaluation),('--num_envs',768),
                ('--wall-seconds',600),('--output',directory/'unused.json'),('--output_path',directory/'player')):
                command[command.index(flag)+1]=str(value)
            command.extend(['--assignment-mode','iid']);execute(name,command,660)
            record=json.loads((directory/'results.json').read_text());coverage.append(dict(panel=[training,evaluation],**record))
            manifest['coverage']=coverage;save()
            if not record['all_windows_complete']:
                result=dict(run_status='COMPLETED',label='UNCLEAR',reason='INSUFFICIENT_COMPLETE_ACQUISITION',coverage=coverage)
                (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');check()
                manifest.update(run_status='COMPLETED',label='UNCLEAR',inputs_unchanged=True);return
        manifest['run_status']='COLLECTION_COMPLETED';save()
        execute('analysis',[PYTHON,'-u',str(ROOT/'scripts/analyze_direct_randomized_test.py'),'--directory',str(output)],360)
        result=json.loads((output/'results.json').read_text());check()
        manifest.update(run_status='COMPLETED',label=result['label'],inputs_unchanged=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
