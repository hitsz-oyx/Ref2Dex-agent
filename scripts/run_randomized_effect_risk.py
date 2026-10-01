#!/usr/bin/env python3
"""Four prospective randomized batches and one frozen-model risk comparison."""
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

PANELS=((286,492),(286,493),(287,492),(287,493))
MODELS=ROOT/'src/task/CmResidual/research/contact_response/output/P-20261001-actuation-effect-factorization-r1'
PRIOR=ROOT/'src/task/CmResidual/research/contact_response/output/P-20261001-fresh-causal-transfer-r1/run_manifest.json'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output.resolve()
    if output.exists() or ROOT not in output.parents:raise ValueError('unique independent output required')
    original=json.loads((REFERENCE/'run_manifest.json').read_text())
    templates={p['training_seed']:p for p in original['phases'] if not p['repeat'] and p['evaluation_seed']==288}
    previous=json.loads(PRIOR.read_text())
    # Recheck all inherited physical/model inputs; never mutate a failed run.
    hashes=dict(previous['input_sha256'])
    sources=[Path(__file__),ROOT/'scripts/run_randomized_effect_environment.py',ROOT/'scripts/analyze_randomized_effect_risk.py',
             ROOT/'src/task/CmResidual/randomized_effect_risk.py',ROOT/'docs/experiments/probes/P-20261001-randomized-effect-risk.md']
    hashes.update({str(p.resolve()):sha(p) for p in sources})
    if any(sha(Path(p))!=value for p,value in hashes.items()):raise ValueError('inherited/source drift before randomized trial')
    if json.loads((MODELS/'run_manifest.json').read_text())['run_status']!='COMPLETED':raise ValueError('model fit incomplete')
    output.mkdir(parents=True);begin=time.monotonic()
    manifest=dict(experiment_id='P-20261001-randomized-effect-risk',run_id=output.name,run_status='RUNNING',pid=os.getpid(),
                  panels=PANELS,phases=[],frozen_models=str(MODELS),input_sha256=hashes,no_training=True,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  wall_limit_seconds=3600,storage_limit_bytes=2*(1<<30),new_random_assignment=True)
    def save():
        manifest['wall_seconds']=time.monotonic()-begin
        (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if time.monotonic()-begin>3300:raise TimeoutError('whole randomized budget')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>2*(1<<30):raise RuntimeError('storage budget')
        if any(sha(Path(p))!=value for p,value in hashes.items()):raise ValueError('input/source drift')
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
            if code:raise RuntimeError(f'{name} exit{code}; inspect its retained log')
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
            execute(name,command,660)
            record=json.loads((directory/'results.json').read_text());coverage.append(dict(panel=[training,evaluation],**record))
            manifest['coverage']=coverage;save()
            if not record['all_windows_complete']:
                result=dict(run_status='COMPLETED',label='UNCLEAR',reason='INSUFFICIENT_COMPLETE_ACQUISITION',coverage=coverage)
                (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');check()
                manifest.update(run_status='COMPLETED',label='UNCLEAR',inputs_unchanged=True);return
        manifest['run_status']='COLLECTION_COMPLETED';save()
        execute('analysis',[PYTHON,'-u',str(ROOT/'scripts/analyze_randomized_effect_risk.py'),'--directory',str(output)],360)
        result=json.loads((output/'results.json').read_text());check()
        manifest.update(run_status='COMPLETED',label=result['label'],inputs_unchanged=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
