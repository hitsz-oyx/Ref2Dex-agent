#!/usr/bin/env python3
"""Freeze fit-global scores, run fresh randomized policies, and score task utility."""
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

PANELS=((286,496),(286,497),(287,496),(287,497))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models',type=Path,required=True);parser.add_argument('--gpu',type=int,default=4);args=parser.parse_args()
    output=args.output.resolve();models=args.models.resolve()
    if output.exists() or ROOT not in output.parents:raise ValueError('unique isolated output required')
    fit=json.loads((models/'run_manifest.json').read_text())
    if fit['run_status']!='COMPLETED' or json.loads((models/'closeout_audit.json').read_text())['status']!='PASS':raise ValueError('fit incomplete/unaudited')
    hashes={**fit['input_sha256'],**fit['output_sha256']}
    sources=[Path(__file__),ROOT/'scripts/run_randomized_task_environment.py',ROOT/'scripts/analyze_randomized_task_selection.py',
        ROOT/'src/task/CmResidual/randomized_task_selection.py',ROOT/'docs/experiments/probes/P-20261001-randomized-task-selection.md',
        models/'run_manifest.json',models/'closeout_audit.json']
    hashes.update({str(path.resolve()):sha(path) for path in sources})
    if any(sha(Path(p))!=value for p,value in hashes.items()):raise ValueError('frozen input drift')
    # Pure label/statistics calculation on CPU, before any test physics.
    import torch
    data=torch.load(models/'fit_data.pt',map_location='cpu',weights_only=False)
    if sorted(data['acquisition_seed'].unique().tolist())!=[492,493]:raise ValueError('fit cohort drift')
    scores=torch.stack([(data['y'][:,2].double()*(data['arm']==arm)/data['propensity'][:,arm].double()).mean() for arm in range(7)])
    output.mkdir(parents=True);controls=output/'frozen_controls.pt'
    torch.save(dict(models=str(models),global_vertical_scores=scores,fit_rows=len(data['y']),
        fit_sha256=sha(models/'fit_data.pt'),policy_names=['actor','random','global','conditional']),controls)
    hashes[str(controls)]=sha(controls)
    original=json.loads((REFERENCE/'run_manifest.json').read_text())
    templates={p['training_seed']:p for p in original['phases'] if not p['repeat'] and p['evaluation_seed']==288}
    begin=time.monotonic();manifest=dict(experiment_id='P-20261001-randomized-task-selection',run_id=output.name,gpu_index=args.gpu,
        run_status='RUNNING',pid=os.getpid(),panels=PANELS,phases=[],frozen_models=str(models),input_sha256=hashes,
        global_vertical_scores_m=scores.tolist(),no_training=True,wall_limit_seconds=3600,storage_limit_bytes=2*(1<<30),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    def save():
        manifest['wall_seconds']=time.monotonic()-begin
        (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if time.monotonic()-begin>3300:raise TimeoutError('task probe budget')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>2*(1<<30):raise RuntimeError('storage budget')
        if any(sha(Path(p))!=value for p,value in hashes.items()):raise ValueError('input drift')
    def execute(name,command,timeout):
        check();gpu=admission(args.gpu)
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
            if code:raise RuntimeError(f'{name} exit{code}; retained log')
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
        for actor,evaluation in PANELS:
            name=f't{actor}_s{evaluation}';directory=output/name
            command=list(templates[actor]['command']);command[2]=str(ROOT/'scripts/run_randomized_task_environment.py')
            position=command.index('--arm');del command[position:position+2]
            for flag,value in (('--run-dir',directory),('--eval-seed',evaluation),('--seed',evaluation),('--num_envs',768),
                ('--wall-seconds',600),('--output',directory/'unused.json'),('--output_path',directory/'player')):
                command[command.index(flag)+1]=str(value)
            command.extend(['--controls',str(controls)]);execute(name,command,660)
            result=json.loads((directory/'results.json').read_text())
            if not result['all_episodes_complete'] or result['complete_episodes']!=768:raise ValueError('incomplete cohort')
        manifest['run_status']='COLLECTION_COMPLETED';save()
        execute('analysis',[PYTHON,'-u',str(ROOT/'scripts/analyze_randomized_task_selection.py'),'--directory',str(output)],360)
        result=json.loads((output/'results.json').read_text());check()
        manifest.update(run_status='COMPLETED',label=result['label'],inputs_unchanged=True)
    except BaseException as error:manifest.update(run_status='FAILED',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
