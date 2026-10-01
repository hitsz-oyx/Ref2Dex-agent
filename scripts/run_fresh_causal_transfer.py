#!/usr/bin/env python3
"""Sequential fresh reference/acquisition/physical-arm matrix with frozen models."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import admission,sha,PYTHON,REFERENCE

PANELS=((286,490),(286,491),(287,490),(287,491))
ARMS=(('zero_a',0,0.),('zero_b',0,0.),('x_plus',0,.01),('x_minus',0,-.01),
      ('y_plus',1,.01),('y_minus',1,-.01),('z_plus',2,.01),('z_minus',2,-.01))
MODELS=ROOT/'src/task/CmResidual/research/contact_response/output/P-20261001-actuation-effect-factorization-r1'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    if output.exists() or ROOT not in output.parents:raise ValueError('unique isolated output required')
    reference=json.loads((REFERENCE/'run_manifest.json').read_text())
    templates={p['training_seed']:p for p in reference['phases'] if not p['repeat'] and p['evaluation_seed']==288}
    if reference['run_status']!='COMPLETED':raise ValueError('reference template incomplete')
    frozen=json.loads((MODELS/'run_manifest.json').read_text())
    if frozen['run_status']!='COMPLETED':raise ValueError('unfrozen predictor')
    sources=[Path(__file__),ROOT/'scripts/run_geometry_causal_environment.py',ROOT/'scripts/analyze_fresh_causal_transfer.py',
             ROOT/'src/task/CmResidual/causal_acquisition.py',ROOT/'src/task/CmResidual/actuation_effect.py',
             ROOT/'src/task/CmResidual/dexplore_cm_geometry.py',ROOT/'src/task/CmResidual/v118_planner.py',
             ROOT/'src/task/CmResidual/paired_evaluation.py',ROOT/'src/task/CmResidual/physical_value_live.py',
             ROOT/'src/task/CmResidual/physical_value_contract.py',ROOT/'scripts/run_contact_response_probe.py',
             ROOT/'scripts/run_paired_physical_value_environment.py',
             ROOT/'third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/cm_geometry.py',
             ROOT/'third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py',
             ROOT/'third_party/DExplore/dexplore/env/tasks/base_dexplore_task.py',
             ROOT/'third_party/DExplore/dexplore/evaluate.py',ROOT/'docs/experiments/probes/P-20261001-fresh-causal-transfer.md']
    for template in templates.values():
        command=template['command']
        sources.extend(Path(command[command.index(flag)+1]) for flag in ('--cfg_env','--cfg_train','--checkpoint'))
        motion=Path(command[command.index('--motion_file')+1]);sources.extend(p for p in motion.rglob('*') if p.is_file())
    sources.extend(MODELS/f'effect_{method}_s{seed}.pt' for method in ('nominal_motion','raw_command','state_only') for seed in (411,412,413))
    asset=ROOT/'third_party/DExplore/dexplore/data/assets'
    sources.extend(p for p in (asset/'inspire_hand_new').rglob('*') if p.is_file())
    sources.extend(p for p in (asset/'mjcf').rglob('*') if p.is_file() and ('airplane' in str(p)))
    hashes={str(p.resolve()):sha(p) for p in sources};output.mkdir(parents=True);begin=time.monotonic()
    manifest=dict(experiment_id='P-20261001-fresh-causal-transfer',run_id=output.name,run_status='RUNNING',pid=os.getpid(),
                  panels=PANELS,phases=[],frozen_models=str(MODELS),input_sha256=hashes,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  wall_limit_seconds=3600,storage_limit_bytes=4*(1<<30),no_training=True)
    def save():
        manifest['wall_seconds']=time.monotonic()-begin
        (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if time.monotonic()-begin>3300:raise TimeoutError('whole causal budget')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>4*(1<<30):raise RuntimeError('storage budget')
        if any(sha(Path(p))!=h for p,h in hashes.items()):raise ValueError('input/source drift')
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
            if code:raise RuntimeError(f'{name} exit{code}; inspect log')
            phase['run_status']='COMPLETED'
        except BaseException as error:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
            phase.update(run_status='FAILED',error=repr(error));raise
        finally:
            phase['wall_seconds']=time.monotonic()-start;save()
        print(json.dumps(dict(name=name,status='COMPLETED',wall_seconds=phase['wall_seconds'])),flush=True)
    def native_command(training,seed,name,mode,axis=0,amplitude=0.):
        command=list(templates[training]['command']);command[2]=str(ROOT/'scripts/run_geometry_causal_environment.py')
        for flag in ('--arm','--training-seed','--eval-seed'):
            pos=command.index(flag);del command[pos:pos+2]
        directory=output/name
        for flag,value in (('--run-dir',directory),('--seed',seed),('--wall-seconds',150),('--output',directory/'unused.json'),('--output_path',directory/'player')):
            command[command.index(flag)+1]=str(value)
        command+=['--mode',mode]
        if mode=='replay':
            ref=output/f't{training}_s{seed}_reference'
            command+=['--initial',str(ref/'initial_state.pt'),'--trace',str(ref/'trace.pt'),'--schedule',str(ref/'schedule.pt'),
                      '--axis',str(axis),'--amplitude',str(amplitude)]
        return command
    save()
    try:
        coverage=[]
        for training,seed in PANELS:
            name=f't{training}_s{seed}_reference'
            execute(name,native_command(training,seed,name,'reference'),180)
            directory=output/name
            result=json.loads((directory/'results.json').read_text())
            if result['run_status']!='COMPLETED':raise ValueError('reference incomplete')
            execute(name+'_schedule',[PYTHON,'-u',str(ROOT/'scripts/analyze_fresh_causal_transfer.py'),'schedule','--directory',str(directory)],180)
            coverage.append(dict(panel=[training,seed],**json.loads((directory/'schedule.json').read_text())))
        manifest['coverage']=coverage;save()
        if sum(row['valid_windows'] for row in coverage)<256 or any(row['valid_windows']<48 for row in coverage):
            result=dict(run_status='COMPLETED',label='UNCLEAR',reason='INSUFFICIENT_ACQUISITION_COVERAGE',coverage=coverage)
            (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
            manifest.update(run_status='COMPLETED',label='UNCLEAR');check();return
        for training,seed in PANELS:
            for arm,axis,amplitude in ARMS:
                name=f't{training}_s{seed}_{arm}'
                execute(name,native_command(training,seed,name,'replay',axis,amplitude),180)
                directory=output/name
                result=json.loads((directory/'results.json').read_text())
                if result['run_status']!='COMPLETED':raise ValueError('response incomplete')
                manifest['phases'][-1]['results_sha256']=sha(directory/'results.json');save()
        manifest['run_status']='COLLECTION_COMPLETED';save()
        execute('analysis',[PYTHON,'-u',str(ROOT/'scripts/analyze_fresh_causal_transfer.py'),'analyze','--directory',str(output)],180)
        result=json.loads((output/'results.json').read_text());check()
        manifest.update(run_status='COMPLETED',label=result['label'],inputs_unchanged=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
