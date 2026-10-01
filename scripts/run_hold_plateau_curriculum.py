#!/usr/bin/env python3
"""One final-only baseline training/evaluation Probe in the isolated worktree."""
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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--references',type=Path,required=True);parser.add_argument('--gpu',type=int,default=5);args=parser.parse_args()
    output=args.output.resolve();references=args.references.resolve()
    if output.exists() or ROOT not in output.parents:raise ValueError('unique isolated output required')
    old=ROOT/'src/task/CmResidual/research/contact_response/output/P-20261001-hold-plateau-substrate-r1'
    prior=json.loads((old/'run_manifest.json').read_text());hashes=dict(prior['input_sha256'])
    correction=json.loads((old/'analysis_correction_r1/trajectory_audit.json').read_text())
    if correction['status']!='PASS' or correction['independently_scored_episodes']!=768:
        raise ValueError('prior task not independently closed')
    original=json.loads((REFERENCE/'run_manifest.json').read_text())
    template=next(p for p in original['phases'] if p['training_seed']==286 and p['evaluation_seed']==288 and not p['repeat'])
    command=template['command'];source=Path(command[command.index('--checkpoint')+1])
    source_sha=command[command.index('--checkpoint-sha256')+1]
    sources=[Path(__file__),ROOT/'scripts/train_hold_plateau_environment.py',ROOT/'scripts/analyze_hold_plateau_curriculum.py',
        ROOT/'src/task/CmResidual/hold_plateau_training.py',ROOT/'src/task/CmResidual/hold_plateau_agent.py',
        ROOT/'docs/experiments/probes/P-20261001-hold-plateau-curriculum.md',
        ROOT/'docs/decisions/D-20261001-hold-plateau-curriculum.md',old/'analysis_correction_r1/trajectory_audit.json',
        old/'analysis_correction_r1/results.json',ROOT/'src/task/CmResidual/tools/dexplore_cm_off_rank_bootstrap.py',
        ROOT/'src/task/CmResidual/tools/dexplore_ddp_rank_bootstrap.py',ROOT/'src/task/CmResidual/tools/dexplore_ddp_compat.py',
        ROOT/'src/task/CmResidual/dexplore_approach_agent.py',ROOT/'third_party/DExplore/dexplore/run.py',
        ROOT/'third_party/DExplore/dexplore/learning/dexplore_agent.py',ROOT/'third_party/DExplore/dexplore/learning/common_agent.py']
    library=Path(PYTHON).parents[1]/'lib/python3.8/site-packages/rl_games'
    sources.extend([library/'torch_runner.py',library/'algos_torch/torch_ext.py',library/'common/a2c_common.py'])
    hashes.update({str(p.resolve()):sha(p) for p in sources})
    if sha(source)!=source_sha or any(sha(Path(p))!=h for p,h in hashes.items()):raise ValueError('input/source drift')
    import yaml  # Pure configuration handling; no model compute on the parent.
    env_config=yaml.safe_load(Path(command[command.index('--cfg_env')+1]).read_text())
    train_config=yaml.safe_load(Path(command[command.index('--cfg_train')+1]).read_text())
    env_config['env'].update(numEnvs=96,hybridInitProb=1.,enableEarlyTermination=False)
    config=train_config['params']['config']
    config.update(full_experiment_name='hold_plateau_baseline_s721',horizon_length=16,minibatch_size=512,mini_epochs=4,
        save_frequency=1000,save_best_after=1000,save_intermediate=False,learning_rate=1e-5,lr_schedule='constant')
    output.mkdir(parents=True)
    for name,data in (('environment.yaml',env_config),('training.yaml',train_config)):
        p=output/name;p.write_text(yaml.safe_dump(data,sort_keys=False));hashes[str(p)]=sha(p)
    final=output/'train/hold_plateau_baseline_s721/nn/GRAB.pth'
    begin=time.monotonic()
    manifest=dict(experiment_id='P-20261001-hold-plateau-curriculum',run_id=output.name,run_status='RUNNING',pid=os.getpid(),
        isolated_worktree=str(ROOT),source_checkpoint=str(source),source_checkpoint_sha256=source_sha,
        final_checkpoint=str(final),synthetic_task=True,no_cm=True,training_seed=721,new_epochs=300,new_interactions=460800,
        panels=[[721,500],[721,501]],phases=[],input_sha256=hashes,wall_limit_seconds=3000,storage_limit_bytes=1<<30,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    def save():
        manifest['wall_seconds']=time.monotonic()-begin
        (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if time.monotonic()-begin>3000:raise TimeoutError('whole baseline Probe budget')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>1<<30:raise RuntimeError('storage budget')
        if any(sha(Path(p))!=h for p,h in hashes.items()):raise ValueError('protected input drift')
    def execute(name,cmd,timeout,native=True):
        check();gpu=admission(args.gpu) if native else None
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'] if native else '',LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_EXTENSIONS_DIR=str(output/'cache/torch_extensions'),XDG_CACHE_HOME=str(output/'cache'),
            TORCHINDUCTOR_CACHE_DIR=str(output/'cache/inductor'),TRITON_CACHE_DIR=str(output/'cache/triton'),
            REF2DEX_HOLD_REFERENCES_MANIFEST=str(references/'run_manifest.json'))
        env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        phase=dict(name=name,command=cmd,run_status='RUNNING',admission=gpu,
            device_reason='GPU native simulation/model compute' if native else 'CPU checkpoint hashes and trajectory labels, no model inference')
        manifest['phases'].append(phase);save();process=None;start=time.monotonic()
        print(json.dumps(dict(name=name,status='STARTED')),flush=True)
        try:
            with (output/(name+'.log')).open('x') as log:
                process=subprocess.Popen(cmd,cwd=ROOT/'third_party/DExplore',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                phase.update(pid=process.pid,pgid=process.pid);save()
                code=process.wait(timeout=min(timeout,max(1,3000-(time.monotonic()-begin))))
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
        train=[PYTHON,'-u',str(ROOT/'scripts/train_hold_plateau_environment.py'),'--cm-distill-coef','0',
            '--actual-epochs','300','--scratch-resume-checkpoint',str(source),'--scratch-resume-sha256',source_sha,
            '--learning-rate','1e-5','--task','Dexplore_Inspire','--cfg_env',str(output/'environment.yaml'),
            '--cfg_train',str(output/'training.yaml'),'--checkpoint',str(source),'--motion_file',str(references/'references'),
            '--headless','--num_envs','96','--seed','721','--sim_device','cuda:0','--rl_device','cuda:0',
            '--graphics_device_id','0','--output_path',str(output/'train')]
        execute('training',train,1800)
        if not final.is_file():raise ValueError('missing fixed endpoint')
        hashes[str(final)]=sha(final);manifest['final_checkpoint_sha256']=hashes[str(final)];save()
        execute('checkpoint_audit',[PYTHON,'-u',str(ROOT/'scripts/analyze_hold_plateau_curriculum.py'),'--directory',str(output),'--checkpoint-only'],120,native=False)
        for evaluation in (500,501):
            name=f't721_s{evaluation}';directory=output/name
            cmd=list(template['command']);cmd[2]=str(ROOT/'scripts/run_hold_plateau_environment.py')
            pos=cmd.index('--arm');del cmd[pos:pos+2]
            for flag,value in (('--run-dir',directory),('--training-seed',721),('--eval-seed',evaluation),('--seed',evaluation),
                ('--checkpoint',final),('--checkpoint-sha256',hashes[str(final)]),('--num_envs',192),('--wall-seconds',300),
                ('--cfg_env',output/'environment.yaml'),('--cfg_train',output/'training.yaml'),('--motion_file',references/'references'),
                ('--output',directory/'unused.json'),('--output_path',directory/'player')):
                cmd[cmd.index(flag)+1]=str(value)
            cmd.extend(['--references-manifest',str(references/'run_manifest.json')]);execute(name,cmd,400)
        manifest['run_status']='COLLECTION_COMPLETED';save()
        execute('analysis',[PYTHON,'-u',str(ROOT/'scripts/analyze_hold_plateau_curriculum.py'),'--directory',str(output)],120,native=False)
        check();result=json.loads((output/'results.json').read_text())
        manifest.update(run_status='COMPLETED',label=result['label'],protected_inputs_unchanged=True)
    except BaseException as error:manifest.update(run_status='FAILED',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
