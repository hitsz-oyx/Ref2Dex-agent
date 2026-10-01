#!/usr/bin/env python3
"""Bounded real contact-consequence support for the HF09 ranker Probe."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission,PYTHON,R7,MOTIONS


def run(args):
    experiment_id='P-20261001-contact-consequence-ranking'
    base=(ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    output=args.output.absolute()
    if output.parent.resolve()!=base or output.is_symlink():raise ValueError('outside owned task output')
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    previous=[]
    if args.prior:
        old=json.loads((args.prior/'run_manifest.json').read_text())
        if old['experiment_id']!=experiment_id or old['run_status'] not in ('FAILED','STOPPED'):
            raise ValueError('prior is not same terminal failed experiment')
        previous=list(old.get('prior_attempts',[]))+[dict(path=str(args.prior.resolve()),seconds=old['elapsed_seconds'],
                    bytes=sum(p.stat().st_size for p in args.prior.rglob('*') if p.is_file()))]
    prior_seconds=sum(p['seconds'] for p in previous);prior_bytes=sum(p['bytes'] for p in previous)
    route_path=ROOT/'src/task/CmResidual/configs/hf02_temporal_canonical_route.json'
    route=json.loads(route_path.read_text());inputs=[route_path,R7/'environment.yaml',R7/'training.yaml']
    for spec in route['experts'].values():
        path=(ROOT/spec['checkpoint']).resolve()
        if sha(path)!=spec['sha256']:raise ValueError('expert drift')
        inputs.append(path)
    for spec in route['motions']:
        path=MOTIONS/spec['name']/'interaction_hand_inspire.pt'
        if sha(path)!=spec['interaction_hand_sha256']:raise ValueError('motion drift')
        inputs.append(path)
    sources=[Path(__file__),ROOT/'scripts/collect_randomized_contact_consequences.py',
             ROOT/'scripts/collect_contact_consequences.py',ROOT/'src/task/CmResidual/contact_consequence.py',
             ROOT/'src/task/CmResidual/paired_evaluation.py',ROOT/'src/task/CmResidual/physical_value_live.py',
             ROOT/'third_party/DExplore/dexplore/evaluate.py']
    hashes={str(p.resolve()):sha(p) for p in inputs+sources}
    manifest=dict(experiment_id=experiment_id,family='HF09',probe_index_in_family=2,run_status='RUNNING',
                  pid=os.getpid(),command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  phases=[],input_sha256=hashes,prior_attempts=previous,wall_limit_seconds=3600,output_limit_bytes=8<<30,
                  actor_training=False,cm_training=False,stage='randomized_actual_state_collection')
    def save(): (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if prior_seconds+time.monotonic()-started>3540:raise TimeoutError('collection/fitting budget')
        if prior_bytes+sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>8<<30:raise ValueError('storage budget')
        if any(sha(p)!=hashes[str(p.resolve())] for p in sources):raise ValueError('source drift during collection')
    save()
    try:
        for seed in (331,332,333):
            check();admission=gpu_admission(args.gpu);directory=output/f'seed{seed}'
            source=(ROOT/route['experts']['source_e260']['checkpoint']).resolve()
            command=[PYTHON,'-u',str(ROOT/'scripts/collect_randomized_contact_consequences.py'),
                     '--output-dir',str(directory),'--panel-seed',str(seed),'--assignment-seed',str(7000+seed),
                     '--windows-per-episode','8','--wall-seconds','240',
                     '--task','Dexplore_Inspire','--cfg_env',str(R7/'environment.yaml'),'--cfg_train',str(R7/'training.yaml'),
                     '--checkpoint',str(source),'--motion_file',str(MOTIONS),'--headless','--num_envs','96','--seed',str(seed),
                     '--sim_device','cuda:0','--rl_device','cuda:0','--graphics_device_id','0','--disable-early-termination',
                     '--output',str(directory/'unused.json'),'--output_path',str(directory/'player')]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                     LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                     PYTHONHASHSEED=str(seed),TORCH_EXTENSIONS_DIR=str(output/'cache/torch_extensions'),XDG_CACHE_HOME=str(output/'cache'))
            env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
            phase=dict(seed=seed,assignment_seed=7000+seed,run_status='STARTED',gpu=admission,command=command,directory=str(directory))
            manifest['phases'].append(phase);save();process=None;begin=time.monotonic()
            print(json.dumps(dict(seed=seed,run_status='STARTED')),flush=True)
            try:
                with (output/f'seed{seed}.log').open('x') as log:
                    process=subprocess.Popen(command,cwd=ROOT/'third_party/DExplore',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    phase.update(pid=process.pid,pgid=process.pid);save()
                    code=process.wait(timeout=min(290,3540-prior_seconds-(time.monotonic()-started)))
                if code:raise RuntimeError(f'native exit{code} seed{seed}')
                result=json.loads((directory/'results.json').read_text())
                if result['run_status']!='COMPLETED':raise ValueError('incomplete support collection')
                phase.update(run_status='COMPLETED',result=result,record_sha256=sha(directory/'records.pt'))
                print(json.dumps(dict(seed=seed,result=result)),flush=True)
            except BaseException as error:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                phase.update(run_status='FAILED',error=repr(error));raise
            finally:phase['elapsed_seconds']=time.monotonic()-begin;save()
        if any(sha(Path(p))!=digest for p,digest in hashes.items()):raise ValueError('original input mutated')
        manifest.update(run_status='COMPLETED',stage='COLLECTION_COMPLETE_AWAITING_MODEL_FIT',input_hashes_unchanged=True,
                        cumulative_seconds=prior_seconds+time.monotonic()-started,
                        output_bytes=prior_bytes+sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
        check()
    except BaseException as error:manifest.update(run_status='FAILED',error=repr(error));raise
    finally:manifest['elapsed_seconds']=time.monotonic()-started;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=4)
    p.add_argument('--prior',type=Path)
    run(p.parse_args())
