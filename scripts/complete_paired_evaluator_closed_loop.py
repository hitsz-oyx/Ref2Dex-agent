#!/usr/bin/env python3
"""Measure actual closed-loop off repeats before any existing-arm inference.

The saved-action replay panel diagnoses physics only. This completion uses
native actor actions and the same snapshots/RNG, and losslessly archives new
traces to stay inside the original audit's disk budget.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import ARMS,BASE,NN,PANELS,PYTHON,R7,gpu_admission,sha


def archive_trace(directory):
    """Preserve the complete raw byte stream and verify it before unlinking."""
    raw=directory/'trace.pt'
    original=sha(raw)
    packed=directory/'trace.pt.gz'
    if packed.exists(): raise FileExistsError(packed)
    with raw.open('rb') as source,gzip.open(packed,'wb',compresslevel=1) as target:
        for block in iter(lambda:source.read(1<<20),b''): target.write(block)
    h=hashlib.sha256()
    with gzip.open(packed,'rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    if h.hexdigest()!=original: raise ValueError('lossless trace verification failed')
    record=dict(archive=str(packed.resolve()),archive_sha256=sha(packed),
                raw_sha256=original,raw_bytes=raw.stat().st_size,archive_bytes=packed.stat().st_size,
                lossless_roundtrip_verified=True)
    (directory/'trace_archive.json').write_text(json.dumps(record,indent=2)+'\n')
    raw.unlink()  # only this new owned trace; verified identical byte copy retained
    return record


def run(args):
    from src.task.CmResidual.paired_evaluation import repeatability_gate
    reference=args.reference.resolve()
    old=json.loads((reference/'run_manifest.json').read_text())
    if old['run_status']!='COMPLETED' or old.get('decision')!='CLOSE_CURRENT_ROUTE_EVALUATOR_RESOLUTION':
        raise ValueError('expected terminal replay-only audit; never run alongside its children')
    old_result=json.loads((reference/'results.json').read_text())
    if old_result['policy_matrix_started']: raise ValueError('matrix already executed')
    output=args.output.absolute()
    if output.parent.resolve()!=BASE.resolve() or output.is_symlink(): raise ValueError('owned unique output required')
    output.mkdir(exist_ok=False)
    started=time.monotonic()
    prior=list(old.get('prior_attempts',[]))
    prior.append(dict(path=str(reference),elapsed_seconds=old['elapsed_seconds'],
                     bytes=sum(p.stat().st_size for p in reference.rglob('*') if p.is_file())))
    prior_seconds=sum(p['elapsed_seconds'] for p in prior)
    prior_bytes=sum(p['bytes'] for p in prior)
    inputs=dict(old['input_sha256'])
    inputs[str(Path(__file__).resolve())]=sha(__file__)
    inputs[str(reference/'run_manifest.json')]=sha(reference/'run_manifest.json')
    inputs[str(reference/'results.json')]=sha(reference/'results.json')
    off={}; off_phases={}
    for phase in old['phases']:
        if phase['arm']=='plain_off' and not phase['repeat']:
            key=(phase['training_seed'],phase['evaluation_seed'])
            directory=Path(phase['directory'])
            off[key]=json.loads((directory/'results.json').read_text())
            off_phases[key]=phase
            for name in ('initial_state.pt','trace.pt','results.json'):
                path=directory/name;inputs[str(path)]=sha(path)
    if set(off)!=set(PANELS): raise ValueError('incomplete baseline panel')
    if any(sha(Path(p))!=digest for p,digest in inputs.items()): raise ValueError('input drift before completion')
    manifest=dict(experiment_id='P-20261001-paired-evaluator-resolution',family='HD02',run_id='r4',
                  run_status='RUNNING',pid=os.getpid(),command=sys.argv,phases=[],input_sha256=inputs,
                  prior_attempts=prior,no_training=True,no_new_cm_mechanism=True,baseline_status='PARTIAL',
                  wall_limit_seconds=3600,storage_limit_bytes=8<<30,
                  correction='replay-only audit cannot substitute for true closed-loop repeats',
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    def save(): (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if prior_seconds+time.monotonic()-started>3540: raise TimeoutError('whole audit wall budget')
        if prior_bytes+sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>(8<<30):
            raise RuntimeError('whole audit storage budget')
    def execute(arm,key):
        check()
        admission=gpu_admission(args.gpu)
        if any(a['uuid']!=admission['uuid'] for a in old['admissions']): raise ValueError('GPU changed')
        t,seed=key
        first=off_phases[key]
        ref=Path(first['directory'])
        checkpoint=R7/f'train_{arm}_s{t}'/NN
        command=list(first['command'])
        name=f'{arm}_t{t}_s{seed}_closed_loop'
        directory=output/name
        def replace(flag,value): command[command.index(flag)+1]=str(value)
        replace('--run-dir',directory);replace('--arm',arm)
        replace('--checkpoint',checkpoint);replace('--checkpoint-sha256',inputs[str(checkpoint.resolve())])
        replace('--output',directory/'unused.json');replace('--output_path',directory/'player')
        command+=['--initial',str(ref/'initial_state.pt'),'--trace',str(ref/'trace.pt')]
        if '--replay-actions' in command: raise ValueError('closed-loop actions must never be overridden')
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['uuid'],LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',
                         OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
                         CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED=str(seed),
                         TORCH_EXTENSIONS_DIR=str(reference/'cache/torch_extensions'),XDG_CACHE_HOME=str(output/'cache'))
        environment['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+environment.get('LD_LIBRARY_PATH','')
        phase=dict(name=name,arm=arm,training_seed=t,evaluation_seed=seed,run_status='STARTED',
                   command=command,directory=str(directory),gpu_admission=admission,actions='native deterministic mean')
        manifest['phases'].append(phase);save()
        begin=time.monotonic();process=None
        print(json.dumps(dict(name=name,run_status='STARTED')),flush=True)
        try:
            with (output/(name+'.log')).open('x') as log:
                process=subprocess.Popen(command,cwd=ROOT/'third_party/DExplore',env=environment,
                                         stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                phase.update(pid=process.pid,pgid=process.pid);save()
                code=process.wait(timeout=min(290,3540-prior_seconds-(time.monotonic()-started)))
            if code: raise RuntimeError(f'native exit {code}: {name}')
            result=json.loads((directory/'results.json').read_text())
            if result['run_status']!='COMPLETED' or result['complete_episodes']!=96:
                raise ValueError('incomplete native closed-loop evaluation')
            if result['initial_state_fingerprint']!=off[key]['initial_state_fingerprint']:
                raise ValueError('full initial-state mismatch')
            if result['checkpoint_sha256']!=inputs[str(checkpoint.resolve())]: raise ValueError('wrong actor checkpoint')
            phase.update(run_status='COMPLETED',result_sha256=sha(directory/'results.json'),
                         trace_archive=archive_trace(directory))
            check()
            return result
        except BaseException as error:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
            phase.update(run_status='FAILED',error=repr(error));raise
        finally:
            phase['elapsed_seconds']=time.monotonic()-begin;save()
    save()
    try:
        repeats={key:execute('plain_off',key) for key in PANELS}
        noise=repeatability_gate([e for r in off.values() for e in r['per_episode']],
                                 [e for r in repeats.values() for e in r['per_episode']],
                                 trace_contract_valid=all(r['trace_contract_valid'] and r['actor_and_rms_unchanged'] for r in repeats.values()),
                                 closed_loop_equivalent=True)  # actual native actor, never trace forcing
        noise['repeat_kind']='actual closed-loop off actor; common initial state/RNG; endogenous action trace retained'
        (output/'repeatability.json').write_text(json.dumps(noise,indent=2)+'\n')
        report=dict(run_status='COMPLETED',repeatability=noise,policy_matrix_started=False,
                    label='UNPROMISING',decision='CLOSE_CURRENT_ROUTE_EVALUATOR_RESOLUTION',
                    baseline_status='PARTIAL',cm_utility='UNPROVEN')
        if noise['passed']:
            results={'plain_off':off}
            for arm in ('direct_q','cm_value'):
                results[arm]={key:execute(arm,key) for key in PANELS}
            counts={arm:sum(r['stable_success_count'] for r in panels.values()) for arm,panels in results.items()}
            drops={arm:sum(r['drop_after_success_count'] for r in panels.values()) for arm,panels in results.items()}
            effects={arm:(counts['cm_value']-counts[arm])/384 for arm in ('plain_off','direct_q')}
            nonnegative=all(sum(results['cm_value'][(t,s)]['stable_success_count']-results[arm][(t,s)]['stable_success_count']
                               for s in (288,289))>=0 for t in (286,287) for arm in ('plain_off','direct_q'))
            drop_gate=all((drops['cm_value']-drops[arm])/384<=.05 for arm in ('plain_off','direct_q'))
            intervals={}
            for arm in ('plain_off','direct_q'):
                differences=[]
                for key in PANELS:
                    cm={e['pair_id']:e for e in results['cm_value'][key]['per_episode']}
                    for e in results[arm][key]['per_episode']:
                        differences.append(int(cm[e['pair_id']]['stable_success'])-int(e['stable_success']))
                mean=sum(differences)/384
                se=math.sqrt(sum((d-mean)**2 for d in differences)/(383*384))
                intervals[arm]=dict(mean=mean,approximate_paired95=[mean-1.96*se,mean+1.96*se],scope='descriptive Probe only')
            promising=min(effects.values())>=.05 and nonnegative and drop_gate
            report.update(policy_matrix_started=True,terminal_counts=counts,drop_counts=drops,effects=effects,
                          nonnegative_per_training_seed=nonnegative,drop_gate=drop_gate,paired_intervals=intervals,
                          label='PROMISING' if promising else 'UNPROMISING',
                          decision='REQUIRES_VALIDATION' if promising else 'CLOSE_CURRENT_CM_IMPLEMENTATION')
        if any(sha(Path(p))!=digest for p,digest in inputs.items()): raise ValueError('input drift after completion')
        report.update(input_hashes_unchanged=True,elapsed_seconds=time.monotonic()-started,
                      cumulative_audit_seconds=prior_seconds+time.monotonic()-started,
                      cumulative_output_bytes=prior_bytes+sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
        (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        manifest.update(run_status='COMPLETED',decision=report['decision'],label=report['label'],
                        input_hashes_unchanged=True,elapsed_seconds=time.monotonic()-started)
        save();check();print(json.dumps(report),flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',error=repr(error),elapsed_seconds=time.monotonic()-started)
        save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=4)
    run(p.parse_args())


if __name__=='__main__': main()
