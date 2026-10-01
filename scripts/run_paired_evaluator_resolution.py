#!/usr/bin/env python3
"""Frozen HF08 endpoint audit; inference arms are gated by off repeatability."""
from __future__ import annotations
import argparse
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
BASE=ROOT/'src/task/CmResidual/research/physical_value/output'
R7=BASE/'P-20260930-cm-physical-value/r7'
NN='inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000420.pth'
MOTIONS=Path('/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-baseline/outputs/CmResidual/agent_contact_option_airplane_motions')
PYTHON='/home2/wyy/miniconda3/envs/graspenv/bin/python'
PANELS=((286,288),(286,289),(287,288),(287,289))
ARMS=('plain_off','direct_q','cm_value')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def gpu_admission(index):
    gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    rows={int(line.split(',')[0]):line.split(',') for line in gpus.splitlines()}
    uuid=rows[index][1].strip()
    if int(rows[index][2])>100 or uuid in apps: raise RuntimeError('GPU occupied; no interference allowed')
    return dict(index=index,uuid=uuid,gpus=gpus,compute_apps=apps,time=time.time())


def run(args):
    from src.task.CmResidual.paired_evaluation import repeatability_gate
    output=args.output.absolute()
    if output.parent.resolve()!=BASE.resolve() or output.is_symlink():
        raise ValueError('output must be unique owned task directory')
    output.mkdir(exist_ok=False)
    started=time.monotonic()
    prior_seconds=0
    prior_bytes=0
    prior=[]
    if args.prior_attempt:
        previous=json.loads((args.prior_attempt/'run_manifest.json').read_text())
        if previous['run_status'] not in ('FAILED','STOPPED'):
            raise ValueError('prior attempt still live')
        prior=list(previous.get('prior_attempts',[]))
        prior.append(dict(path=str(args.prior_attempt.resolve()),elapsed_seconds=float(previous['elapsed_seconds']),
                         bytes=sum(p.stat().st_size for p in args.prior_attempt.rglob('*') if p.is_file())))
        if len({p['path'] for p in prior})!=len(prior): raise ValueError('duplicate prior accounting')
        prior_seconds=sum(p['elapsed_seconds'] for p in prior)
        prior_bytes=sum(p['bytes'] for p in prior)
    original_paths=[R7/'environment.yaml',R7/'training.yaml',R7/'results.json']
    original_paths += [R7/f'train_{arm}_s{t}'/NN for arm in ARMS for t in (286,287)]
    original_paths += sorted((R7/'models').glob('*.pt'))
    original_paths += sorted(R7.glob('collect_s28*/transitions_*.pt'))
    route=json.loads((ROOT/'src/task/CmResidual/configs/hf02_temporal_canonical_route.json').read_text())
    for row in route['motions']:
        path=MOTIONS/row['name']/'interaction_hand_inspire.pt'
        if sha(path)!=row['interaction_hand_sha256']: raise ValueError('motion provenance drift')
        original_paths.append(path)
    sources=[ROOT/'scripts/run_paired_physical_value_environment.py',Path(__file__),
             ROOT/'src/task/CmResidual/paired_evaluation.py',ROOT/'src/task/CmResidual/physical_value_contract.py',
             ROOT/'src/task/CmResidual/physical_value_live.py',ROOT/'third_party/DExplore/dexplore/evaluate.py',
             ROOT/'third_party/DExplore/dexplore/env/tasks/base_dexplore_task.py',
             ROOT/'third_party/DExplore/dexplore/env/tasks/base_task.py',
             ROOT/'third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py',
             ROOT/'third_party/DExplore/dexplore/learning/common_player.py']
    hashes={str(p.resolve()):sha(p) for p in original_paths+sources}
    manifest=dict(experiment_id='P-20261001-paired-evaluator-resolution',family='HD02',run_id=args.output.name.rsplit('-',1)[-1],
        run_status='RUNNING',pid=os.getpid(),command=sys.argv,phases=[],input_sha256=hashes,
        original_hf08_inputs_read_only=True,no_training=True,no_new_cm_mechanism=True,
        baseline_status='PARTIAL',wall_limit_seconds=3600,storage_limit_bytes=8<<30,
        prior_attempts=prior,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    def save(): (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if prior_seconds+time.monotonic()-started>3540: raise TimeoutError('whole audit budget')
        if prior_bytes+sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>(8<<30):
            raise RuntimeError('storage budget')
        if any(sha(p)!=hashes[str(p.resolve())] for p in sources): raise RuntimeError('source drift')
    def execute(arm,t,seed,repeat=False,reference=None):
        check()
        admission=gpu_admission(args.gpu)
        manifest.setdefault('admissions',[]).append(admission)
        name=f'{arm}_t{t}_s{seed}_'+('repeat' if repeat else 'first')
        directory=output/name
        checkpoint=R7/f'train_{arm}_s{t}'/NN
        command=[PYTHON,'-u',str(ROOT/'scripts/run_paired_physical_value_environment.py'),
            '--run-dir',str(directory),'--arm',arm,'--training-seed',str(t),'--eval-seed',str(seed),
            '--checkpoint-sha256',hashes[str(checkpoint.resolve())],'--wall-seconds','240',
            '--task','Dexplore_Inspire','--cfg_env',str(R7/'environment.yaml'),'--cfg_train',str(R7/'training.yaml'),
            '--checkpoint',str(checkpoint),'--motion_file',str(MOTIONS),'--headless','--num_envs','96',
            '--seed',str(seed),'--sim_device','cuda:0','--rl_device','cuda:0','--graphics_device_id','0',
            '--disable-early-termination','--output',str(directory/'unused.json'),'--output_path',str(directory/'player')]
        if reference:
            command += ['--initial',str(reference/'initial_state.pt'),'--trace',str(reference/'trace.pt')]
        if repeat: command+=['--replay-actions']
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['uuid'],LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED=str(seed),
            TORCH_EXTENSIONS_DIR=str(output/'cache/torch_extensions'),XDG_CACHE_HOME=str(output/'cache'))
        environment['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+environment.get('LD_LIBRARY_PATH','')
        phase=dict(name=name,arm=arm,training_seed=t,evaluation_seed=seed,repeat=repeat,
                   command=command,run_status='STARTED',directory=str(directory),gpu_uuid=admission['uuid'])
        manifest['phases'].append(phase); save()
        begin=time.monotonic(); process=None
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
                raise ValueError('incomplete native evaluation')
            phase.update(run_status='COMPLETED',result_sha256=sha(directory/'results.json'))
            return directory,result
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
        off={}; repeats={}
        for t,seed in PANELS:
            directory,first=execute('plain_off',t,seed)
            off[(t,seed)]=(directory,first)
            _,second=execute('plain_off',t,seed,repeat=True,reference=directory)
            repeats[(t,seed)]=second
        first_episodes=[e for _,r in off.values() for e in r['per_episode']]
        second_episodes=[e for r in repeats.values() for e in r['per_episode']]
        noise=repeatability_gate(first_episodes,second_episodes,
            trace_contract_valid=all(r['trace_contract_valid'] and r['initial_state_fingerprint']==off[k][1]['initial_state_fingerprint']
                                     for k,r in repeats.items()),
            closed_loop_equivalent=all(r['closed_loop_equivalent'] for r in repeats.values()))
        (output/'repeatability.json').write_text(json.dumps(noise,indent=2)+'\n')
        manifest['repeatability']=noise;save()
        report=dict(run_status='COMPLETED',repeatability=noise,baseline_status='PARTIAL',
                    cm_utility='UNPROVEN',policy_matrix_started=False,label='UNPROMISING',
                    decision='CLOSE_CURRENT_ROUTE_EVALUATOR_RESOLUTION')
        if noise['passed']:
            results={'plain_off':{key:result for key,(_,result) in off.items()}}
            for arm in ('direct_q','cm_value'):
                results[arm]={}
                for t,seed in PANELS:
                    _,result=execute(arm,t,seed,reference=off[(t,seed)][0])
                    results[arm][(t,seed)]=result
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
                intervals[arm]=dict(mean=mean,approximate_paired95=[mean-1.96*se,mean+1.96*se],
                                    boundary='descriptive paired normal interval; Probe, not Validation')
            promising=min(effects.values())>=.05 and nonnegative and drop_gate
            report.update(policy_matrix_started=True,terminal_counts=counts,drop_counts=drops,effects=effects,
                nonnegative_per_training_seed=nonnegative,drop_gate=drop_gate,paired_intervals=intervals,
                label='PROMISING' if promising else 'UNPROMISING',
                decision='REQUIRES_VALIDATION' if promising else 'CLOSE_CURRENT_CM_IMPLEMENTATION')
        if any(sha(Path(p))!=value for p,value in hashes.items()): raise ValueError('original/input drift')
        report.update(input_hashes_unchanged=True,elapsed_seconds=time.monotonic()-started,
            output_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
        (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        manifest.update(run_status='COMPLETED',decision=report['decision'],label=report['label'],
                        input_hashes_unchanged=True,elapsed_seconds=time.monotonic()-started)
        save();check()
        print(json.dumps(report),flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',error=repr(error),elapsed_seconds=time.monotonic()-started)
        save();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=4)
    parser.add_argument('--prior-attempt',type=Path)
    run(parser.parse_args())


if __name__=='__main__': main()
